"""PyBullet plant + model wrappers for the controllers in `annc`.

* `BulletArm`  - the simulated *plant* (torque-controlled active joints, other
  joints locked by stiff position servos, external force at the end effector,
  optional payload change, a translucent "ghost" robot showing q_d, video capture).
* `BulletModel` - a *nominal model* in its own DIRECT client exposing the
  M(q) / bias(q, dq, f) interface the model-based controller needs. It never
  sees payload changes made to the plant, so it is a genuinely imperfect model.

Sign convention (paper eq. 2): M a = tau - J^T f - C dq - G, i.e. the robot
feels the force -f, which is what we apply in PyBullet.
"""
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as pb
import pybullet_data


def _resolve(urdf):
    return urdf if os.path.isabs(urdf) or os.path.exists(urdf) else os.path.join(pybullet_data.getDataPath(), urdf)


def urdf_with_extra_mass(urdf, link_name, kg):
    """Copy of `urdf` with `kg` added to <mass> of `link_name` (mesh paths made absolute)."""
    path = _resolve(urdf)
    base = os.path.dirname(os.path.abspath(path))
    tree = ET.parse(path)
    for el in tree.iter():
        fn = el.get("filename")
        if fn and not os.path.isabs(fn) and "://" not in fn:
            el.set("filename", os.path.join(base, fn))
    for link in tree.getroot().iter("link"):
        if link.get("name") == link_name:
            mass = link.find("inertial/mass")
            mass.set("value", str(float(mass.get("value")) + kg))
            break
    else:
        raise ValueError(f"link {link_name!r} not found in {path}")
    fd, out = tempfile.mkstemp(suffix=".urdf", prefix="payload_")
    with os.fdopen(fd, "wb") as fh:
        tree.write(fh)
    return out


def _movable_joints(cid, body):
    return [j for j in range(pb.getNumJoints(body, physicsClientId=cid))
            if pb.getJointInfo(body, j, physicsClientId=cid)[2] in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC)]


class _Dynamics:
    """Shared helpers: full-DOF mass matrix / inverse dynamics restricted to active joints."""

    def _setup(self, cid, body, active, locked_q, ee_link, ee_offset):
        self.cid, self.body = cid, body
        self.dofs = _movable_joints(cid, body)
        self.active = list(active)
        self.ia = [self.dofs.index(j) for j in self.active]   # positions in the DOF vector
        self.q_full0 = np.zeros(len(self.dofs))
        for j, v in (locked_q or {}).items():
            self.q_full0[self.dofs.index(j)] = v
        self.ee_link = ee_link
        self.ee_offset = np.asarray(ee_offset, dtype=float)  # in the URDF link frame
        for j in range(-1, pb.getNumJoints(body, physicsClientId=cid)):
            pb.changeDynamics(body, j, linearDamping=0, angularDamping=0, physicsClientId=cid)

    def _full(self, qa, dqa=None):
        q = self.q_full0.copy(); q[self.ia] = qa
        dq = np.zeros_like(q)
        if dqa is not None:
            dq[self.ia] = dqa
        return q, dq

    def _mass_bias_jac(self, q_full, dq_full):
        cid, body = self.cid, self.body
        M = np.array(pb.calculateMassMatrix(body, q_full.tolist(), physicsClientId=cid))
        h = np.array(pb.calculateInverseDynamics(body, q_full.tolist(), dq_full.tolist(),
                                                 [0.0] * len(q_full), physicsClientId=cid))
        lin, _ = pb.calculateJacobian(body, self.ee_link, self.ee_offset.tolist(), q_full.tolist(),
                                      dq_full.tolist(), [0.0] * len(q_full), physicsClientId=cid)
        J = np.array(lin)[[0, 2]]          # world x and z rows: motion in the vertical x-z plane
        ia = self.ia
        return M[np.ix_(ia, ia)], h[ia], J[:, ia]


class BulletModel(_Dynamics):
    """Nominal rigid-body model (separate DIRECT client) for the MB controller."""

    def __init__(self, urdf, active, locked_q=None, ee_link=-1, ee_offset=(0, 0, 0)):
        cid = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        pb.setGravity(0, 0, -9.81, physicsClientId=cid)
        body = pb.loadURDF(urdf, useFixedBase=True, flags=pb.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
        self._setup(cid, body, active, locked_q, ee_link, ee_offset)

    def _sync(self, qa, dqa=None):
        q, dq = self._full(qa, dqa)
        for k, j in enumerate(self.dofs):
            pb.resetJointState(self.body, j, q[k], dq[k], physicsClientId=self.cid)
        return q, dq

    def M(self, qa):
        q, dq = self._sync(qa)
        return self._mass_bias_jac(q, dq)[0]

    def bias(self, qa, dqa, f):
        q, dq = self._sync(qa, dqa)
        _, h, J = self._mass_bias_jac(q, dq)
        return h + J.T @ f

    def full_model(self, q_full, dq_full):
        """M, h, J (active-joint blocks) at an arbitrary full joint state."""
        for k, j in enumerate(self.dofs):
            pb.resetJointState(self.body, j, q_full[k], dq_full[k], physicsClientId=self.cid)
        return self._mass_bias_jac(q_full, dq_full)

    def close(self):
        pb.disconnect(self.cid)

    def ee_position(self, qa):
        self._sync(qa)
        st = pb.getLinkState(self.body, self.ee_link, computeForwardKinematics=True, physicsClientId=self.cid)
        p, _ = pb.multiplyTransforms(st[4], st[5], self.ee_offset.tolist(), [0, 0, 0, 1])
        return np.array(p)


class BulletArm(_Dynamics):
    """The plant."""

    def __init__(self, urdf, active, locked_q=None, ee_link=-1, ee_offset=(0, 0, 0),
                 dt=1e-3, gui=False, camera=None, ghost=True, floor=False):
        self.gui = gui
        cid = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        pb.setGravity(0, 0, -9.81, physicsClientId=cid)
        pb.setTimeStep(dt, physicsClientId=cid)
        self.dt = dt
        if gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=cid)
        if floor:
            pb.loadURDF("plane.urdf", physicsClientId=cid)
        body = pb.loadURDF(urdf, useFixedBase=True, flags=pb.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)
        self._setup(cid, body, active, locked_q, ee_link, ee_offset)
        # PyBullet's calculateMassMatrix / calculateInverseDynamics do NOT see changeDynamics(mass=...)
        # made after loading, while stepSimulation does. The plant therefore keeps its own "truth"
        # model client, rebuilt from a modified URDF whenever a payload is added.
        self._model_args = (urdf, active, locked_q, ee_link, ee_offset)
        self.truth = BulletModel(*self._model_args)
        self.camera = camera or dict(target=[0.3, 0, 0.2], distance=1.6, yaw=0, pitch=-5)
        if gui:
            pb.resetDebugVisualizerCamera(self.camera["distance"], self.camera["yaw"],
                                          self.camera["pitch"], self.camera["target"], physicsClientId=cid)

        # disable default velocity motors on active joints; lock the rest with stiff servos
        pb.setJointMotorControlArray(body, self.active, pb.VELOCITY_CONTROL,
                                     forces=[0.0] * len(self.active), physicsClientId=cid)
        self.locked = [j for j in self.dofs if j not in self.active]
        if self.locked:
            pb.setJointMotorControlArray(body, self.locked, pb.POSITION_CONTROL,
                                         targetPositions=[self.q_full0[self.dofs.index(j)] for j in self.locked],
                                         forces=[2000.0] * len(self.locked), physicsClientId=cid)

        self.ghost = None
        if ghost:
            g = pb.loadURDF(urdf, basePosition=[0, 0.001, 0], useFixedBase=True, physicsClientId=cid)
            for j in range(-1, pb.getNumJoints(g, physicsClientId=cid)):
                pb.setCollisionFilterGroupMask(g, j, 0, 0, physicsClientId=cid)
                pb.changeVisualShape(g, j, rgbaColor=[0.2, 0.85, 0.3, 0.35], physicsClientId=cid)
            self.ghost = g
            self.ghost_dofs = _movable_joints(cid, g)

    # ---- state
    def reset(self, qa, dqa):
        q, dq = self._full(qa, dqa)
        for k, j in enumerate(self.dofs):
            pb.resetJointState(self.body, j, q[k], dq[k], physicsClientId=self.cid)
        self.set_ghost(qa)

    def state(self):
        st = pb.getJointStates(self.body, self.active, physicsClientId=self.cid)
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def full_state(self):
        st = pb.getJointStates(self.body, self.dofs, physicsClientId=self.cid)
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def plant_model(self, f):
        """True M and h = C dq + G + D dq + J^T f at the current plant state (for the algebraic-loop solve).

        calculateInverseDynamics ignores URDF joint damping, but stepSimulation applies it,
        so it is added here. Without it the "ideal acceleration measurement" is wrong, and the
        (z2^T)^+ w^T a term (gain ~ M^-1 / k_b^2) amplifies the error until the barrier breaks.
        """
        q, dq = self.full_state()
        M, h, J = self.truth.full_model(q, dq)
        damping = np.array([pb.getJointInfo(self.body, j, physicsClientId=self.cid)[6] for j in self.active])
        return M, h + damping * dq[self.ia] + J.T @ f, J

    # ---- actuation
    def step(self, tau, f):
        pb.setJointMotorControlArray(self.body, self.active, pb.TORQUE_CONTROL,
                                     forces=list(tau), physicsClientId=self.cid)
        st = pb.getLinkState(self.body, self.ee_link, computeForwardKinematics=True, physicsClientId=self.cid)
        p, _ = pb.multiplyTransforms(st[4], st[5], self.ee_offset.tolist(), [0, 0, 0, 1])
        F = [-f[0], 0.0, -f[1]]            # robot feels -f (eq. 2)
        pb.applyExternalForce(self.body, self.ee_link, F, p, pb.WORLD_FRAME, physicsClientId=self.cid)
        pb.stepSimulation(physicsClientId=self.cid)

    def set_ghost(self, qa):
        if self.ghost is None:
            return
        q, _ = self._full(qa)
        for k, j in enumerate(self.ghost_dofs):
            pb.resetJointState(self.ghost, j, q[k], 0.0, physicsClientId=self.cid)

    def add_payload(self, kg, link=None):
        link = self.ee_link if link is None else link
        info = pb.getDynamicsInfo(self.body, link, physicsClientId=self.cid)
        # Point mass at the link COM: mass changes, inertia about the COM does not. Passing the
        # inertia explicitly matters - otherwise PyBullet recomputes it from the collision shape
        # (≈0 for links without one), which would silently change the plant.
        pb.changeDynamics(self.body, link, mass=info[0] + kg, localInertiaDiagonal=info[2],
                          physicsClientId=self.cid)
        name = pb.getJointInfo(self.body, link, physicsClientId=self.cid)[12].decode()
        urdf, *rest = self._model_args
        new_urdf = urdf_with_extra_mass(urdf, name, kg)
        self.truth.close()
        self.truth = BulletModel(new_urdf, *rest)
        os.remove(new_urdf)                       # loadURDF has already parsed it
        self._model_args = (urdf, *rest)
        self._extra_mass = getattr(self, "_extra_mass", 0.0) + kg
        pb.changeVisualShape(self.body, link, rgbaColor=[0.85, 0.1, 0.1, 1], physicsClientId=self.cid)

    # ---- rendering
    def frame(self, width=640, height=480):
        c = self.camera
        view = pb.computeViewMatrixFromYawPitchRoll(c["target"], c["distance"], c["yaw"], c["pitch"], 0, 2,
                                                    physicsClientId=self.cid)
        proj = pb.computeProjectionMatrixFOV(50, width / height, 0.05, 10, physicsClientId=self.cid)
        renderer = pb.ER_BULLET_HARDWARE_OPENGL if self.gui else pb.ER_TINY_RENDERER
        img = pb.getCameraImage(width, height, view, proj, renderer=renderer, physicsClientId=self.cid)
        return np.reshape(np.array(img[2], dtype=np.uint8), (height, width, 4))[:, :, :3]

    def close(self):
        self.truth.close()
        pb.disconnect(self.cid)
