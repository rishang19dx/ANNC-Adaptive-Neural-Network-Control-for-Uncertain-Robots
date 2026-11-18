"""PyBullet cross-checks (skipped when pybullet is not installed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

pb = pytest.importorskip("pybullet")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demos"))

from bullet_arm import BulletArm, BulletModel  # noqa: E402

from annc.robot import TwoLinkArm  # noqa: E402

PLANAR = str(ROOT / "demos" / "planar_2link.urdf")
ROBOTS = [(PLANAR, [0, 1], 1, (0.31, 0, 0)), ("kuka_iiwa/model.urdf", [1, 3], 6, (0, 0, 0.045))]


def test_planar_urdf_matches_paper_model():
    """PyBullet's rigid-body model of the URDF equals eqs. (34)-(37) built from Table I."""
    m, a = BulletModel(PLANAR, [0, 1], ee_link=1, ee_offset=(0.31, 0, 0)), TwoLinkArm()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, dq, f = rng.uniform(-3, 3, 2), rng.uniform(-2, 2, 2), rng.uniform(-3, 3, 2)
        assert np.allclose(m.M(q), a.M(q), atol=1e-10)
        assert np.allclose(m.bias(q, dq, f), a.bias(q, dq, f), atol=1e-10)
        assert np.allclose(m.ee_position(q)[[0, 2]], a.fk(q), atol=1e-6)
    m.close()


@pytest.mark.parametrize("urdf,active,link,offset", ROBOTS)
@pytest.mark.parametrize("payload", [0.0, 0.5])
def test_plant_model_predicts_simulated_acceleration(urdf, active, link, offset, payload):
    """The model used for the algebraic-loop solve must match what stepSimulation does,
    including joint damping and payload changes (both silently missed by PyBullet's
    calculateInverseDynamics / calculateMassMatrix otherwise)."""
    p = BulletArm(urdf, active, {}, link, offset, dt=5e-4, ghost=False)
    p.reset(np.array([0.3, -0.5]), np.array([0.4, -0.6]))
    if payload:
        p.add_payload(payload)
    q, dq = p.state()
    M, h, _ = p.plant_model(np.zeros(2))
    a_cmd = np.array([1.0, -2.0])
    p.step(h + M @ a_cmd, np.zeros(2))
    _, dq2 = p.state()
    assert np.allclose((dq2 - dq) / p.dt, a_cmd, atol=0.05)
    p.close()
