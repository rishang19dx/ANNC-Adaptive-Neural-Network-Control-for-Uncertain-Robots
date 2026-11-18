import numpy as np
import pytest

from annc.controllers import (AdaptiveNNController, ModelBasedController, PDController,
                              backstep_errors, row_pinv)
from annc.params import ConstraintParams, MBGains, SimParams
from annc.rbf import GaussianRBF
from annc.robot import TwoLinkArm
from annc.signals import ExternalForce, reference
from annc.simulate import Simulation, solve_loop

rng = np.random.default_rng(1)
ARM = TwoLinkArm()
SAMPLES = [(rng.uniform(-np.pi, np.pi, 2), rng.uniform(-2, 2, 2)) for _ in range(20)]


def _grad(fun, x, h=1e-6):
    return np.array([(fun(x + h * e) - fun(x - h * e)) / (2 * h) for e in np.eye(len(x))])


# ---------------- plant: M, C, G consistent with the Lagrangian (eqs. 32-36)
@pytest.mark.parametrize("q,dq", SAMPLES)
def test_inertia_symmetric_pd_and_from_kinetic_energy(q, dq):
    M = ARM.M(q)
    assert np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(M) > 0)
    assert np.isclose(ARM.kinetic_energy(q, dq), 0.5 * dq @ M @ dq)


@pytest.mark.parametrize("q,dq", SAMPLES)
def test_gravity_is_potential_gradient(q, dq):
    assert np.allclose(ARM.G(q), _grad(ARM.potential_energy, q), atol=1e-6)


@pytest.mark.parametrize("q,dq", SAMPLES)
def test_coriolis_matches_lagrange_and_skew_property(q, dq):
    dM = sum(_grad(lambda x: ARM.M(x)[:, :].ravel(), q)[k].reshape(2, 2) * dq[k] for k in range(2))
    dTdq = _grad(lambda x: 0.5 * dq @ ARM.M(x) @ dq, q)
    assert np.allclose(ARM.C(q, dq) @ dq, dM @ dq - dTdq, atol=1e-6)
    N = dM - 2 * ARM.C(q, dq)
    assert np.allclose(N, -N.T, atol=1e-6)


# ---------------- controller building blocks
def test_row_pinv_matches_numpy_and_zero_case():
    z = np.array([0.03, -0.07])
    assert np.allclose(row_pinv(z), np.linalg.pinv(z[None, :]).ravel())
    assert np.all(row_pinv(np.zeros(2)) == 0)
    assert np.isclose(z @ row_pinv(z), 1.0)


def test_alpha_dot_matches_finite_difference():
    cons, k1 = ConstraintParams(), MBGains().k1
    q0, dq, ddq = np.array([0.05, 1.95]), np.array([0.3, 0.2]), np.array([0.1, -0.4])
    state = lambda t: (q0 + dq * t + 0.5 * ddq * t**2, dq + ddq * t)
    _, _, _, dalpha = backstep_errors(*state(0.0), reference(0.0), k1, cons.ka)
    h = 1e-6
    a_p = backstep_errors(*state(h), reference(h), k1, cons.ka)[2]
    a_m = backstep_errors(*state(-h), reference(-h), k1, cons.ka)[2]
    # dalpha uses x2 (not x2_dot), so it is exact for alpha = dqd - A(z1).
    assert np.allclose(dalpha, (a_p - a_m) / (2 * h), atol=1e-6)


@pytest.mark.parametrize("ctrl", [ModelBasedController(ARM), AdaptiveNNController()])
def test_algebraic_loop_solution_satisfies_both_equations(ctrl):
    q, dq, t = np.array([0.08, 2.05]), np.array([0.4, 0.1]), 0.3
    f = ExternalForce(1.0)(t)
    W = ctrl.initial_weights()
    if W is not None:
        W = 0.01 * rng.standard_normal(W.shape)
    law = ctrl(t, q, dq, reference(t), f, W)
    h = ARM.bias(q, dq, f)
    tau, a = solve_loop(law, ARM.M(q), h)
    assert np.allclose(tau, law.tau0 + law.B @ a)
    assert np.allclose(ARM.M(q) @ a, tau - h)


def test_rbf_grid_has_256_nodes():
    rbf = GaussianRBF.grid(8)
    assert rbf.size == 256
    s = rbf(np.zeros(8))
    assert s.shape == (256,) and np.all((s > 0) & (s <= 1))


# ---------------- closed loop (short runs)
def _run(ctrl, tf=4.0, q0=(0.1, 2.1)):
    return Simulation(ARM, ctrl, SimParams(t_final=tf, dt=5e-4, q0=np.array(q0), log_every=4)).run()


def test_mb_control_converges_inside_barrier():
    L = _run(ModelBasedController(ARM))
    ka, kb = ConstraintParams().ka, ConstraintParams().kb
    assert np.all(np.abs(L["z1"]) < ka) and np.all(np.abs(L["z2"]) < kb)
    # Joint 2 converges slowly by design: effective rate k1*ka^2 = 5*0.0289 ~ 0.14 1/s.
    assert np.all(np.abs(L["z1"][-1]) < 0.95 * np.abs(L["z1"][0]))
    assert abs(L["z1"][-1, 0]) < 5e-3


def test_nn_control_respects_barrier_with_initial_error():
    L = _run(AdaptiveNNController())
    ka, kb = ConstraintParams().ka, ConstraintParams().kb
    assert np.all(np.abs(L["z1"]) < ka) and np.all(np.abs(L["z2"]) < kb)
    assert not L["violated"].any()


def test_pd_without_barrier_violates_constraints():
    L = _run(PDController(), q0=(0.0, 2.0))
    assert L["violated"].any()


def test_step_size_convergence():
    a = Simulation(ARM, AdaptiveNNController(), SimParams(t_final=1.0, dt=1e-4, log_every=50)).run()
    b = Simulation(ARM, AdaptiveNNController(), SimParams(t_final=1.0, dt=5e-4, log_every=10)).run()
    assert np.allclose(a["q"], b["q"], atol=1e-3)
