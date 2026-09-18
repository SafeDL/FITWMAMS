from __future__ import annotations
import numpy as np
from bayesian_ma_idm.src.driver import BayesianIDMDriver
from bayesian_ma_idm.src.reference_kernels import conditional_gp, se_kernel

def test_gp_conditional_matches_direct_multivariate_normal_formula() -> None:
    times=np.array([0.,.2,.4,.6]); residual=np.array([.1,-.2,.3,.05]); sigma=0.7; ell=.45; target=.8
    mean,var=conditional_gp(times,residual,target,sigma,ell,observation_noise=0.)
    k=se_kernel(times,times,sigma,ell);cross=se_kernel(np.array([target]),times,sigma,ell)[0]
    assert abs(mean-cross@np.linalg.solve(k,residual))<1e-8
    assert abs(var-(sigma**2-cross@np.linalg.solve(k,cross)))<1e-8

def test_conditional_gp_sampling_recovers_analytic_moments() -> None:
    mean, variance = conditional_gp(np.array([0., .2, .4]), np.array([.2, -.1, .1]), .6, .5, .3)
    values = mean + np.sqrt(variance) * np.random.default_rng(3).standard_normal(10_000)
    assert abs(values.mean() - mean) < 4 * np.sqrt(variance / len(values))
    assert abs(values.var() - variance) < .05 * variance


def test_online_driver_is_reproducible_and_keeps_gp_state() -> None:
    theta = np.asarray([33.0, 2.0, 1.6, 1.5, 1.67])
    driver = BayesianIDMDriver(
        "ma_idm", theta, 0.2, lengthscale_s=1.5, iid_sigma=0.05
    )
    first = driver.decision(
        gap_m=25.0,
        speed_mps=20.0,
        leader_speed_mps=19.0,
        process_standard_normal=0.4,
        iid_standard_normal=-0.2,
    )
    driver.decision(
        gap_m=24.5,
        speed_mps=19.8,
        leader_speed_mps=19.0,
        process_standard_normal=0.4,
        iid_standard_normal=-0.2,
    )
    assert len(driver._gp.residuals) == 2
    driver.reset(seed=7)
    repeated = driver.decision(
        gap_m=25.0,
        speed_mps=20.0,
        leader_speed_mps=19.0,
        process_standard_normal=0.4,
        iid_standard_normal=-0.2,
    )
    assert repeated == first
