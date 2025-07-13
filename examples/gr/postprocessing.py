# compare results of DALIA with INLA_DIST and generating field


import os

import numpy as np

path = os.path.dirname(__file__)


if __name__ == "__main__":
    n_latent_parameters = 6

    # read in results DALIA
    # f_value_dalia = np.load(f"{path}/outputs/f_value_dalia.npy")
    theta_mean_dalia = np.load(f"{path}/outputs/theta_mean_dalia.npy")
    if len(theta_mean_dalia) == 1:
        theta_mean_dalia = theta_mean_dalia[0]
    x_mean_dalia = np.load(f"{path}/outputs/x_mean_dalia.npy")

    # read in results INLA_DIST
    theta_mean_inla_dist = np.loadtxt(
        f"{path}/INLA_DIST_outputs/theta_mean_INLA_DIST_1_1.dat"
    )
    x_mean_inla_dist = np.loadtxt(
        f"{path}/INLA_DIST_outputs/x_mean_INLA_DIST_{n_latent_parameters}_1.dat"
    )

    # read in results generating field
    x_true = np.load(f"{path}/inputs/x_original.npy")
    theta_true = np.load(f"{path}/inputs/theta_original.npy")

    # compare results of DALIA with INLA_DIST and generating field
    # print("f_value_dalia: ", f_value_dalia)

    print("\ntheta_mean_dalia: ", theta_mean_dalia)
    print("theta_mean_inla_dist: ", theta_mean_inla_dist)
    print("theta_true: ", theta_true)

    print("\nx_mean_dalia: ", x_mean_dalia)
    print("x_mean_inla_dist: ", x_mean_inla_dist)
    print("x_true: ", x_true)

    assert np.allclose(
        theta_mean_dalia, theta_mean_inla_dist, atol=1e-3
    ), "theta_mean_dalia and theta_mean_inla_dist are not close."
    assert np.allclose(
        x_mean_dalia, x_mean_inla_dist, atol=1e-3
    ), "x_mean_dalia and x_mean_inla_dist are not close."

    print("\nResults are consistent!")
