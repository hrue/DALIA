import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import numpy as np

from pyinla import xp, backend_flags
from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from pyinla.utils import get_host, print_msg, extract_diagonal
from examples_utils.parser_utils import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    print_msg("--- Example: Gaussian spatio-temporal model with regression ---")
    args = parse_args()

    # Configurations of the submodels
    # . Spatio-temporal submodel
    spatio_temporal_dict = {
        "type": "spatio_temporal",
        "input_dir": f"{BASE_DIR}/inputs_spatio_temporal",
        "spatial_domain_dimension": 2,
        "r_s": 0,
        "r_t": 0,
        "sigma_st": 0,
        "manifold": "sphere",
        "ph_s": {"type": "penalized_complexity", "alpha": 0.01, "u": 0.5},
        "ph_t": {"type": "penalized_complexity", "alpha": 0.01, "u": 5},
        "ph_st": {"type": "penalized_complexity", "alpha": 0.01, "u": 3},
    }
    spatio_temporal = SpatioTemporalSubModel(
        config=submodels_config.parse_config(spatio_temporal_dict),
    )
    # . Regression submodel
    regression_dict = {
        "type": "regression",
        "input_dir": f"{BASE_DIR}/inputs_regression",
        "n_fixed_effects": 6,
        "fixed_effects_prior_precision": 0.001,
    }
    regression = RegressionSubModel(
        config=submodels_config.parse_config(regression_dict),
    )

    # Configurations of the likelihood
    likelihood_dict = {
        "type": "gaussian",
        "prec_o": 4,
        # "prior_hyperparameters": {"type": "gaussian", "mean": 1.4, "precision": 0.5},
        "prior_hyperparameters": {
            "type": "penalized_complexity",
            "alpha": 0.01,
            "u": 4,
        },
    }

    # Creation of the model by combining the submodels and the likelihood
    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "serinv"},
        "minimize": {
            "max_iter": args.max_iter,
            "gtol": 1e-3,
            "disp": True,
            "maxcor": len(model.theta),
        },
        "f_reduction_tol": 1e-3,
        "theta_reduction_tol": 1e-4,
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=model,
        config=pyinla_config.parse_config(pyinla_dict),
    )

<<<<<<< HEAD
    # print_msg("\n--- References ---")
    theta_ref = xp.array(np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy"))
    x_ref = xp.array(np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy"))
    
    x_ref_2 = xp.array(np.loadtxt("/home/hpc/ihpc/ihpc060h/pyINLA/dev/sandbox_gaussian_spatioTemporal_verySmall/INLA_DIST_outputs/x_mean_INLA_DIST_466_1.dat"))
    print("x_ref_2:", x_ref_2[:10])
    print("x_ref[:10]:", x_ref[:10])
    
    x_ref_3 = xp.array(np.loadtxt("/home/hpc/ihpc/ihpc060h/b_INLA/src/mean_latent_parameters.txt"))
    
    print("difference in x_ref:", np.linalg.norm(x_ref - x_ref_2))
    print("difference in x_ref_2:", np.linalg.norm(x_ref_2 - x_ref_3))

    
    model.theta = theta_ref
    Qprior = model.construct_Q_prior()
    Q_ref = model.construct_Q_conditional(eta = model.a @ model.x)
    rhs_ref = model.construct_information_vector(eta=model.a @ model.x, x_i = model.x)
    
    pyinla.solver.cholesky(Q_ref, "bta")
    x_ref_check = pyinla.solver.solve(rhs_ref, "bta")
    # print last 6 values of x_ref
    print_msg("x[-6:]:", x_ref[-6:])
    print_msg("x_check[-6:]:", x_ref_check[-6:])
    
    print("x[:10] - x_check[:10]:", x_ref[:10] - x_ref_check[:10])
    
    print("difference in x_ref:", np.linalg.norm(x_ref[:460] - x_ref_check[:460]))
    exit()

=======
>>>>>>> a4e3b95216fb0d343471502ad00844fbef1521c4
    results = pyinla.run()

    print_msg("\n--- Results ---")
    print_msg("Theta values:\n", results["theta"])
    print_msg("Covariance of theta:\n", results["cov_theta"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects :],
    )

    print_msg("\n--- Comparisons ---")
    # Compare hyperparameters
    theta_ref = np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy")
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(results['theta'] - get_host(theta_ref)):.4e}",
    )

    # Compare latent parameters
    x_ref = np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy")
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(results['x'] - get_host(x_ref)):.4e}",
    )

    # Compare marginal variances of latent parameters
    var_latent_params = results["marginal_variances_latent"]
    Qconditional = pyinla.model.construct_Q_conditional(eta=model.a @ model.x)
    Qinv_ref = xp.linalg.inv(Qconditional.toarray())
    print_msg(
        "Norm (marg var latent - ref):    ",
        f"{np.linalg.norm(var_latent_params - xp.diag(Qinv_ref)):.4e}",
    )

    # Compare marginal variances of observations
    # var_obs = pyinla.get_marginal_variances_observations(theta=theta_ref, x_star=x_ref)
    # var_obs_ref = extract_diagonal(model.a @ Qinv_ref @ model.a.T)
    # print_msg(
    #     "Norm (var_obs - var_obs_ref):    ",
    #     f"{xp.linalg.norm(var_obs - var_obs_ref):.4e}",
    # )

    print_msg("\n--- Finished ---")
