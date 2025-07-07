import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from pyinla import xp

from pyinla.configs import likelihood_config, pyinla_config, submodels_config
from pyinla.core.model import Model
from pyinla.core.pyinla import PyINLA
from pyinla.submodels import RegressionSubModel, SpatioTemporalSubModel
from examples_utils.parser_utils import parse_args
from pyinla.utils import print_msg, get_host

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
        "r_s": -0.960279229160082,
        "r_t": -0.3068528194400548,
        "sigma_st": -2.112085713764618,
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
        "prior_hyperparameters": {
            "type": "penalized_complexity",
            "alpha": 0.01,
            "u": 5,
        },    }

    # Creation of the model by combining the submodels and the likelihood
    model = Model(
        submodels=[regression, spatio_temporal],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )
    print_msg(model)

    # Configurations of PyINLA
    pyinla_dict = {
        "solver": {"type": "serinv", "min_processes": args.solver_min_p},
        "minimize": {"max_iter": args.max_iter, "gtol": 1e-1, "disp": True},
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    pyinla = PyINLA(
        model=model,
        config=pyinla_config.parse_config(pyinla_dict),
    )
    
    # print_msg("\n--- References ---")
    theta_ref = xp.array(np.load(f"{BASE_DIR}/reference_outputs/theta_ref.npy"))
    x_ref = xp.array(np.load(f"{BASE_DIR}/reference_outputs/x_ref.npy"))

    results = pyinla.run()
    
    print_msg("\n--- Results ---")
    print_msg("Theta values:\n", results["theta"])
    print_msg("Covariance of theta:\n", results["cov_theta"])
    print_msg(
        "Mean of the fixed effects:\n",
        results["x"][-model.submodels[-1].n_fixed_effects:],
    )

    print_msg("\n--- Comparisons ---")

    # Compare hyperparameters
    print_msg(
        "Norm (theta - theta_ref):        ",
        f"{np.linalg.norm(results['theta'] - get_host(theta_ref)):.4e}",
    )
    
    # Compare latent parameters
    print_msg(
        "Norm (x - x_ref):                ",
        f"{np.linalg.norm(results['x'] - get_host(x_ref)):.4e}",
    )

    print_msg("\n--- Finished ---")