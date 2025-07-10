import os
import time

# import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as scsp

from dalia.configs import likelihood_config, dalia_config, submodels_config
from dalia.core.model import Model
from dalia.core.dalia import DALIA
from dalia.submodels import BrainiacSubModel
from dalia.utils import xp, print_msg

path = os.path.dirname(__file__)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__)) + "/inputs_brainiac_cmPRS"

    m = 2  # number of annotations per feature
    b = 1000  # number of latent variables / number of features
    sigma_a2 = 1.0 / 1.0
    precision_mat = sigma_a2 * scsp.eye(m)

    theta_ref = xp.load(f"{base_dir}/theta_original.npy")
    x_ref = np.load(f"{base_dir}/beta_original.npy")

    initial_h2 = theta_ref[0]
    initial_alpha = theta_ref[1:]

    brainiac_dict = {
        "type": "brainiac",
        "input_dir": f"{base_dir}/inputs_brainiac",
        "h2": initial_h2,
        "alpha": initial_alpha,
        "ph_h2": {"type": "beta", "alpha": 5.0, "beta": 1.0},
        "ph_alpha": {
            "type": "gaussian_mvn",
            "mean": theta_ref[1:],
            "precision": precision_mat,  # sp.sparse.csc_matrix(precision_mat),
        },
    }
    brainiac = BrainiacSubModel(
        config=submodels_config.parse_config(brainiac_dict),
    )
    print(brainiac)

    print("SubModel initialized.")

    likelihood_dict = {"type": "gaussian", "fix_hyperparameters": True}
    model = Model(
        submodels=[brainiac],
        likelihood_config=likelihood_config.parse_config(likelihood_dict),
    )

    print(model)

    print("Model initialized.")

<<<<<<< HEAD
    print("model.theta", model.theta)
    print("length(model.theta)", len(model.theta))
    print("model.theta_keys", model.theta_keys)

    eta = np.ones((model.n_observations, 1))

    model.construct_Q_prior()
    model.construct_Q_conditional(eta)

    # compare to reference solution
    Qprior_ref = sp.load_npz(f"{base_dir}/inputs_brainiac/Qprior_original.npz")
    Qcond_ref = sp.load_npz(f"{base_dir}/inputs_brainiac/Qconditional_original.npz")

    print("Qcond_ref\n", Qcond_ref.toarray())
    print("Qcond\n", model.Q_conditional.toarray())

    print(
        "norm(Qprior_ref - model.Q_prior) = ",
        np.linalg.norm((Qprior_ref - model.Q_prior).toarray()),
    )
    print(
        "norm(Qcond_ref - model.Q_conditional) = ",
        np.linalg.norm((Qcond_ref - model.Q_conditional).toarray()),
    )

    # Q_prior_dense = model.Q_prior.todense()
    # print("Q_prior_dense\n", Q_prior_dense)
    # Q_cond_dense = model.Q_conditional.todense()
    # print("Q_cond_dense\n", Q_cond_dense)

    # plt.matshow(Q_prior_dense)
    # plt.suptitle("Q_prior from brainiac model")
    # plt.savefig("Q_prior.png")

    # plt.matshow(Q_cond_dense)
    # plt.suptitle("Q_conditional from brainiac model")
    # plt.savefig("Q_conditional.png")

    dalia_dict = {
        # "solver": {"type": "serinv"},
=======
    pyinla_dict = {
>>>>>>> 52192c0 (light cleanup)
        "solver": {"type": "dense"},
        "minimize": {
            "max_iter": 50,
            "gtol": 1e-3,
            "disp": True,
        },
        "inner_iteration_max_iter": 50,
        "eps_inner_iteration": 1e-3,
        "eps_gradient_f": 1e-3,
        "simulation_dir": ".",
    }
    dalia = DALIA(
        model=model,
        config=dalia_config.parse_config(dalia_dict),
    )

<<<<<<< HEAD
    print("x ref: ", x_ref)
    # minimization_result = dalia.minimize()

    # output = dalia._evaluate_f(model.theta)
    # x = model.x
    # print("x: ", x)

    print("\n------ Compare to reference solution ------\n")
    # load reference solution
    # theta_ref = np.load(f"{base_dir}/inputs_brainiac/theta_original.npy")

    # x_ref = np.load(f"{base_dir}/inputs_brainiac/beta_original.npy")
    # x = minimization_result["x"]
    # print("\nx    ", x)
    # print("x_ref", x_ref)
    # print("norm(x_ref - x) = ", np.linalg.norm(x_ref - x))

    results = dalia.run()

=======
    tic = time.time()
    minimization_result = pyinla.run()
    toc = time.time()
    print("Elapsed time pyinla.run(): ", toc - tic)

    print("\n------ Compare to reference solution ------\n")
>>>>>>> 52192c0 (light cleanup)
    print("theta_ref: ", theta_ref)

    theta = minimization_result["theta_interpret"]
    print("theta_interpret:", theta)

    x = minimization_result["x"]
    print("norm(x_ref - x) = ", np.linalg.norm(x_ref - x))

    # print Hessian mode

    # marginal variances latent parameters
