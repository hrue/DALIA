# Poisson likelihood, Regression + Spatio-Temporal submodel example

This example contain a small spatio-temporal model.

--- Model ---
    n_hyperparameters: 3
    n_latent_parameters: 2272
    n_observations: 13656
    likelihood: poisson

--- SpatioTemporalSubModel ---
    ns: 283
    nt: 8
    manifold: plane

--- RegressionSubModel ---
    n_fixed_effects: 8
    fixed_effects_prior_precision: 0.001
