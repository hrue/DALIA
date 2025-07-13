# Gaussian likelihood, Regression + Spatio-Temporal submodel example

This example contain a large spatio-temporal model.

--- Model ---
    n_hyperparameters: 4
    n_latent_parameters: 1000506
    n_observations: 2001000
    likelihood: gaussian

--- SpatioTemporalSubModel ---
    ns: 4002
    nt: 250
    manifold: plane
 
--- RegressionSubModel ---
    n_fixed_effects: 6
    fixed_effects_prior_precision: 0.001
