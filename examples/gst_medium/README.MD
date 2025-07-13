# Gaussian likelihood, Regression + Spatio-Temporal submodel example

This example contain a medium spatio-temporal model.

--- Model ---
    n_hyperparameters: 4
    n_latent_parameters: 81206
    n_observations: 162400
    likelihood: gaussian

--- SpatioTemporalSubModel ---
    ns: 812
    nt: 100
    manifold: plane
    
--- RegressionSubModel ---
    n_fixed_effects: 6
    fixed_effects_prior_precision: 0.001
