import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for Y ~ N(mu, sigma^2)
mu = 0.0
sigma = 1.0

# Define the transformation: Y = log(X)
# We want to find the PDF of X


def lognormal_pdf(x, mu, sigma):
    """Analytical log-normal PDF."""
    return (1 / (x * np.sqrt(2 * np.pi * sigma**2))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2 * sigma**2)
    )


# Generate x values (x > 0)
x = np.linspace(0.01, 5, 500)

# Compute f_Y(log x)
f_Y_logx = norm.pdf(np.log(x), loc=mu, scale=sigma)
# Compute |d/dx log x| = 1/x
jacobian = 1 / x
# Change of variables formula
f_X = f_Y_logx * jacobian

# Analytical log-normal PDF for comparison
f_X_analytical = lognormal_pdf(x, mu, sigma)

# Plot the results
plt.figure(figsize=(7, 4))
plt.plot(x, f_X, label="Change of variables", lw=2)
plt.plot(x, f_X_analytical, "--", label="Analytical log-normal", lw=2)
plt.title("PDF of X via change of variables (Y = log X, Y ~ N(0,1))")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
