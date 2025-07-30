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

# --- New Example: Y = sqrt(X), Y ~ N(mu, sigma^2) ---


def sqrt_pdf(x, mu, sigma):
    """PDF of X when Y = sqrt(X), Y ~ N(mu, sigma^2)."""
    # Inverse: X = Y^2
    # Jacobian: d/dx sqrt(x) = 1/(2*sqrt(x))
    # f_X(x) = f_Y(sqrt(x)) * |1/(2*sqrt(x))|, x > 0
    fy = norm.pdf(np.sqrt(x), loc=mu, scale=sigma)
    jac = 1 / (2 * np.sqrt(x))
    return fy * jac


# Generate x values (x > 0)
x2 = np.linspace(0.01, 9, 500)

# Compute f_Y(sqrt(x))
f_Y_sqrtx = norm.pdf(np.sqrt(x2), loc=mu, scale=sigma)
# Compute |d/dx sqrt(x)| = 1/(2*sqrt(x))
jacobian2 = 1 / (2 * np.sqrt(x2))
# Change of variables formula
f_X2 = f_Y_sqrtx * jacobian2

# Analytical formula for comparison (same as above)
f_X2_analytical = sqrt_pdf(x2, mu, sigma)

# Plot the results
plt.figure(figsize=(7, 4))
plt.plot(x2, f_X2, label="Change of variables (Y = sqrt(X))", lw=2)
plt.plot(x2, f_X2_analytical, "--", label="Analytical (Y = sqrt(X))", lw=2)
plt.title("PDF of X via change of variables (Y = sqrt(X), Y ~ N(0,1))")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
