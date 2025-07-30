import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# Parameters for Y ~ N(mu, sigma^2)
mu = 5.0
sigma = 1.0

# --- Improved grid: choose y values covering 99% of probability mass, then map to x ---
# For Y = log(X):
ymin, ymax = norm.ppf([0.005, 0.995], loc=mu, scale=sigma)
y_grid = np.linspace(ymin, ymax, 500)
x = np.exp(y_grid)  # x > 0

# Compute f_Y(log x) and jacobian for new grid
f_Y_logx = norm.pdf(np.log(x), loc=mu, scale=sigma)
jacobian = 1 / x
f_X = f_Y_logx * jacobian
# Use scipy's lognorm as reference
f_X_reference = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

plt.figure(figsize=(7, 4))
plt.plot(x, f_X, label="Change of variables (improved grid)", lw=2)
plt.plot(x, f_X_reference, "--", label="scipy.stats.lognorm", lw=2)
plt.title("PDF of X via change of variables (Y = log X, Y ~ N(0,1))")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

area_f_X = np.trapezoid(f_X, x)
area_f_X_reference = np.trapezoid(f_X_reference, x)
print(
    f"[Improved grid] Normalization (numerical): Change of variables (Y=log X): {area_f_X:.6f}"
)
print(
    f"[Improved grid] Normalization (numerical): scipy.stats.lognorm: {area_f_X_reference:.6f}"
)

# For Y = sqrt(X):
ymin2, ymax2 = norm.ppf([0.005, 0.995], loc=mu, scale=sigma)
y_grid2 = np.linspace(ymin2, ymax2, 500)
x2 = y_grid2**2  # x > 0

# For consistency, use y_grid2 directly for f_Y_sqrtx
f_Y_sqrtx = norm.pdf(y_grid2, loc=mu, scale=sigma)
jacobian2 = 1 / (2 * np.sqrt(x2))
f_X2 = f_Y_sqrtx * jacobian2
# No closed-form reference in scipy for this case, so just plot

plt.figure(figsize=(7, 4))
plt.plot(x2, f_X2, label="Change of variables (improved grid)", lw=2)
plt.title("PDF of X via change of variables (Y = sqrt(X), Y ~ N(0,1))")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

area_f_X2 = np.trapezoid(f_X2, x2)
print(
    f"[Improved grid] Normalization (numerical): Change of variables (Y=sqrt X): {area_f_X2:.6f}"
)
