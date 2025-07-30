# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import NDArray, xp


# sigmoid (inverse logit) function
def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + xp.exp(-x))


def jacobian_sigmoid(x: NDArray) -> NDArray:
    """Jacobian of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def cloglog(x: NDArray, direction: str) -> NDArray:
    if direction == "forward":
        return xp.log(-xp.log(1 - x))
    elif direction == "backward":
        return 1 - xp.exp(-xp.exp(x))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def scaled_logit(x: NDArray, direction: str) -> NDArray:
    k = 1.0 / 12.0
    if direction == "forward":
        ## TODO: check special function log1p
        return (1.0 / k) * xp.log(x / (1.0 - x))
    elif direction == "backward":
        return 1 / (1 + xp.exp(-k * x))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def jacobian_scaled_logit(x: NDArray, direction: str) -> NDArray:
    k = 1.0 / 12.0
    if direction == "forward":
        return (1.0 / k) * (1.0 / x + 1.0 / (1.0 - x))
    elif direction == "backward":
        s = scaled_logit(x, "backward")
        return k * s * (1 - s)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def identity(x: NDArray, direction: str) -> NDArray:
    """Identity function."""
    if direction == "forward":
        return x
    elif direction == "backward":
        return x
    else:
        raise ValueError(f"Unknown direction: {direction}")


def jacobian_identity(x: NDArray, direction: str) -> NDArray:
    """Jacobian of the identity function."""
    if direction == "forward":
        return xp.ones_like(x)
    elif direction == "backward":
        return xp.ones_like(x)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def log(x: NDArray, direction: str) -> NDArray:
    """Logarithm function."""
    if direction == "forward":
        return xp.log(x)
    elif direction == "backward":
        return xp.exp(x)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def jacobian_log(x: NDArray, direction: str) -> NDArray:
    """Jacobian of the logarithm function."""
    if direction == "forward":
        return 1 / x
    elif direction == "backward":
        return xp.exp(x)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def evaluate_gaussian(
    theta: NDArray,
    mean: NDArray,
    precision: NDArray,
) -> NDArray:
    """Evaluate the Gaussian distribution."""

    pdf = (
        1
        / (xp.sqrt(2 * xp.pi / precision))
        * xp.exp(-0.5 * precision * (theta - mean) ** 2)
    )

    return pdf


def fit_and_normalize_spline(x, y, s=0, ext=1, n_points=500):
    """
    Fit a univariate spline through (x, y), normalize it so its integral is 1,
    and return the normalized spline function and the normalization constant.
    """
    from scipy.interpolate import UnivariateSpline
    from scipy.integrate import quad
    import numpy as np

    # Sort x and y for monotonicity (required for spline)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Fit spline
    spline = UnivariateSpline(x_sorted, y_sorted, s=s, ext=ext)

    # Compute normalization constant
    area, _ = quad(spline, x_sorted[0], x_sorted[-1])

    def normalized_spline(z):
        return spline(z) / area

    return normalized_spline, area


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)  # For reproducibility
    # generate random 1-dimensional mean between -5 and 5
    theta_mean = np.random.uniform(-5, 5)
    print(f"Mean: {theta_mean}")

    # generate random 1-dimensional precision between 0.1 and 3
    theta_precision = np.random.uniform(0.1, 3)
    print(f"Precision: {theta_precision}")

    # theta_transform = scaled_logit
    # theta_transform_jacobian = jacobian_scaled_logit

    theta_transform = log
    theta_transform_jacobian = jacobian_log

    # construct normal distribution
    interval = (
        theta_mean - 3 / np.sqrt(theta_precision),
        theta_mean + 3 / np.sqrt(theta_precision),
    )

    # create equidistant grid in the interval
    grid = np.linspace(interval[0], interval[1], 100)
    grid_y = theta_transform(grid, "backward")
    print(f"Grid y: Min:", grid_y.min(), ".Max:", grid_y.max())
    # compute pdf on the grid
    pdf = evaluate_gaussian(grid, theta_mean, theta_precision)

    # compute f_Y
    jacobian_y = theta_transform_jacobian(grid_y, "forward")
    pdf_y = evaluate_gaussian(grid, theta_mean, theta_precision) * jacobian_y

    # plot the pdfs side by side, plus log-normal
    import matplotlib.pyplot as plt
    import scipy.stats

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(grid, pdf, label="PDF")
    axes[0].plot(
        grid,
        scipy.stats.norm.pdf(grid, loc=theta_mean, scale=1 / np.sqrt(theta_precision)),
        label="SciPy PDF",
        linestyle="dashed",
    )
    axes[0].set_title("Normal Distribution PDF")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].plot(grid_y, pdf_y, label="Transformed PDF")
    axes[1].set_title("Transformed PDF")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # Fit and plot normalized spline on the transformed PDF plot
    normalized_spline, area = fit_and_normalize_spline(grid_y, pdf_y)
    print(f"Spline area before normalization: {area}")
    y_dense = np.linspace(np.min(grid_y), np.max(grid_y), 500)
    axes[1].plot(
        y_dense,
        normalized_spline(y_dense),
        label="Normalized Spline",
        linestyle="dotted",
    )
    axes[1].legend()

    # Log-normal: s = stddev of log, scale = exp(mean of log)
    s = 1 / np.sqrt(theta_precision)
    scale = np.exp(theta_mean)
    x_lognorm = np.linspace(np.exp(grid[0]), np.exp(grid[-1]), 100)
    pdf_lognorm = scipy.stats.lognorm.pdf(x_lognorm, s=s, scale=scale)
    axes[2].plot(x_lognorm, pdf_lognorm, label="Log-normal PDF")
    axes[2].set_title("Log-normal PDF")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Density")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    ## we want to return for each marginal distribution
    # the "grid_y" and the "pdf_y" (transformed pdf) values
