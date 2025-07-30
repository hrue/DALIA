# Change of Variables for Probability Distributions

This document explains how to compute the probability distribution of a random variable \( X \), when you only have access to a transformation \( Y = g(X) \) and the distribution of \( Y \) is known.

We focus on the case where:
- The transformation \( g \) is **monotonic** and **differentiable**.
- The distribution of \( Y \) is known in **closed form**.
- The goal is to compute the **PDF of \( X \)**.

---

## General Case: Monotonic Transformation

Suppose:
- \( Y = g(X) \), where \( g \) is differentiable and strictly monotonic.
- You know the probability density function (PDF) of \( Y \), denoted \( f_Y(y) \).
- You want to recover the PDF of \( X \), denoted \( f_X(x) \).

### Change of Variables Formula

If \( g \) is invertible and differentiable, the PDF of \( X \) is:

\[
f_X(x) = f_Y(g(x)) \cdot \left| \frac{d}{dx} g(x) \right|
\]

This formula works because:
- \( g(x) \) transforms the variable \( X \) into \( Y \),
- \( \left| \frac{d}{dx} g(x) \right| \) adjusts the density to account for how the transformation stretches or compresses space.

---

## Example: Gaussian Distribution of \( Y \)

Let:
- \( Y = g(X) = \log X \)
- \( Y \sim \mathcal{N}(\mu, \sigma^2) \), a normal distribution with mean \( \mu \) and variance \( \sigma^2 \)

We want to compute the PDF of \( X \).

### Step 1: Invert the Transformation

\[
Y = \log X \quad \Rightarrow \quad X = e^Y
\]

So the inverse transformation is \( g^{-1}(y) = e^y \), and we will evaluate \( f_Y \) at \( g(x) = \log x \).

---

### Step 2: Compute the Derivative of \( g(x) \)

\[
g(x) = \log x \quad \Rightarrow \quad g'(x) = \frac{1}{x}, \quad \text{for } x > 0
\]

---

### Step 3: Apply the Change of Variables Formula

\[
f_X(x) = f_Y(\log x) \cdot \left| \frac{1}{x} \right|
= \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(\log x - \mu)^2}{2\sigma^2} \right) \cdot \frac{1}{x}
\]

---

### Final Answer: Log-Normal Distribution

\[
f_X(x) = \frac{1}{x \sqrt{2\pi \sigma^2}} \exp\left( -\frac{(\log x - \mu)^2}{2\sigma^2} \right), \quad x > 0
\]

This is the **log-normal distribution**, which arises naturally when the logarithm of a variable is normally distributed.

---

## Summary

If \( Y = g(X) \sim \mathcal{N}(\mu, \sigma^2) \), and \( g \) is monotonic and differentiable, then the PDF of \( X \) is:

\[
f_X(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(g(x) - \mu)^2}{2\sigma^2} \right) \cdot \left| g'(x) \right|
\]

This method allows you to compute distributions of variables that are *implicitly defined* via transformations of known random variables.

---
