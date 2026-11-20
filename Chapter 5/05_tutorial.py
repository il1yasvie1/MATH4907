"""Reproduce the numerical answers and figures for MATH4907 Tutorial 5.

Run from this directory with Python, NumPy, SciPy, and Matplotlib installed.
The random streams are fixed separately for each question.
"""

from __future__ import annotations

from math import erf, exp, gamma, log, pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)


def ci(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean, sample SD, estimated SE, and 95% half-width."""
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / sqrt(len(values))
    return mean, sd, se, 1.96 * se


def hist_with_density(
    values: np.ndarray, x: np.ndarray, density: np.ndarray, name: str, label: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.hist(values, bins=50, density=True, color="#5C83B8", edgecolor="white",
            linewidth=0.35, alpha=0.85, label="Simulated density")
    ax.plot(x, density, color="#A21E25", linewidth=2.2, label=label)
    ax.set(xlabel="$x$", ylabel="Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180)
    plt.close(fig)


def latex_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    # Q1: this deliberately small LCG has period four.
    state = 5
    states = []
    for _ in range(20):
        state = (7 * state + 3) % 16
        states.append(state)
    uniforms = np.asarray(states) / 16
    transformed = 2 + 3 * uniforms

    # Q2: exponential distribution with mean 2.
    rng = np.random.default_rng(490702)
    exponential = -2 * np.log1p(-rng.random(10_000))
    q2_mean, q2_sd, _, _ = ci(exponential)
    grid = np.linspace(0, 16, 500)
    hist_with_density(exponential, grid, 0.5 * np.exp(-grid / 2),
                      "q2_exponential.png", r"$f(x)=\frac{1}{2}e^{-x/2}$")

    # Q3: draw until exactly 10,000 proposals have been accepted.
    rng = np.random.default_rng(490703)
    accepted: list[float] = []
    proposals = 0
    while len(accepted) < 10_000:
        x_proposal = sqrt(rng.random())
        proposals += 1
        if rng.random() <= x_proposal:
            accepted.append(x_proposal)
    accepted_array = np.asarray(accepted)
    grid = np.linspace(0, 1, 500)
    hist_with_density(accepted_array, grid, 3 * grid**2,
                      "q3_rejection.png", r"$f(x)=3x^2$")

    # Q4: nested prefixes make the sample-size comparison transparent.
    rng = np.random.default_rng(490704)
    integrand = np.exp(-rng.random(100_000)**2)
    integral_exact = sqrt(pi) * erf(1) / 2
    q4 = {n: ci(integrand[:n]) for n in (100, 1_000, 10_000, 100_000)}

    # Q5: same nested sequence for 10,000 and 100,000 paths.
    s0, strike, rate, sigma, maturity = 95.0, 100.0, 0.01, 0.4, 1.0
    drift = (rate - 0.5 * sigma**2) * maturity
    vol = sigma * sqrt(maturity)
    discount = exp(-rate * maturity)
    rng = np.random.default_rng(490705)
    z = rng.standard_normal(100_000)
    terminal = s0 * np.exp(drift + vol * z)
    discounted_payoff = discount * np.maximum(terminal - strike, 0)
    q5 = {n: ci(discounted_payoff[:n]) for n in (10_000, 100_000)}
    normal_cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    d1 = (log(s0 / strike) + (rate + 0.5 * sigma**2) * maturity) / vol
    d2 = d1 - vol
    call_exact = s0 * normal_cdf(d1) - strike * discount * normal_cdf(d2)

    # Q6: one observation is one discounted pair average.
    rng = np.random.default_rng(490706)
    z_pairs = rng.standard_normal(10_000)
    terminal_plus = s0 * np.exp(drift + vol * z_pairs)
    terminal_minus = s0 * np.exp(drift - vol * z_pairs)
    pairs = 0.5 * discount * (np.maximum(terminal_plus - strike, 0)
                              + np.maximum(terminal_minus - strike, 0))
    q6 = ci(pairs)
    crude_same_z = discount * np.maximum(terminal_plus - strike, 0)
    q6_crude_same_z = ci(crude_same_z)

    # Q7: same X_i are used in crude and adjusted estimators.
    rng = np.random.default_rng(490707)
    x = -np.log1p(-rng.random(10_000))
    h = x**0.9
    coefficient = -float(np.cov(h, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    adjusted = h + coefficient * (x - 1)
    q7_crude, q7_cv = ci(h), ci(adjusted)
    q7_exact = gamma(1.9)
    q7_c_exact = -0.9 * q7_exact

    rows = []
    def macro(name: str, value: str) -> None:
        rows.append(rf"\newcommand{{\{name}}}{{{value}}}")

    macro("QOneStates", ", ".join(map(str, states)))
    macro("QOneUniforms", ", ".join(f"{v:.4f}" for v in uniforms))
    macro("QOneTransformed", ", ".join(f"{v:.4f}" for v in transformed))
    macro("QTwoMean", latex_float(q2_mean))
    macro("QTwoVariance", latex_float(q2_sd**2))
    macro("QThreeProposals", str(proposals))
    macro("QThreeEfficiency", latex_float(10_000 / proposals))
    macro("QFourExact", latex_float(integral_exact, 9))
    for n, label in ((100, "A"), (1_000, "B"), (10_000, "C"), (100_000, "D")):
        mean, sd, se, half = q4[n]
        macro(f"QFour{label}Mean", latex_float(mean))
        macro(f"QFour{label}SD", latex_float(sd))
        macro(f"QFour{label}Low", latex_float(mean - half))
        macro(f"QFour{label}High", latex_float(mean + half))
    macro("QFiveExact", latex_float(call_exact))
    for n, label in ((10_000, "A"), (100_000, "B")):
        mean, sd, se, half = q5[n]
        macro(f"QFive{label}Mean", latex_float(mean))
        macro(f"QFive{label}SD", latex_float(sd))
        macro(f"QFive{label}SE", latex_float(se))
        macro(f"QFive{label}Low", latex_float(mean - half))
        macro(f"QFive{label}High", latex_float(mean + half))
    for name, result in (("QSixAV", q6), ("QSixCrude", q6_crude_same_z)):
        mean, sd, se, half = result
        macro(name + "Mean", latex_float(mean))
        macro(name + "SD", latex_float(sd))
        macro(name + "SE", latex_float(se))
        macro(name + "Low", latex_float(mean - half))
        macro(name + "High", latex_float(mean + half))
    macro("QSevenCoeff", latex_float(coefficient))
    macro("QSevenExact", latex_float(q7_exact))
    macro("QSevenExactCoeff", latex_float(q7_c_exact))
    for name, result in (("QSevenCrude", q7_crude), ("QSevenCV", q7_cv)):
        mean, sd, se, half = result
        macro(name + "Mean", latex_float(mean))
        macro(name + "SD", latex_float(sd))
        macro(name + "SE", latex_float(se))
    (HERE / "05_results.tex").write_text("\n".join(rows) + "\n")

    print("Q1 states:", states)
    print("Q2 mean/variance:", q2_mean, q2_sd**2)
    print("Q3 proposals/efficiency:", proposals, 10_000 / proposals)
    print("Q4 exact:", integral_exact, "results:", q4)
    print("Q5 exact:", call_exact, "results:", q5)
    print("Q6 antithetic:", q6, "same-Z crude:", q6_crude_same_z)
    print("Q7 exact:", q7_exact, "c*:", q7_c_exact,
          "c_hat:", coefficient, "crude:", q7_crude, "CV:", q7_cv)


if __name__ == "__main__":
    main()
