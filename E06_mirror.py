# -*- coding: utf-8 -*-
"""Two second-order-indistinguishable information states with different predictability.

Construction
------------
The 1D quadrature rule uses fixed nodes {-2,-1,0,1,2} with weights solved
analytically so that
    mu0 = 1,  mu1 = 0,  mu2 = 1,  mu3 = gamma,  mu4 = 3
exactly (zero mean, unit variance, zero kurtosis, skewness gamma; for |gamma| <= 1
all weights are non-negative). The 3D cloud is the tensor product (125 points), so
the covariance is exact element by element with no sampling noise.

A mirror pair gamma <-> -gamma has identical variance and fourth moment bit for
bit; only the third moment flips sign.

Observables
-----------
(a) Second-order quantities: det Sigma(t) and the characteristic-function modulus
    |phi_t(k)|, expected to be bit-identical across a mirror pair (no second-order
    method separates the two states).
(b) Predictability: the one-sided exceedance probability P(d_x > Delta), expected
    to differ across the pair (it is a function of the distribution shape, not of
    the second moments).

Output: results/E06_mirror.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n01_common as C


def asym_rule_1d(gamma):
    """Fixed nodes {-2,-1,0,1,2}; weights match mu0..mu4 exactly (analytical)."""
    assert abs(gamma) <= 1.0 + 1e-12, "|gamma|<=1 is required for non-negative weights"
    nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    w = np.array([(1.0 - gamma) / 12.0,
                  (1.0 + gamma) / 6.0,
                  0.5,
                  (1.0 - gamma) / 6.0,
                  (1.0 + gamma) / 12.0])
    w = w / w.sum()
    return nodes, w


def check_rule(gamma):
    """Self-check: the first four moments of the rule."""
    nd, w = asym_rule_1d(gamma)
    m = [float(np.sum(w * nd ** k)) for k in range(5)]
    return m


def run_case(gamma, Sigma0, T=3.0, h=0.002, n_snap=31, delta=1.5, kprobe=(0.15, 0.0, 0.0)):
    """Integrate the 125-point cloud and record second-order quantities and exceedance probabilities per snapshot."""
    nodes, weights = asym_rule_1d(gamma)
    X, W = C.build_cloud(nodes, weights, Sigma0)
    W = W / W.sum()
    kn = np.array(kprobe, dtype=float)

    times = np.linspace(0.0, T, n_snap)
    step = int(round(T / h / (n_snap - 1)))
    x = X.copy()
    snaps = []
    for i in range(n_snap):
        if i > 0:
            for _ in range(step):
                x = C.rk4(x, 0.0, h)
        mu = np.sum(W[:, None] * x, axis=0)
        d = x - mu
        cov = (d * W[:, None]).T @ d
        sig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        det = float(np.prod(sig))
        # characteristic-function modulus (depends only on the even part of |p| -> equal for a mirror pair)
        ph = np.abs(np.sum(W * np.exp(1j * (x @ kn))))
        # one-sided exceedance probability (depends on the shape -> differs for a mirror pair)
        dx = d[:, 0]
        P_sup = float(np.sum(W[dx > delta]))
        P_inf = float(np.sum(W[dx < -delta]))
        # third moment (confirms that only the third moment changes)
        m3 = float(np.sum(W * dx ** 3))
        snaps.append(dict(t=float(times[i]), det=det, absphi=float(ph),
                          P_sup=P_sup, P_inf=P_inf, m3=m3))
    return snaps


def main():
    Sigma0 = np.eye(3)

    print("=== rule check (mu0, mu1, mu2, mu3, mu4) ===")
    for g in (0.0, 0.5, 1.0):
        print("  gamma=%4.2f -> %s" % (g, ["%.6f" % v for v in check_rule(g)]))

    out = {"meta": dict(Sigma0=Sigma0.tolist(), T=3.0, h=0.002, delta=1.5,
                        nodes=[-2.0, -1.0, 0.0, 1.0, 2.0]),
           "cases": {}}
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        for sgn, tag in ((+1, "plus"), (-1, "minus")):
            gg = sgn * g
            key = "%s%.2f" % (tag, g)
            out["cases"][key] = run_case(gg, Sigma0)
            print("done", key)

    C.save_json("E06_mirror.json", out)

    # ---- summary: are the second-order quantities bit-identical, and how large is the predictability gap ----
    print("\n=== comparison (t = T) ===")
    print(" gamma |  |D det Sigma|  |D |phi||  |D P_sup|  (rel)  | D m3")
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        a = out["cases"]["plus%.2f" % g][-1]
        b = out["cases"]["minus%.2f" % g][-1]
        dd = abs(a["det"] - b["det"])
        dp = abs(a["absphi"] - b["absphi"])
        dP = a["P_sup"] - b["P_sup"]
        rel = 2.0 * dP / max(a["P_sup"] + b["P_sup"], 1e-30)
        print("  %.2f  |   %.3e   |  %.3e  |   %+.4f    (%+.1f%%)  | %+.4f"
              % (g, dd, dp, dP, 100 * rel, a["m3"] - b["m3"]))


if __name__ == "__main__":
    main()
