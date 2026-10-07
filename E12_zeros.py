# -*- coding: utf-8 -*-
"""Analytic comparison of zero distance and truncation validity (Orszag--McLaughlin).

The moment-truncation criterion requires kcut < kstar, where kstar is the distance
to the nearest zero of the characteristic function. On L63 the poles lie far
outside the observation band and cannot be reached. The Orszag--McLaughlin model
instead has a Hopf exact solution for the characteristic function of its invariant
measure,
    Psi_0(u) = Gamma(N/2) (R|u|/2)^{1-N/2} J_{N/2-1}(R|u|),
whose zeros are Bessel zeros and analytically known: kstar = j_{N/2-1,1} / R.

The criterion is therefore directly testable: if the truncation error is only a
function of the dimensionless ratio kcut/kstar, then data for different (N, R)
collapse onto a single curve and diverge as the ratio -> 1.

Method (purely analytic, no time integration)
---------------------------------------------
1. Write Psi_0 as a power series in z = k^2 (closed-form coefficients, Ma & Marston
   Eq. 28);
2. Obtain the power series of ln Psi_0 by recursing on the coefficients of
   Psi'/Psi = (ln Psi)';
3. Take an n-th order truncation and measure the maximum absolute error as kcut
   varies;
4. Plot against kcut/kstar and check whether different (N, R) collapse.

Output: results/E12_zeros.json
"""
import json
import os
import sys
from math import gamma

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n01_common as C


def besselJ(nu, x, nmax=400):
    """Power series for J_nu(x) (still converging well for x near j_{nu,1})."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for m in range(nmax):
        t = ((-1.0) ** m / (gamma(m + 1.0) * gamma(m + nu + 1.0))
             * (x / 2.0) ** (2 * m + nu))
        out = out + t
        if np.max(np.abs(t)) < 1e-17:
            break
    return out


def psi0(k, N, R):
    """Exact zero-mode characteristic function (k = |u|)."""
    x = R * np.asarray(k, dtype=float)
    nu = N / 2.0 - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = gamma(N / 2.0) * (x / 2.0) ** (1.0 - N / 2.0) * besselJ(nu, x)
    return val


def first_zero(nu):
    """First positive zero of J_nu: scan + bisection."""
    xs = np.linspace(0.5, max(12.0, nu * 1.2 + 8.0), 20001)
    v = besselJ(nu, xs)
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    assert len(idx) > 0, "no zero found"
    a, b = xs[idx[0]], xs[idx[0] + 1]
    for _ in range(200):
        m = 0.5 * (a + b)
        if np.sign(besselJ(nu, np.array([a]))[0]) == np.sign(besselJ(nu, np.array([m]))[0]):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def psi_series_coeffs(N, R, nmax=10):
    """Psi_0 = sum a_l z^l, z = k^2 (closed form, Ma & Marston Eq. 28)."""
    a = np.zeros(nmax + 1)
    for l in range(nmax + 1):
        a[l] = (gamma(N / 2.0) / gamma(l + N / 2.0)
                * (-1.0) ** l / (2.0 ** (2 * l) * gamma(l + 1.0))
                * R ** (2 * l))
    return a


def log_series_coeffs(a):
    """Coefficients of L = ln P from P = sum a_l z^l (recursion from P'/P = L')."""
    n = len(a)
    b = np.zeros(n)
    for m in range(1, n):
        s = m * a[m]
        for j in range(1, m):
            s -= (m - j) * b[m - j] * a[j]
        b[m] = s / (m * a[0])
    return b


def main():
    NS = [5, 7, 9, 11]
    RS = [0.5, 1.0, 2.0]
    out = {"meta": dict(N=NS, R=RS), "cases": []}

    print("=== zeros and truncation error (purely analytic) ===")
    print("  N    R     kstar     |  max|dlnPsi| at kcut/kstar = 0.5 / 0.8 / 0.95")
    for N in NS:
        nu = N / 2.0 - 1.0
        j1 = first_zero(nu)
        for R in RS:
            kstar = j1 / R
            a = psi_series_coeffs(N, R)
            b = log_series_coeffs(a)

            def ln_exact(k):
                return np.log(psi0(k, N, R))

            def ln_trunc(k, north):
                k = np.asarray(k, dtype=float)
                return sum(b[m] * k ** (2 * m) for m in range(north + 1))

            row = dict(N=N, R=R, kstar=float(kstar), j1=float(j1), ratios={})
            for ratio in (0.5, 0.8, 0.95):
                kc = ratio * kstar
                kk = np.linspace(1e-4 * kc, kc, 4001)
                e2 = float(np.max(np.abs(ln_exact(kk) - ln_trunc(kk, 1))))   # n = 2 truncation
                e4 = float(np.max(np.abs(ln_exact(kk) - ln_trunc(kk, 2))))   # n = 4 truncation
                row["ratios"]["%.2f" % ratio] = dict(e2=e2, e4=e4,
                                                     kc=float(kc),
                                                     le=float(abs(ln_exact(np.array([kc]))[0])))
            out["cases"].append(row)
            print("  %2d  %.1f  %8.4f  |  %.3e / %.3e / %.3e"
                  % (N, R, kstar, row["ratios"]["0.50"]["e2"],
                     row["ratios"]["0.80"]["e2"], row["ratios"]["0.95"]["e2"]))

    # ---- collapse check: is e2 only a function of kcut/kstar ----
    print("\n=== collapse check (n=2 truncation; listed by kcut/kstar) ===")
    ratios = np.linspace(0.1, 0.99, 19)
    curves = {}
    for row in out["cases"]:
        N, R, kstar = row["N"], row["R"], row["kstar"]
        a = psi_series_coeffs(N, R)
        b = log_series_coeffs(a)
        yy = []
        for rt in ratios:
            kc = rt * kstar
            kk = np.linspace(1e-6 * kc, kc, 3001)
            yy.append(float(np.max(np.abs(np.log(psi0(kk, N, R))
                                          - sum(b[m] * kk ** (2 * m) for m in range(2))))))
        curves["N%d_R%.1f" % (N, R)] = yy
    out["collapse"] = dict(ratios=ratios.tolist(), curves=curves)

    spread = []
    for i in range(len(ratios)):
        vals = [curves[k][i] for k in curves if curves[k][i] > 0]
        if vals:
            spread.append(max(vals) / min(vals))
    print("  max/min ratio of e2 across (N,R) at a fixed kcut/kstar:")
    print("    median %.2f, maximum %.2f" % (float(np.median(spread)), float(np.max(spread))))

    # movement of kstar with N at fixed R
    print("\n=== movement of kstar = j_{N/2-1,1}/R with N (R=1) ===")
    for N in NS:
        j1 = first_zero(N / 2.0 - 1.0)
        print("  N=%2d  j_{%.1f,1}=%.4f   (Airy asymptotics nu+1.8558 nu^{1/3}=%.4f)"
              % (N, N / 2.0 - 1.0, j1,
                 (N / 2.0 - 1.0) + 1.8558 * (N / 2.0 - 1.0) ** (1.0 / 3.0)))

    C.save_json("E12_zeros.json", out)


if __name__ == "__main__":
    main()
