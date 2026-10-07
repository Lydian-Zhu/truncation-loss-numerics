# -*- coding: utf-8 -*-
"""Order-separation contours on the (eps, T) plane.

The ratio of Delta_odd to Delta_even (or their log deviations) is mapped on the
(eps, T) plane to show
  1. the (eps, T) range over which the odd-branch lead holds;
  2. the eps-direction power structure (the odd eps^1 / even eps^2 boundary);
  3. how long the two branches persist in T (when the odd branch overtakes the
     even one and when it reverses).

Convention (identical to E07/E08)
---------------------------------
1D five-point rule with nodes {-2,-1,0,1,2} and weights solving a linear system to
match the first four moments exactly: mu0=1, mu1=0, mu2=1, mu3=gamma, mu4=3+kurt4.
3D tensor product -> 125 points. Covariance = eps^2 I, cloud center on the
attractor. dot kappa_m uses central differences.

At each (eps, T) grid point:
  W3 = |dot kappa_3(T)|   (odd-branch rate)
  W4 = |dot kappa_4(T)|   (even-branch rate)
  R  = log10(W4/W3)       (lead; positive means the odd branch leads)
Normalization: the eps-direction exponent difference should be a constant 1
(ideal case).

Output: results/E10_window.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n01_common as C

NODES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
GAMMA = 0.5
KURT4 = 0.0                 # only skewness nonzero (k3only family)
EPS_LIST = [0.02, 0.04, 0.08, 0.16, 0.32]
T_LIST = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
H_INT = 1e-3
H_DER = 0.02
RELAX = 20.0


def rule_1d(gamma, kurt4):
    A = np.array([[q ** k for q in NODES] for k in range(5)], dtype=float)
    m = np.array([1.0, 0.0, 1.0, gamma, 3.0 + kurt4])
    w = np.linalg.solve(A, m)
    assert np.all(w > -1e-12), ("negative weight", gamma, kurt4, w)
    return NODES.copy(), w


def cumulants_1d(X, W, axis=0):
    q = X[:, axis]
    w = W / W.sum()
    m1 = float(np.sum(w * q))
    d = q - m1
    m2 = float(np.sum(w * d ** 2))
    m3 = float(np.sum(w * d ** 3))
    m4 = float(np.sum(w * d ** 4))
    return np.array([m1, m2, m3, m4 - 3.0 * m2 ** 2])


def mul_cloud(X, T, h=H_INT):
    x = X.copy()
    for _ in range(int(round(T / h))):
        x = C.rk4(x, 0.0, h)
    return x


_MU0 = None


def attractor_point():
    global _MU0
    if _MU0 is None:
        x = np.array([1.0, 1.0, 1.0])
        for _ in range(int(round(RELAX / 0.002))):
            x = C.rk4(x, 0.0, 0.002)
        _MU0 = x
    return _MU0


def measure(eps, T, nodes, weights, hd=H_DER):
    """Return (kappa_3, kappa_4, dkappa_3, dkappa_4) at time T."""
    X, W = C.build_cloud(nodes, weights, (eps ** 2) * np.eye(3), mean=attractor_point())
    km = cumulants_1d(mul_cloud(X, T - hd), W)
    k0 = cumulants_1d(mul_cloud(X, T), W)
    kp = cumulants_1d(mul_cloud(X, T + hd), W)
    dk = (kp - km) / (2.0 * hd)
    return dict(kappa=k0, dkappa=dk)


def fit_power(xs, ys):
    e = np.log(np.array(xs, float))
    v = np.log(np.abs(np.array(ys, float)))
    ok = np.isfinite(v)
    A = np.vstack([e[ok], np.ones(ok.sum())]).T
    p, _ = np.linalg.lstsq(A, v[ok], rcond=None)[0]
    return float(p), float(p)


def main():
    nodes, weights = rule_1d(GAMMA, KURT4)
    print("=== E10: (eps,T) contours ===")
    print("rule check mu0..mu4 =", ["%.6f" % np.sum(weights * nodes ** k) for k in range(5)])

    out = {"meta": dict(gamma=GAMMA, kurt4=KURT4, eps=EPS_LIST, T=T_LIST,
                        h_int=H_INT, h_der=H_DER),
           "grid": [], "powers": {}}

    # scan over (eps, T)
    for T in T_LIST:
        row = []
        for eps in EPS_LIST:
            r = measure(eps, T, nodes, weights)
            w3 = abs(r["dkappa"][2])
            w4 = abs(r["dkappa"][3])
            row.append(dict(eps=eps, w3=w3, w4=w4,
                            lead=float(np.log10(w4 / w3)) if w3 > 0 and w4 > 0 else None))
        out["grid"].append(dict(T=T, row=row))
        print("\nT = %.2f" % T)
        print("   eps      |dk3|       |dk4|      log10(|dk4|/|dk3|)")
        for c in row:
            ld = ("%+.4f" % c["lead"]) if c["lead"] is not None else "  n/a"
            print("   %.2f   %.4e  %.4e   %s" % (c["eps"], c["w3"], c["w4"], ld))

    # fit the eps exponent at each T
    print("\n=== eps exponent at each T ===")
    for T in T_LIST:
        row = [g for g in out["grid"] if g["T"] == T][0]["row"]
        p3 = fit_power([c["eps"] for c in row], [c["w3"] for c in row])[0]
        p4 = fit_power([c["eps"] for c in row], [c["w4"] for c in row])[0]
        out.setdefault("powers_by_T", []).append(dict(T=T, p3=p3, p4=p4, lead=p4 - p3))
        print("  T=%.2f  p(dk3)=%.3f  p(dk4)=%.3f  lead %.3f" % (T, p3, p4, p4 - p3))

    # T-scaling at each eps (optional)
    C.save_json("E10_window.json", out)
    print("\nwrote results/E10_window.json")


if __name__ == "__main__":
    main()
