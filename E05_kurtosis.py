# -*- coding: utf-8 -*-
"""Same-covariance, different-kurtosis pair (paper Table 4, even channel).

The two shapes have identical covariance element by element (guaranteed by the
build_cloud tensor-product construction) and differ only in the fourth-order
cumulant kappa4: the outermost family spike(a=2.8) (kappa4 = +3.472) against the
Gaussian family gauss(k4=0) (kappa4 = 0).

The measured quantity is the window-rate difference in direction 0,
    Delta_0(T) = FTLE_0(spike) - FTLE_0(gauss),
the contribution of the lost even-order part to the measurable rate.

Configuration matches the paper: kappa = 0 (Lorenz-63), T = 0.10,
Sigma0 = eps^2 R(32 deg) diag(1, 1.1, 0.9) R(32 deg)^T, h = 0.002.

Output: results/E05_kurtosis.json
"""
import numpy as np

import n01_common as C
import n02_shapes as E

KAPPA = 0.0
T = 0.10
H = 0.002
T_SPIN = 60.0
EPS_LIST = [0.10, 0.20, 0.30, 0.40]
BASE = "gauss(k4=0)"
PAIR = "spike(a=2.8)"


def main():
    q0 = C.integrate(np.array([1.0, 1.0, 1.0]), KAPPA, T_SPIN)
    shapes = C.rule_table()
    print("q0 =", np.array2string(q0, precision=6))
    print("pairing: %s (kappa4=%.4f)  vs  %s (kappa4=%.4f)"
          % (PAIR, C.kurtosis_of_rule(*shapes[PAIR]),
             BASE, C.kurtosis_of_rule(*shapes[BASE])))

    rec = {"config": dict(kappa=KAPPA, T=T, h=H, T_SPIN=T_SPIN,
                          ROT_DEG=E.ROT_DEG, Sigma0_eig=[1.0, 1.1, 0.9],
                          eps=EPS_LIST, base=BASE, pair=PAIR,
                          q0=q0.tolist()),
           "runs": []}

    rows = []
    for eps in EPS_LIST:
        S0 = E.Sigma0_of(eps)
        ft = {}
        for nm in (BASE, PAIR):
            nd, wt = shapes[nm]
            X0, W = C.build_cloud(nd, wt, S0, mean=q0)
            _, _, ftle, _, _, _, _ = E.ftle_vector(X0, W, KAPPA, T)
            ft[nm] = ftle
            rec["runs"].append(dict(eps=eps, shape=nm, ftle=ftle.tolist()))
        d0 = float(ft[PAIR][0] - ft[BASE][0])
        rel = 100.0 * d0 / abs(ft[BASE][0])
        rows.append((eps, d0, rel))
        print("  eps=%.2f  Delta_0=%+.6e   relative amplitude=%+.4f%%" % (eps, d0, rel))

    # exponent: least-squares slope of log|Delta| against log eps
    x = np.log(np.array([r[0] for r in rows]))
    y = np.log(np.abs(np.array([r[1] for r in rows])))
    p = float(np.polyfit(x, y, 1)[0])
    print("  scaling exponent p(Delta_even) = %.4f   (predicted 2)" % p)
    rec["summary"] = dict(rows=[dict(eps=r[0], delta=r[1], rel_percent=r[2]) for r in rows],
                          exponent_even=p)
    C.save_json("E05_kurtosis.json", rec)


if __name__ == "__main__":
    main()
