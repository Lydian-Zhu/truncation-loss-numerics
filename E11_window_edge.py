# -*- coding: utf-8 -*-
"""Effective-window boundary on the (eps, T) plane: a power-cleanliness diagnostic.

Where T <= 0.5 the eps exponents are clean (p3~3, p4~5); after T >= 0.75 p3 starts
to drift. This is not noise but the failure, along the T direction, of the
"leading powers do not mix" assumption behind th:split. This script quantifies that
boundary.

For each T, fit p3 and p4 over the eps grid and define
  clean(T) = |p3 - 3| + |p4 - 5|        (total deviation from the ideal exponents)
  leads(T) = p4 - p3                     (odd-branch lead)
Small clean(T) means clean exponents and valid order-separation; large clean(T)
means mixed exponents and a failed th:split assumption. The time at which clean(T)
reaches a threshold (e.g. 0.5) is recorded as T_*, the upper bound of the
order-separated effective window.

The eps dependence is also scanned: whether T_* is larger at smaller eps (a longer
window for finer resolution).

Output: results/E11_window_edge.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import E10_window as E10

EPS_LIST = [0.02, 0.04, 0.08, 0.16, 0.32]
T_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6]
THRESH = 0.5


def main():
    nodes, weights = E10.rule_1d(E10.GAMMA, E10.KURT4)
    print("=== E11: effective window of order-by-order separation (power cleanliness) ===")
    out = {"meta": dict(eps=EPS_LIST, T=T_LIST, thresh=THRESH), "rows": []}

    for T in T_LIST:
        w3, w4 = [], []
        for eps in EPS_LIST:
            r = E10.measure(eps, T, nodes, weights)
            w3.append(abs(r["dkappa"][2]))
            w4.append(abs(r["dkappa"][3]))
        p3 = E10.fit_power(EPS_LIST, w3)[0]
        p4 = E10.fit_power(EPS_LIST, w4)[0]
        clean = abs(p3 - 3.0) + abs(p4 - 5.0)
        out["rows"].append(dict(T=T, p3=p3, p4=p4, lead=p4 - p3, clean=clean))
        flag = "  <== mixed" if clean > THRESH else ""
        print("  T=%.2f   p3=%.3f  p4=%.3f  lead=%.3f  clean=%.3f%s"
              % (T, p3, p4, p4 - p3, clean, flag))

    # find T_*: first time clean exceeds the threshold
    Tstar = None
    for r in out["rows"]:
        if r["clean"] > THRESH:
            Tstar = r["T"]
            break
    print("\norder-by-order-separation window bound T_* = %s (clean > %.1f)"
          % ("%.2f" % Tstar if Tstar else ">%.2f" % T_LIST[-1], THRESH))
    out["T_star"] = Tstar

    # eps dependence: fix T=0.5 and look at clean versus eps (using different eps subsets)
    print("\n=== eps dependence (T=0.5, varying the eps subset upper bound) ===")
    rows2 = []
    for hi in (0.08, 0.16, 0.32):
        eps_sub = [e for e in EPS_LIST if e <= hi]
        w3, w4 = [], []
        for eps in eps_sub:
            r = E10.measure(eps, 0.5, nodes, weights)
            w3.append(abs(r["dkappa"][2]))
            w4.append(abs(r["dkappa"][3]))
        p3 = E10.fit_power(eps_sub, w3)[0]
        p4 = E10.fit_power(eps_sub, w4)[0]
        rows2.append(dict(eps_max=hi, npts=len(eps_sub), p3=p3, p4=p4,
                          clean=abs(p3 - 3.0) + abs(p4 - 5.0)))
        print("  eps<=%.2f (%d points)  p3=%.3f p4=%.3f clean=%.3f"
              % (hi, len(eps_sub), p3, p4, abs(p3 - 3.0) + abs(p4 - 5.0)))
    out["eps_dependence"] = rows2

    E10.C.save_json("E11_window_edge.json", out)
    print("\nwrote results/E11_window_edge.json")


if __name__ == "__main__":
    main()
