"""Integration step-size convergence test for the window rate Delta_i(T) against the RK4 step h.

Setup: fix the initial state, system, observation scale and window length, and
halve the RK4 step by a factor of 4: h = 2e-3 -> 1e-3 -> 5e-4 -> 2.5e-4. Compare
the per-direction window rates Delta_i(T) = T^{-1} ln(sigma_i(T)/sigma_i(0)).

Criterion: RK4 has truncation error O(h^4), so halving h should shrink the change
by about 1/16. Bit-for-bit agreement would require all 15 digits identical, a
stronger requirement; the actual digit count is reported.

Output: results/E16_step_convergence.json
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n01_common as C
import n02_shapes as E

# settings matching the main experiment (n02_shapes: T_WIN=0.25, H=0.002, reference family gauss(k4=0))
KAPPA = 0.0
T = E.T_WIN
EPS = 0.12
H_LIST = [2e-3, 1e-3, 5e-4, 2.5e-4]
FAMILY = "gauss(k4=0)"


def ftle_with_h(X0, W, kappa, T, h):
    """Window rate with the integration step passed in explicitly (mirrors n02_shapes.ftle_vector, only the step changes)."""
    XT = E.propagate_cloud(X0, W, kappa, T, h=h)
    _, c0, _ = C.weighted_moments(X0, W)
    _, cT, _ = C.weighted_moments(XT, W)
    e0, v0 = C.sym_eig(c0)
    eT, _ = C.sym_eig(cT)
    s0 = np.sqrt(np.maximum(e0, 0.0))
    sT = np.sqrt(np.maximum(eT, 0.0))
    return np.log(sT / s0) / T, v0


def main():
    nd, wt = C.rule_table()[FAMILY]        # reference family, matching the n02_shapes main experiment
    S = E.Sigma0_of(EPS)
    q0 = C.integrate(np.array([1.0, 1.0, 1.0]), KAPPA, E.T_SPIN)
    X, W = C.build_cloud(nd, wt, S, mean=q0)

    out = dict(config=dict(kappa=KAPPA, T=T, eps=EPS, h_list=H_LIST,
                           family=FAMILY, q0=q0.tolist()), runs=[])

    print("=" * 78)
    print("E16 step convergence: window rate Delta_i(T) vs the RK4 step")
    print(f"  kappa={KAPPA}  T={T}  eps={EPS}  family={FAMILY}")
    print("=" * 78)
    print(f"{'h':>10} {'Δ_0':>16} {'Δ_1':>16} {'Δ_2':>16}")
    prev = None
    for h in H_LIST:
        d, v0 = ftle_with_h(X, W, KAPPA, T, h)
        out["runs"].append(dict(h=h, delta=d.tolist()))
        print(f"{h:10.1e} {d[0]:16.10f} {d[1]:16.10f} {d[2]:16.10f}")
        prev = d

    # per-direction relative change between adjacent steps
    print()
    print("relative change between adjacent steps |Delta(h) - Delta(h/2)| / |Delta(h/2)|:")
    for i in range(1, len(H_LIST)):
        d_prev = np.array(out["runs"][i - 1]["delta"])
        d_cur = np.array(out["runs"][i]["delta"])
        rel = np.abs(d_cur - d_prev) / np.abs(d_cur)
        print(f"  h={H_LIST[i-1]:.1e} -> {H_LIST[i]:.1e}: "
              f"{rel[0]:.3e}  {rel[1]:.3e}  {rel[2]:.3e}   (max {rel.max():.3e})")

    # bit-level comparison of the two finest steps
    d_fine = np.array(out["runs"][-1]["delta"])
    d_coarse = np.array(out["runs"][-2]["delta"])
    out["rel_change_finest"] = (np.abs(d_fine - d_coarse) / np.abs(d_fine)).tolist()
    out["max_rel_change_finest"] = float(
        (np.abs(d_fine - d_coarse) / np.abs(d_fine)).max())

    print()
    print(f"max relative change between the two finest steps (h=5e-4 -> 2.5e-4) = "
          f"{out['max_rel_change_finest']:.3e}")

    C.save_json("E16_step_convergence.json", out)
    print("\nwrote results/E16_step_convergence.json")


if __name__ == "__main__":
    main()
