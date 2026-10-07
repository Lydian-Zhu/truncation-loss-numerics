"""Usability of the profile reading H_t(k) = -d_t ln|phi_t(k)| on the observation band.

The object K_t(k) = d_t ln phi_t(k) encapsulates predictability and can be read
either by order (Taylor: n=1 position, n=2 amplitude/spread, n>=3 higher-order
channels) or by scale (the retained profile). This script checks whether the
profile H_t(k) = -d_t ln|phi_t(k)| is well defined, smooth and usable on the
observation band. It shares the same object with the Taylor reading, and that
object has poles at the zeros of phi. If a pole enters the band at small k, the
profile reading is in practice more fragile than a dedicated construction such as
K_beta.

Setup: kappa = 0, eps = 0.40, k along the x axis, two snapshot times T = 0.10 and
0.25.

Output: results/E13_profile.json
"""
import numpy as np

import n01_common as C
import n02_shapes as E

KAPPA = 0.0
EPS = 0.40
H_STEP = 0.01          # time-difference step (must be >= the integration step E.H)
T_LIST = [0.10, 0.25]
K_GRID = np.concatenate([np.arange(0.10, 1.01, 0.10),
                         np.arange(1.25, 3.01, 0.25),
                         np.arange(3.50, 6.01, 0.50)])


def cf_x(X, W, ks):
    return (np.exp(1j * np.outer(ks, X[:, 0])) * (W / W.sum())[None, :]).sum(axis=1)


def profile(X0, W, kappa, t, ks):
    """H_t(k) = -d_t ln|phi_t(k)|, central difference."""
    fp = cf_x(E.propagate_cloud(X0, W, kappa, t + H_STEP), W, ks)
    fm = cf_x(E.propagate_cloud(X0, W, kappa, t - H_STEP), W, ks)
    lp = np.log(np.abs(fp))
    lm = np.log(np.abs(fm))
    return -(lp - lm) / (2 * H_STEP), np.abs(fp), np.abs(fm)


def main():
    q0 = C.integrate(np.array([1.0, 1.0, 1.0]), 0.0, E.T_SPIN)
    S0 = E.Sigma0_of(EPS)
    rules = C.rule_table()
    out = {"config": dict(kappa=KAPPA, eps=EPS, t=T_LIST, k=K_GRID.tolist()), "cases": {}}

    print("=" * 84)
    print("profile H_t(k) = -d_t ln|phi_t(k)|   (k along x, kappa=0, eps=0.40)")
    print("=" * 84)

    for t in T_LIST:
        print(f"\n--- t = {t} ---")
        hdr = f"{'k':>6}"
        for nm in rules:
            hdr += f" | {nm[:12]:>12} {'':>5}"
        print(hdr + "   note")
        prof = {}
        for nm, (nd, wt) in rules.items():
            X, W = C.build_cloud(nd, wt, S0, mean=q0)
            H, ap, am = profile(X, W, KAPPA, t, K_GRID)
            prof[nm] = dict(H=H.tolist(), abs_phi=ap.tolist(), kappa4=C.kurtosis_of_rule(nd, wt),
                            sign_changes=int(np.sum(np.diff(np.sign(H)) != 0)))
            out["cases"].setdefault(f"t={t}", {})[nm] = prof[nm]
        for i, k in enumerate(K_GRID):
            line = f"{k:6.2f}"
            for nm in rules:
                line += f" | {prof[nm]['H'][i]:12.4f} {np.abs(prof[nm]['abs_phi'][i]):5.3f}"
            print(line)

        print()
        for nm in rules:
            H = np.array(prof[nm]["H"])
            good = np.isfinite(H) & (np.abs(H) < 1e3)
            # quadratic fit at small k (k<=1.0)
            m = K_GRID <= 1.0
            c2 = np.polyfit(K_GRID[m], H[m], 2)
            print(f"  {nm:>14}: kappa4={prof[nm]['kappa4']:+8.4f}  "
                  f"|H| range=[{np.nanmin(H):8.3f},{np.nanmax(H):8.3f}]  "
                  f"sign changes {prof[nm]['sign_changes']} times  "
                  f"small-k quadratic coeff={c2[0]:+.4f} (linear term {c2[1]:+.4f})  finite points {good.sum()}/{len(H)}")

    print()
    print("=" * 84)
    print("whether the small-k end connects smoothly to second-order theory (the quadratic coefficient of H should be +1/2 dg_00/dt)")
    print("=" * 84)
    for nm, (nd, wt) in rules.items():
        X, W = C.build_cloud(nd, wt, S0, mean=q0)
        _, cP, _ = C.weighted_moments(E.propagate_cloud(X, W, KAPPA, 0.25 + H_STEP), W)
        _, cM, _ = C.weighted_moments(E.propagate_cloud(X, W, KAPPA, 0.25 - H_STEP), W)
        dg00 = (cP[0, 0] - cM[0, 0]) / (2 * H_STEP)
        H, _, _ = profile(X, W, KAPPA, 0.25, np.array([0.05]))
        print(f"  {nm:>14}:  +1/2 dg00/dt = {0.5*dg00:+.6f}   "
              f"numeric H(0.05)/k^2 = {H[0]/0.05**2:+.6f}")

    C.save_json("E13_profile.json", out)
    print("\nwrote results/E13_profile.json")


if __name__ == "__main__":
    main()
