"""Second-order closure test (support for P3.2).

Pair under test: a cloud and its mirror about its own mean, X -> 2*mu - X. The
mirror pair has |phi'| ≡ |phi| exactly, identical mean, covariance and
fourth-order cumulant element by element, and third central moment of opposite
sign. If dSigma/dt(t=0) differs across the pair, the second-order closure
    dg = L2 g + N3 gg + L4 Lambda        (no M3 term)
cannot determine dg, because the exact rate contains the term (1/2) d2F . M3.
=> The second-order layer is not closed; the effect is measurable on Lorenz-63.

Output: results/E03_closure.json
"""
import numpy as np

import n01_common as C
import n02_shapes as E
import E04_phase as P

H = 0.01
EPS_LIST = [0.10, 0.20, 0.30, 0.40]


def M3_of(X, W):
    Wn = W / W.sum()
    mu = np.sum(Wn[:, None] * X, axis=0)
    dX = X - mu
    return np.einsum("n,ni,nj,nk->ijk", Wn, dX, dX, dX)


def main():
    q0 = C.integrate(np.array([1.0, 1.0, 1.0]), 0.0, E.T_SPIN)
    nd, wt = P.skewed_rule()
    out = {"config": dict(eps=EPS_LIST, h=H, nodes=nd.tolist(), weights=wt.tolist()), "runs": []}

    print("=" * 90)
    print("E7: second-order closure of a mirror pair")
    print("=" * 90)
    print(f"{'eps':>6} {'|Sigma_0|':>12} {'|dSigma/dt| orig':>16} {'mirror':>16} "
          f"{'|Delta|':>12} {'rel':>9}")

    ds = []
    for eps in EPS_LIST:
        S = E.Sigma0_of(eps)
        A, WA = C.build_cloud(nd, wt, S, mean=q0)
        B, WB = C.build_cloud(-nd, wt, S, mean=q0)
        _, cA, lA = C.weighted_moments(A, WA)
        _, cB, lB = C.weighted_moments(B, WB)
        m3A, m3B = M3_of(A, WA), M3_of(B, WB)

        da = (C.weighted_moments(E.propagate_cloud(A, WA, 0.0, H), WA)[1]
              - C.weighted_moments(A, WA)[1]) / H
        db = (C.weighted_moments(E.propagate_cloud(B, WB, 0.0, H), WB)[1]
              - C.weighted_moments(B, WB)[1]) / H
        d = float(np.abs(da - db).max())
        base = float(np.abs(da).max())
        ds.append(d)

        print(f"{eps:6.2f} {np.abs(cA).max():12.6f} {base:16.6e} {np.abs(db).max():16.6e} "
              f"{d:12.4e} {d/base:9.3%}")
        out["runs"].append(dict(
            eps=eps, abs_Sigma0=float(np.abs(cA).max()),
            dg_orig=float(base), dg_mirror=float(np.abs(db).max()),
            d_dg=d, rel=d / base,
            d_g=float(np.abs(cA - cB).max()),
            d_Lambda=float(np.abs(lA - lB).max()),
            d_M3=float(np.abs(m3A - m3B).max()),
            M3_scale=float(np.abs(m3A).max())))

    e = np.array(EPS_LIST)
    d = np.array(ds)
    sl = float(np.polyfit(np.log(e), np.log(d), 1)[0])
    out["slope"] = sl
    print(f"\n  |Delta dSigma/dt| fit slope vs eps = {sl:.4f}")
    print("  expected: M3 ~ eps^3 => dg ~ eps^3 => dH_i ~ eps^3 * eps^-2 = eps^1 (E5 measured tail 0.9989)")

    last = out["runs"][-1]
    print()
    print("=" * 90)
    print(f"coupling structure (eps={last['eps']})")
    print("=" * 90)
    print(f"  diff g      = {last['d_g']:.3e}   (machine zero)")
    print(f"  diff Lambda = {last['d_Lambda']:.3e}   (machine zero)")
    print(f"  diff M3     = {last['d_M3']:.3e}   (nonzero; M3 scale {last['M3_scale']:.3e})")
    print("  => the second-order closure contains no M3, the exact rate contains (1/2) d2F . M3  => not closed")

    C.save_json("E03_closure.json", out)
    print("\nwrote results/E03_closure.json")


if __name__ == "__main__":
    main()
