"""Does the phase (Im K) carry predictability information?

The paper's precision function is H = d_Psi/dt with Psi = -ln|phi_t|, identically
equal to -Re K where K := d_t ln phi_t. The dropped part is Im K = d_t arg phi_t,
the phase rate. Information states with equal |phi| share the same predictability
if that part is only redundant notation, and differ if it carries real information.

Minimal pair: mirror the cloud about its own mean (X -> 2*mu - X).
    phi'(k) = e^{2ik.mu} conj(phi(k))  =>  |phi'| ≡ |phi| (exact)
    equal mean, equal covariance element by element, equal even central moments;
    odd central moments flip sign. The Lorenz flow is not reflection-equivariant,
    so the two states evolve differently, which is the source of the measurable
    difference.

Ensemble scale is arbitrary. Deterministic quadrature rules, no sampling noise.

Output: results/E04_phase.json
"""
import math

import numpy as np

import n01_common as C
import n02_shapes as E

KAPPA_LIST = [0.0, 0.0025, 0.010]
T_WIN_LIST = [0.05, 0.10, 0.25]
EPS_LIST = [0.10, 0.20, 0.30, 0.40]
T_SPIN = E.T_SPIN


def skewed_rule(p=0.25, q=0.45):
    """Three-point skewed rule: nodes [-a, 0, b], weights [p, 1-p-q, q]; mean 0 and variance 1 exactly."""
    b = math.sqrt(p / (q * (p + q)))
    a = b * q / p
    return np.array([-a, 0.0, b]), np.array([p, 1.0 - p - q, q])


def moments_of(nd, wt, upto=5):
    return [float(np.sum(wt * nd ** k)) for k in range(1, upto + 1)]


def cf_along_x(X, W, ks):
    """phi(k) along k = (k,0,0) for each k in ks."""
    return (np.exp(1j * np.outer(ks, X[:, 0])) * (W / W.sum())[None, :]).sum(axis=1)


def kappa_of(X0, W, kappa, t, dt=0.01, ks=(-0.30, -0.20, -0.10, 0.10, 0.20, 0.30)):
    """K = d_t ln phi_t(k). Forward difference at t=0, central difference at t>0.

    dt must be at least the integration step H; otherwise the inner step count is
    rounded to 0 and the result is identically zero. ks uses small |k| so that phi
    stays away from its zeros (branch issues in ln).
    """
    ks = np.asarray(ks, float)
    if t <= 1e-12:
        pp = cf_along_x(E.propagate_cloud(X0, W, kappa, dt), W, ks)
        pm = cf_along_x(X0, W, ks)
        return (np.log(pp) - np.log(pm)) / dt, ks
    Xp = E.propagate_cloud(X0, W, kappa, t + dt)
    Xm = E.propagate_cloud(X0, W, kappa, t - dt)
    pp = cf_along_x(Xp, W, ks)
    pm = cf_along_x(Xm, W, ks)
    return (np.log(pp) - np.log(pm)) / (2 * dt), ks


def main():
    nd, wt = skewed_rule()
    out = {
        "config": dict(kappa=KAPPA_LIST, T_win=T_WIN_LIST, eps=EPS_LIST,
                       nodes=nd.tolist(), weights=wt.tolist()),
    }

    print("=" * 78)
    print("1. rule check: under mirroring, even moments are equal and odd moments flip sign")
    print("=" * 78)
    m_s = moments_of(nd, wt)
    m_m = moments_of(-nd, wt)
    print(f"{'':>6} {'m1':>12} {'m2':>12} {'m3':>12} {'m4':>12} {'m5':>12}")
    print("  orig  " + "".join(f"{v:12.6f}" for v in m_s))
    print("  mirror" + "".join(f"{v:12.6f}" for v in m_m))
    print(f"  m2 diff = {abs(m_s[1]-m_m[1]):.2e}   m4 diff = {abs(m_s[3]-m_m[3]):.2e}"
          f"   m3 sum = {m_s[2]+m_m[2]:.2e}")
    out["rule"] = dict(orig=m_s, mirror=m_m)

    q0 = C.integrate(np.array([1.0, 1.0, 1.0]), 0.0, T_SPIN)
    eps_ref = 0.20
    S0 = E.Sigma0_of(eps_ref)

    X, W = C.build_cloud(nd, wt, S0, mean=q0)
    Xm, Wm = C.build_cloud(-nd, wt, S0, mean=q0)

    print()
    print("=" * 78)
    print("2. cloud check (eps=0.20, a point on the Lorenz attractor)")
    print("=" * 78)
    mu, cov, Lam = C.weighted_moments(X, W)
    mu_m, cov_m, Lam_m = C.weighted_moments(Xm, Wm)
    print(f"  mean diff max     = {np.abs(mu-mu_m).max():.3e}")
    print(f"  cov diff max   = {np.abs(cov-cov_m).max():.3e}   (relative to |Sigma|={np.abs(cov).max():.3e})")
    print(f"  4th-cumulant diff   = {np.abs(Lam-Lam_m).max():.3e}")
    M3 = np.einsum("n,ni,nj,nk->ijk", W / W.sum(), X - mu, X - mu, X - mu)
    M3m = np.einsum("n,ni,nj,nk->ijk", Wm / Wm.sum(), Xm - mu_m, Xm - mu_m, Xm - mu_m)
    print(f"  3rd central-moment diff   = {np.abs(M3-M3m).max():.3e}   (should be ~2x)")
    out["cloud"] = dict(
        d_mean=float(np.abs(mu - mu_m).max()),
        d_cov=float(np.abs(cov - cov_m).max()),
        d_Lam=float(np.abs(Lam - Lam_m).max()),
        M3_max=float(np.abs(M3).max()), M3_mirror_max=float(np.abs(M3m).max()),
    )

    print()
    print("=" * 78)
    print("3. characteristic function: |phi| identical, phase different (k along x)")
    print("=" * 78)
    ks = np.array([0.05, 0.15, 0.30, 0.50, 0.80, 1.20])
    p1 = cf_along_x(X, W, ks)
    p2 = cf_along_x(Xm, Wm, ks)
    print(f"{'k':>8} {'|phi|':>11} {'|phi_mirror|':>13} {'||diff':>10} "
          f"{'arg phi':>11} {'arg mirror':>11}")
    for i, k in enumerate(ks):
        d = abs(abs(p1[i]) - abs(p2[i]))
        print(f"{k:8.2f} {abs(p1[i]):11.6f} {abs(p2[i]):13.6f} {d:10.2e} "
              f"{np.angle(p1[i]):11.6f} {np.angle(p2[i]):11.6f}")
    out["cf"] = dict(k=ks.tolist(), abs_diff=float(np.max(np.abs(np.abs(p1) - np.abs(p2)))),
                     arg_orig=np.angle(p1).tolist(), arg_mirror=np.angle(p2).tolist())

    print()
    print("=" * 78)
    print("4. difference of K = d_t ln phi_t itself (t=0, small k)")
    print("=" * 78)
    K1, kk = kappa_of(X, W, 0.0, 0.0)
    K2, _ = kappa_of(Xm, Wm, 0.0, 0.0)
    print(f"{'k':>8} {'Re K':>12} {'Re K_mirror':>13} {'Im K':>12} {'Im K_mirror':>13}")
    for i, k in enumerate(kk):
        print(f"{k:8.2f} {K1[i].real:12.5f} {K2[i].real:13.5f} "
              f"{K1[i].imag:12.5f} {K2[i].imag:13.5f}")
    print(f"  max Re diff = {np.abs(K1.real-K2.real).max():.3e}"
          f"   max Im diff = {np.abs(K1.imag-K2.imag).max():.3e}")
    print("  note: at t=0 neither cloud has evolved; the Re difference comes from the different cloud shapes (third moment flipped)")
    out["K_t0"] = dict(k=kk.tolist(),
                       Re_orig=K1.real.tolist(), Re_mirror=K2.real.tolist(),
                       Im_orig=K1.imag.tolist(), Im_mirror=K2.imag.tolist())

    print()
    print("=" * 78)
    print("5. decisive test: does the window FTLE (the measurable of the paper's H) change under mirroring?")
    print("=" * 78)
    runs = []
    for kappa in KAPPA_LIST:
        for eps in EPS_LIST:
            S = E.Sigma0_of(eps)
            A, WA = C.build_cloud(nd, wt, S, mean=q0)
            B, WB = C.build_cloud(-nd, wt, S, mean=q0)
            for T in T_WIN_LIST:
                _, _, ftA, _, _, _, _ = E.ftle_vector(A, WA, kappa, T)
                _, _, ftB, _, _, _, _ = E.ftle_vector(B, WB, kappa, T)
                d = ftB - ftA
                runs.append(dict(kappa=kappa, eps=eps, T=T,
                                 ftle_orig=ftA.tolist(), ftle_mirror=ftB.tolist(),
                                 diff=d.tolist()))
                print(f"  kappa={kappa:6.4f} eps={eps:.2f} T={T:5.2f} | "
                      f"FTLE orig  = {np.array2string(ftA, precision=5)}")
                print(f"  {'':29s} | FTLE mirror = {np.array2string(ftB, precision=5)}"
                      f"   ΔFTLE = {np.array2string(d, precision=3)}")
    out["runs"] = runs

    print()
    print("=" * 78)
    print("6. summary: per-direction relative amplitude |Delta_i|/|FTLE_i| (paper Table 3 convention)")
    print("=" * 78)
    print(f"{'kappa':>8} {'eps':>6} {'T':>6} {'max|Δ|':>12} "
          f"{'rel(d0)':>10} {'rel(d1)':>10} {'rel(d2)':>10}")
    worst = 0.0
    for r in runs:
        d = np.abs(r["diff"]).max()
        ft = np.abs(r["ftle_orig"])
        # paper convention: per-direction relative amplitude, denominator is that direction's own rate
        rel_dir = [abs(r["diff"][i]) / ft[i] if ft[i] > 0 else float("nan")
                   for i in range(3)]
        worst = max(worst, d)
        print(f"{r['kappa']:8.4f} {r['eps']:6.2f} {r['T']:6.2f} {d:12.4e} "
              f"{rel_dir[0]:10.4f} {rel_dir[1]:10.4f} {rel_dir[2]:10.4f}")
    print(f"\n  overall max |Δ FTLE| = {worst:.4e}")
    print("  reference: in E2 the kappa4 channel at eps=0.4 gives Delta = 0.15% of lambda_0 (~1e-3)")
    # per-direction convention (paper Table 3 uses this); summary also stores the per-direction amplitude for each (kappa,eps,T)
    out["summary"] = dict(max_abs_diff=float(worst),
                          rel_dir=[[r["kappa"], r["eps"], r["T"],
                                    [abs(r["diff"][i]) / abs(r["ftle_orig"][i])
                                     for i in range(3)]] for r in runs])

    print()
    print("=" * 78)
    print("7. scaling: eps-exponent of DeltaFTLE (the kurtosis channel is eps^2; this channel TBD)")
    print("=" * 78)
    q0b = C.integrate(np.array([1.0, 1.0, 1.0]), 0.0, T_SPIN)
    for kappa in (0.0, 0.010):
        for T in (0.05, 0.10):
            es, ds = [], []
            for eps in EPS_LIST:
                S = E.Sigma0_of(eps)
                A, WA = C.build_cloud(nd, wt, S, mean=q0b)
                B, WB = C.build_cloud(-nd, wt, S, mean=q0b)
                _, _, fA, _, _, _, _ = E.ftle_vector(A, WA, kappa, T)
                _, _, fB, _, _, _, _ = E.ftle_vector(B, WB, kappa, T)
                es.append(eps)
                ds.append(abs((fB - fA)[0]))
            es, ds = np.array(es), np.array(ds)
            ok = ds > 0
            sl = np.polyfit(np.log(es[ok]), np.log(ds[ok]), 1)[0] if ok.sum() > 1 else float("nan")
            print(f"  kappa={kappa:6.4f} T={T:5.2f}  eps=" +
                  " ".join(f"{e:.2f}" for e in es) + "   |Delta_0|=" +
                  " ".join(f"{d:.3e}" for d in ds) + f"   slope={sl:.4f}")
            out.setdefault("scaling", []).append(
                dict(kappa=kappa, T=T, eps=es.tolist(), d0=ds.tolist(), slope=float(sl)))

    C.save_json("E04_phase.json", out)
    print("\nwrote results/E04_phase.json")


if __name__ == "__main__":
    main()
