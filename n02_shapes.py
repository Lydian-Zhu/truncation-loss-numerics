# -*- coding: utf-8 -*-
"""Same-covariance, different-kurtosis pair: beyond-second-order channel and its scaling exponent.

Prediction (paper thm:non_gaussian / eq:non_gaussian)
-----------------------------------------------------
  Hdot_i = -lambda_i + beta_{i,abcd} Lambda_{abcd} + O(eps^4)
  beta_{i,abcd} = -(1/(2 sigma_i^2)) v_i^p v_i^q L^{(4)}_{pq,abcd}
  L^{(4)}_{pq,abcd} = (1/6)( d^3_{abc}F_p delta_qd + d^3_{abc}F_q delta_pd )
With the covariance fixed element by element, shape enters only through Lambda
(the N^{(3)} term sees only g). The window-FTLE shape difference
Delta_i = FTLE_i^shape - FTLE_i^gauss is predicted in advance as
  Delta_i^pred = (1/(2 sigma_i^2(0))) v_i(0)^T A v_i(0),
  A_pq = L^{(4)}_{pq,abcd} Lambda_{abcd}   (each shape uses its own Lambda).

Two-case design
---------------
  kappa = 0   : Lorenz-63, nabla^3 F == 0  =>  A == 0, the kurtosis channel is
                exactly zero and the shape difference is only an O(eps^4)
                residual, so the predicted scaling exponent is 4.
  kappa != 0  : cubic-Lorenz, only nabla^3 F nonzero => shape difference O(eps^2),
                predicted exponent 2.

Proxy measurement (validation side, real nonlinear evolution only)
------------------------------------------------------------------
  FTLE_i(T) = (1/T) ln( sigma_i(T) / sigma_i(0) ), with sigma_i the square roots
  of the ensemble-covariance eigenvalues. From Hdot_i = -d ln sigma_i / dt,
  < -Hdot_i >_T = FTLE_i(T), an identity with the time integral of Hdot_i.

Shape families are 5-node moment-matching quadrature rules per dimension
(deterministic, no sampling noise); the covariance is exact element by element and
identical across shapes; kappa4 is swept continuously from -1.0 to +3.47 with
bounded support <= 2.8 sigma.

Output: results/E05b_kurtosis_scan.json
"""
import numpy as np

import n01_common as C

# ---------------------------------------------------------------- Config
EPS_LIST = [0.08, 0.12, 0.18, 0.27, 0.40]
KAPPA_LIST = [0.0, 0.0025, 0.005, 0.010]
T_WIN = 0.25
H = 0.002
T_SPIN = 60.0
ROT_DEG = 32.0          # rotate about z so v_i has x and y components (otherwise the contraction vanishes)


def rot_z(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])


def Sigma0_of(eps):
    R = rot_z(ROT_DEG)
    return eps ** 2 * (R @ np.diag([1.0, 1.1, 0.9]) @ R.T)


def propagate_cloud(X0, W, kappa, T, h=H):
    """Advance all nodes together with RK4 (vectorized); return the final cloud."""
    X = X0.copy()
    n = int(round(T / h))
    for _ in range(n):
        k1 = C.f_lorenz(X, kappa)
        k2 = C.f_lorenz(X + 0.5 * h * k1, kappa)
        k3 = C.f_lorenz(X + 0.5 * h * k2, kappa)
        k4 = C.f_lorenz(X + h * k3, kappa)
        X = X + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X


def ftle_vector(X0, W, kappa, T):
    XT = propagate_cloud(X0, W, kappa, T)
    _, c0, lam = C.weighted_moments(X0, W)
    _, cT, _ = C.weighted_moments(XT, W)
    e0, v0 = C.sym_eig(c0)
    eT, _ = C.sym_eig(cT)
    s0 = np.sqrt(np.maximum(e0, 0.0))
    sT = np.sqrt(np.maximum(eT, 0.0))
    gap0 = (e0[0] - e0[1]) / e0[0]
    gapT = (eT[0] - eT[1]) / eT[0]
    return s0, sT, (np.log(sT / s0) / T), v0, lam, gap0, gapT


def predict_delta(Lam, kappa, eigvec0, sigma0):
    """Prior prediction of the shape contribution to the window FTLE, (1/(2 sigma_i^2)) v_i^T A v_i."""
    T3 = C.d3F_lorenz(kappa)                     # T3[p,a,b,c]
    # A_pq = L^{(4)}_{pq,abcd} Lambda_{abcd}
    #     = (1/6)( T3[p,a,b,c] Lambda[q,a,b,c] + T3[q,a,b,c] Lambda[p,a,b,c] )
    A = np.einsum("pabc,qabc->pq", T3, Lam) / 6.0 \
        + np.einsum("qabc,pabc->pq", T3, Lam) / 6.0
    A = 0.5 * (A + A.T)
    out = np.array([float(v @ A @ v) / (2.0 * s ** 2)
                    for v, s in zip(eigvec0.T, sigma0)])
    return out, A


def main():
    shapes = C.rule_table()
    rec = {"config": dict(EPS_LIST=EPS_LIST, KAPPA_LIST=KAPPA_LIST, T_WIN=T_WIN, h=H,
                          T_SPIN=T_SPIN, ROT_DEG=ROT_DEG,
                          Sigma0_eig=[1.0, 1.1, 0.9])}
    shape_info = {}
    for nm, (nd, wt) in shapes.items():
        X, W = C.build_cloud(nd, wt, np.eye(3))
        shape_info[nm] = dict(kappa4=C.kurtosis_of_rule(nd, wt),
                              extent=float(np.abs(nd).max()),
                              n_nodes=int(len(nd) ** 3))
    rec["shapes"] = shape_info

    print("=== shape families ===")
    print(f"{'name':>16} {'kappa4':>9} {'ext node':>8} {'n nodes':>7}")
    for nm, si in shape_info.items():
        print(f"{nm:>16} {si['kappa4']:9.4f} {si['extent']:8.3f} {si['n_nodes']:7d}")

    runs = []
    for kappa in KAPPA_LIST:
        q0 = C.integrate(np.array([1.0, 1.0, 1.0]), kappa, T_SPIN)
        print(f"\n### kappa={kappa:g}  q0={np.array2string(q0, precision=4)}  "
              f"|nabla^3F|={6*kappa:g}")
        for eps in EPS_LIST:
            S0 = Sigma0_of(eps)
            per_shape = {}
            for nm, (nd, wt) in shapes.items():
                X0, W = C.build_cloud(nd, wt, S0, mean=q0)
                s0, sT, ft, v0, Lam, g0, gT = ftle_vector(X0, W, kappa, T_WIN)
                pred, A = predict_delta(Lam, kappa, v0, s0)
                per_shape[nm] = dict(ftle=ft.tolist(), pred=pred.tolist(),
                                     kappa4=shape_info[nm]["kappa4"],
                                     Lam_0000_norm=float(np.abs(Lam).max()),
                                     A_fro=float(np.linalg.norm(A)))
            ref = per_shape["gauss(k4=0)"]
            for nm, d in per_shape.items():
                runs.append(dict(kappa=kappa, eps=eps, shape=nm,
                                 kappa4=d["kappa4"],
                                 ftle=d["ftle"], pred=d["pred"],
                                 d_meas=[float(a - b) for a, b in
                                         zip(d["ftle"], ref["ftle"])],
                                 d_pred=[float(a - b) for a, b in
                                         zip(d["pred"], ref["pred"])],
                                 Lam_max=d["Lam_0000_norm"], A_fro=d["A_fro"]))
            # print the two most extreme shapes at each eps
            for nm in ("spike(a=1.5)", "spike(a=2.8)"):
                d = [r for r in runs if r["kappa"] == kappa and r["eps"] == eps
                     and r["shape"] == nm][0]
                print(f"  eps={eps:5.2f} {nm:>13}: d_meas={np.array2string(np.array(d['d_meas']), precision=3, floatmode='fixed')}"
                      f"  d_pred={np.array2string(np.array(d['d_pred']), precision=3, floatmode='fixed')}")
    rec["runs"] = runs

    # ---- scaling-exponent fit: |Delta_{i*}| vs eps ----
    fits = []
    print("\n=== scaling exponent (log|Delta_{i*}| vs log eps, i*=fastest-growing direction) ===")
    for kappa in KAPPA_LIST:
        for nm in ("spike(a=1.5)", "spike(a=2.0)", "spike(a=2.8)"):
            xs, ys = [], []
            for eps in EPS_LIST:
                r = [q for q in runs if q["kappa"] == kappa and q["eps"] == eps
                     and q["shape"] == nm][0]
                xs.append(np.log(eps))
                ys.append(np.log(abs(r["d_meas"][0]) + 1e-300))
            sl, ic = np.polyfit(xs, ys, 1)
            fits.append(dict(kappa=kappa, shape=nm, slope=float(sl), ic=float(ic)))
            print(f"  kappa={kappa:6.4f}  {nm:>13}  slope = {sl:6.3f}")
    rec["fits"] = fits
    C.save_json("E05b_kurtosis_scan.json", rec)
    return rec


if __name__ == "__main__":
    main()
