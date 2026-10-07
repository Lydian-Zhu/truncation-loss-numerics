# -*- coding: utf-8 -*-
"""Analytic coefficients c_3 / c_4 on L63 (quantitative form of the odd-order lead).

Derivation skeleton (deterministic flow, no diffusion)
------------------------------------------------------
Exact covariance evolution (Taylor-expand F about its mean, keeping the
second-order remainder):
    dot g_{ij} = J_{ia} g_{aj} + g_{ia} J_{ja}          <- L^(2) (tangent-linear)
               + 1/2 d^2_{ab}F_i M_3^{abj}              <- channel (th:chain)
               + O(M_4, ...)
A deterministic flow has no deterministic "g·g" term: the N^(3)(g,g) of the paper's
eq:trunc_gdot is notation for the stochastic (diffusive) case and, in the
noise-free limit, is absorbed into chan(M_3).

Third central moment evolution:
    dot M_3^{ijk} = 1/2 ( d^2_{ab}F_i M_4^{abjk} + d^2_{ab}F_j M_4^{abki}
                          + d^2_{ab}F_k M_4^{abij} ),
    M_4 = S(g) + Lambda,   S(g)_{ijkl} = g_ij g_kl + g_ik g_jl + g_il g_jk.

L63-specific facts
------------------
  nabla^2 F is a constant tensor: d^2_{xz}F_y = d^2_{zx}F_y = -1,
                                 d^2_{xy}F_z = d^2_{yx}F_z = +1,
  nabla^3 F = 0  ==>  L^(4)(Lambda) = 0 (the even branch does not pass through the
  third-derivative channel).
The channel equation then reduces to constant coefficients:
    dot g^{,(II)}_{y j} = -M_3^{xzj},   dot g^{,(II)}_{z j} = +M_3^{xyj},
    dot g^{,(II)}_{x j} = 0.            (★)

eps power bookkeeping
---------------------
Rescale kappa_m ∝ eps^m. With normalized G = g/eps^2 and T = M_3/eps^3 and initial
values G(0) = C0 (O(1)), T(0) = gamma_3 * TH(0) (O(1), TH(0) a given shape),
expand G = G0 + eps G1 + ..., T = T0 + eps T1 + ...:
  G0' = L2(G0)                         -> G0(t) = Phi_t C0 Phi_t^T
  T0' = M3GEN(G0)                      -> T0(t) = ∫ M3GEN(G0)
  G1' = L2(G1) + chan(T0)              -> odd-branch backflow (eps^1)
  T1' = M3GEN_lin(G0, G1)              -> T1 (eps^1)
  G2' = L2(G2) + chan(T1)              -> even branch (eps^2)
Reading Delta_i = -dot H_i = -<v_i, dot g v_i>/(2 sigma_bar_i^2) with
sigma_bar_i^2 = eps^2 * ev_i (ev_i the i-th eigenvalue of G0), so
  Delta_i^odd  = -eps * <v_i, chan(T0) v_i> / (2 ev_i)   ∝ gamma_3 eps
  Delta_i^even = -eps^2 * <v_i, chan(T1) v_i> / (2 ev_i) ∝ gamma_4 eps^2

Definitions
  c_3^i := Delta_i^odd /(gamma_3 eps) = -<v_i, chan(T0) v_i>/(2 ev_i),
  c_4^i := Delta_i^even/(gamma_4 eps^2)
where T0 is normalized to unit gamma_3 and T1 to unit gamma_4 (see main). Both are
O(1) numbers independent of eps; these are the c_3, c_4 left undetermined by
proposition th:split.

Output: results/E09_c3_coeff.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n01_common as C

H_INT = 1e-3
X0 = (1.0, 1.0, 1.0)
RELAX = 20.0
T_MEAS = 0.5


# --------------------------------------------------------- L63 constant tensors
def A2():
    A = np.zeros((3, 3, 3))
    A[1, 0, 2] = A[1, 2, 0] = -1.0
    A[2, 0, 1] = A[2, 1, 0] = +1.0
    return A


A2T = A2()


def _contract(M, out_idx):
    """1/2 d^2_ab F_i M^{ab, out_idx} (solved for each set of output indices)."""
    return 0.5 * np.einsum("iab,ab...->i...", A2T, M)


def chan(T3):
    """1/2 d^2_bc F_i T^{bcj} -> 3x3 (full form of channel (★))."""
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = 0.5 * sum(A2T[i, a, b] * T3[a, b, j]
                                  for a in range(3) for b in range(3))
    return out


def S4(g):
    return (np.einsum("ij,kl->ijkl", g, g)
            + np.einsum("ik,jl->ijkl", g, g)
            + np.einsum("il,jk->ijkl", g, g))


def M3_rate(M4):
    """dot M_3^{ijk} = 1/2 ( d2F_i M4^{abjk} + d2F_j M4^{abki} + d2F_k M4^{abij} )."""
    out = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for (p, r, t) in ((i, j, k), (j, k, i), (k, i, j)):
                    s += 0.5 * sum(A2T[p, a, b] * M4[a, b, r, t]
                                   for a in range(3) for b in range(3))
                out[i, j, k] = s
    return out


def L2(G, J):
    return J @ G + G @ J.T


def step(x, Phi, h):
    J = C.J_lorenz(x, 0.0)
    k1 = C.f_lorenz(x, 0.0); K1 = J @ Phi
    x2 = x + 0.5 * h * k1
    k2 = C.f_lorenz(x2, 0.0); K2 = C.J_lorenz(x2, 0.0) @ (Phi + 0.5 * h * K1)
    x3 = x + 0.5 * h * k2
    k3 = C.f_lorenz(x3, 0.0); K3 = C.J_lorenz(x3, 0.0) @ (Phi + 0.5 * h * K2)
    x4 = x + h * k3
    k4 = C.f_lorenz(x4, 0.0); K4 = C.J_lorenz(x4, 0.0) @ (Phi + h * K3)
    return (x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4),
            Phi + h / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4))


def run(T, C0, TH0=None, Lam0=None, h=H_INT, x0=X0):
    """Advance to T. TH0: initial value of T0 (unit gamma_3, given shape); Lam0: fourth-order cumulant (unit gamma_4); both normalized, independent of eps."""
    x = np.array(x0, float)
    for _ in range(int(round(RELAX / 0.002))):
        x = C.rk4(x, 0.0, 0.002)
    Phi = np.eye(3)
    G1 = np.zeros((3, 3))
    G2 = np.zeros((3, 3))
    T0 = np.zeros((3, 3, 3)) if TH0 is None else np.array(TH0, float)
    T1 = np.zeros((3, 3, 3))
    n = int(round(T / h))
    for _ in range(n):
        J = C.J_lorenz(x, 0.0)
        G0 = Phi @ C0 @ Phi.T
        M4_0 = S4(G0) + (Lam0 if Lam0 is not None else 0.0)
        dT0 = M3_rate(M4_0)
        # source for T1: first-order correction of M4 = directional derivative of S4 along G1 (S4 is quadratic)
        M4_1 = (np.einsum("ij,kl->ijkl", G1, G0) + np.einsum("ij,kl->ijkl", G0, G1)
                + np.einsum("ik,jl->ijkl", G1, G0) + np.einsum("ik,jl->ijkl", G0, G1)
                + np.einsum("il,jk->ijkl", G1, G0) + np.einsum("il,jk->ijkl", G0, G1))
        dT1 = M3_rate(M4_1)
        dG1 = L2(G1, J) + chan(T0)
        dG2 = L2(G2, J) + chan(T1)
        T0 = T0 + h * dT0
        T1 = T1 + h * dT1
        G1 = G1 + h * dG1
        G2 = G2 + h * dG2
        x, Phi = step(x, Phi, h)
    G0 = Phi @ C0 @ Phi.T
    return dict(G0=G0, G1=G1, G2=G2, T0=T0, T1=T1, Phi=Phi)


def coefficients(C0, T=T_MEAS, h=H_INT, x0=X0, TH0=None, Lam0=None):
    r = run(T, C0, TH0=TH0, Lam0=Lam0, h=h, x0=x0)
    ev, evec = np.linalg.eigh(r["G0"])
    ch0 = chan(r["T0"])
    ch1 = chan(r["T1"])
    c3, c4 = [], []
    for i in range(3):
        v = evec[:, i]
        c3.append(-0.5 * float(v @ ch0 @ v) / ev[i])
        c4.append(-0.5 * float(v @ ch1 @ v) / ev[i])
    return dict(ev=ev.tolist(), c3=c3, c4=c4,
                G1=r["G1"].tolist(), G2=r["G2"].tolist(),
                T0_full=r["T0"].tolist(), T1_full=r["T1"].tolist(),
                T0_max=float(np.abs(r["T0"]).max()),
                T1_max=float(np.abs(r["T1"]).max()))


def main():
    print("=" * 74)
    print("coefficients c_3 (odd branch) and c_4 (even branch) on L63  --  T = %.2f" % T_MEAS)
    print("=" * 74)
    print("channel equation (★): dot g_{y j} = -M_3^{xzj},  dot g_{z j} = +M_3^{xyj},  dot g_{x j} = 0")
    print("on L63 nabla^3 F = 0  ==>  L^(4)(Lambda) = 0 (the even branch does not pass through the third-derivative channel)")

    C0 = np.eye(3)
    out = {"meta": dict(T=T_MEAS, h=H_INT, x0=list(X0), C0="I",
                        note="c3 = -<v_i,chan(T0)v_i>/(2 ev_i); same for c4 with T1")}

    # ---- odd branch: initially only skewness (kappa_4(0) = 0, Lam0 = 0) ----
    print("\n--- odd branch c_3: initial state has only skewness (Lam0 = 0) ---")
    a = coefficients(C0, Lam0=None)
    print("  G0 eigenvalues/eps^2 =", np.round(a["ev"], 8))
    for i in range(3):
        print("    i=%d   sigma_bar^2/eps^2 = %12.8f   c_3^i = %+.6e"
              % (i, a["ev"][i], a["c3"][i]))
    out["odd"] = a

    # ---- even branch: dynamical source of the even branch for the same (skewness-only) initial state ----
    print("\n--- even branch c_4 (same initial state, Lam0 = 0, even branch generated by the dynamics) ---")
    for i in range(3):
        print("    i=%d   c_4^i = %+.6e" % (i, a["c4"][i]))
    out["even_dyn"] = a["c4"]

    # ---- step-size convergence ----
    print("\n--- step-size convergence (c_3 should be stable) ---")
    conv = []
    for h in (2e-3, 1e-3, 5e-4):
        r = coefficients(C0, h=h)
        conv.append(dict(h=h, c3=r["c3"], c4=r["c4"]))
        print("  h=%.4g  c3 = [%s]" % (h, ", ".join("%+.6e" % v for v in r["c3"])))
    out["convergence"] = conv

    # ---- multiple initial values ----
    print("\n--- multiple initial values (c_3 varies with the orbit point) ---")
    seeds = [(1.0, 1.0, 1.0), (5.0, -2.0, 20.0), (-8.0, 4.0, 25.0), (12.0, 3.0, 30.0)]
    sd = []
    for s in seeds:
        r = coefficients(C0, x0=s)
        sd.append(dict(x0=list(s), c3=r["c3"], c4=r["c4"], ev=r["ev"]))
        print("  seed %-18s c3 = [%s]" % (str(s), ", ".join("%+.4e" % v for v in r["c3"])))
    out["seeds"] = sd

    # ---- sign/order statistics of c_3 against c_4 ----
    print("\n--- comparison: |c_3| vs |c_4| (same initial value, L63) ---")
    print("  direction  sigma_bar^2/eps^2      |c_3|          |c_4|       |c3/c4|")
    for i in range(3):
        r = abs(a["c3"][i]) / max(abs(a["c4"][i]), 1e-300)
        print("    %d     %14.8f   %.6e   %.6e   %.3e"
              % (i, a["ev"][i], abs(a["c3"][i]), abs(a["c4"][i]), r))

    # ---- key: relative-amplitude convention (consistent with the definition of Delta_i) ----
    # Delta_i^odd  = -0.5 <v_i, chan(T0) v_i> / ev_i * gamma_3 eps  (normalized)
    # relative amplitude = Delta_i / (intrinsic rate of sigma_bar_i); more directly, the eps exponent
    # here report the dimensionless |c_3^i| / ev_i and |c_4^i| / ev_i
    print("\n--- dimensionless convention: |c_3^i|/ev_i and |c_4^i|/ev_i (stripping the 1/sigma^2 weight) ---")
    print("  direction  |c_3|/ev        |c_4|/ev       ratio")
    for i in range(3):
        q3 = abs(a["c3"][i]) / a["ev"][i]
        q4 = abs(a["c4"][i]) / a["ev"][i]
        print("    %d   %.6e   %.6e   %.3e" % (i, q3, q4, q3 / max(q4, 1e-300)))

    # ---- tensor-norm convention (a direction-independent scalar) ----
    print("\n--- tensor-norm convention (direction independent) ---")
    print("  ||chan(T0)||_F = %.6e" % float(np.linalg.norm(chan(np.array(a["T0_full"])), "fro")))
    print("  ||chan(T1)||_F = %.6e" % float(np.linalg.norm(chan(np.array(a["T1_full"])), "fro")))
    out["norm_chan_T0"] = float(np.linalg.norm(chan(np.array(a["T0_full"])), "fro"))
    out["norm_chan_T1"] = float(np.linalg.norm(chan(np.array(a["T1_full"])), "fro"))

    # ---- finite-difference check: the G1' equation is time-consistent ----
    # with h=1e-3 at T=0.5 and 0.51 (different step counts), the difference quotient matches chan(T0) and L2(G1).
    print("\n--- finite-difference check (G1' = L2(G1)+chan(T0)) ---")
    h = 1e-3
    T1q, T2q = 0.50, 0.51
    r1 = run(T1q, C0, h=h)
    r2 = run(T2q, C0, h=h)
    dG1_fd = (np.array(r2["G1"]) - np.array(r1["G1"])) / (T2q - T1q)
    xt = C.integrate(np.array(X0, float), 0.0, RELAX, 0.002)
    xt = C.integrate(xt, 0.0, 0.5 * (T1q + T2q), h)
    Jt = C.J_lorenz(xt, 0.0)
    T0m = 0.5 * (np.array(r1["T0"]) + np.array(r2["T0"]))
    dG1_an = L2(0.5 * (np.array(r1["G1"]) + np.array(r2["G1"])), Jt) + chan(T0m)
    err = np.max(np.abs(dG1_fd - dG1_an)) / max(np.max(np.abs(dG1_an)), 1e-30)
    print("  difference quotient vs RHS relative deviation = %.3e" % err)
    out["fd_check"] = float(err)

    C.save_json("E09_c3_coeff.json", out)
    print("\nwrote results/E09_c3_coeff.json")


if __name__ == "__main__":
    main()
