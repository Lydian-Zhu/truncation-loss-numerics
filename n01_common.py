# -*- coding: utf-8 -*-
"""Common utilities: cubic-Lorenz system, RK4, moment-matching quadrature rules, JSON I/O.

Design notes
------------
1. The system is a one-parameter cubic-Lorenz family. kappa = 0 is Lorenz-63
   (nabla^3 F == 0); kappa != 0 makes only nabla^3 F nonzero, giving a built-in
   null/positive design.
2. Initial ensembles use moment-matching quadrature rules (5 nodes per dimension,
   125 points in 3D) rather than Monte Carlo: the covariance matches exactly
   element by element with no sampling noise. The 1D node distribution is a
   generalized normal p(q) ~ exp(-|q/alpha|^s), with s a continuous knob
   (s = 2 Gaussian, s -> inf near-uniform, s < 1 peaked with heavy tails).
3. Gauss rules are built by discretized Stieltjes orthogonalization, matching the
   first nine moments of the 1D distribution exactly.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


# ---------------------------------------------------------------- System
def f_lorenz(x, kappa):
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([SIGMA * (y - x),
                     x * (RHO - z) - y + kappa * x ** 3,
                     x * y - BETA * z], axis=-1)


def J_lorenz(x, kappa):
    x, y, z = x[..., 0], x[..., 1], x[..., 2]
    n = x.shape
    J = np.zeros(n + (3, 3))
    J[..., 0, 0] = -SIGMA
    J[..., 0, 1] = SIGMA
    J[..., 1, 0] = RHO - z + 3.0 * kappa * x ** 2
    J[..., 1, 1] = -1.0
    J[..., 1, 2] = -x
    J[..., 2, 0] = y
    J[..., 2, 1] = x
    J[..., 2, 2] = -BETA
    return J


def d3F_lorenz(kappa):
    """Third-derivative tensor T[p,a,b,c] = d^3 F_p / dq_a dq_b dq_c (constant in q)."""
    T = np.zeros((3, 3, 3, 3))
    T[1, 0, 0, 0] = 6.0 * kappa      # only d^3 f_y / dx^3 = 6 kappa
    return T


def rk4(x, kappa, h):
    k1 = f_lorenz(x, kappa)
    k2 = f_lorenz(x + 0.5 * h * k1, kappa)
    k3 = f_lorenz(x + 0.5 * h * k2, kappa)
    k4 = f_lorenz(x + h * k3, kappa)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(x0, kappa, T, h=0.002):
    n = int(round(T / h))
    x = np.array(x0, dtype=float)
    for _ in range(n):
        x = rk4(x, kappa, h)
    return x


def lyap_qr(kappa, T=300.0, h=0.002, q0=(1.0, 1.0, 1.0)):
    """Benettin QR Lyapunov spectrum (long-time asymptotic values).

    The variational equation dot Phi = J(q(t)) Phi is advanced with the same RK4
    scheme as the trajectory, with QR reorthonormalization every step. An earlier
    version propagated the first-order approximation I + h J, which shifted
    lambda_1 from 0.906 to 0.958 (about 5.7% systematic bias); it is not used.
    """
    x = np.array(q0, float)
    Phi = np.eye(3)
    S = np.zeros(3)
    n = int(round(T / h))
    for _ in range(n):
        k1 = f_lorenz(x, kappa)
        K1 = J_lorenz(x, kappa) @ Phi
        x2 = x + 0.5 * h * k1
        k2 = f_lorenz(x2, kappa)
        K2 = J_lorenz(x2, kappa) @ (Phi + 0.5 * h * K1)
        x3 = x + 0.5 * h * k2
        k3 = f_lorenz(x3, kappa)
        K3 = J_lorenz(x3, kappa) @ (Phi + 0.5 * h * K2)
        x4 = x + h * k3
        k4 = f_lorenz(x4, kappa)
        K4 = J_lorenz(x4, kappa) @ (Phi + h * K3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        Phi = Phi + h / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4)
        Q, R = np.linalg.qr(Phi)
        d = np.diag(R)
        sgn = np.sign(d)
        sgn[sgn == 0.0] = 1.0
        Phi = Q * sgn
        S += np.log(np.abs(d))
    return S / (n * h)


# ------------------------------------------- 1D distributions and Gauss rules
def gennorm_pdf(q, s, alpha):
    return np.exp(-np.abs(q / alpha) ** s)


def gennorm_kurtosis(s):
    from math import gamma
    return gamma(5.0 / s) * gamma(1.0 / s) / gamma(3.0 / s) ** 2 - 3.0


def gennorm_alpha_for_unit_var(s):
    from math import gamma
    return 1.0 / np.sqrt(gamma(3.0 / s) / gamma(1.0 / s))


def gauss_rule_1d(s, n=5, L=None, ngrid=None):
    """Build an n-point Gauss rule for p(q) ~ exp(-|q/alpha|^s) via discretized Stieltjes.

    Truncation range is set adaptively from the tail probability, requiring
    p(L) ~ e^{-40}; otherwise heavy-tailed cases (s <= 1) lose tail mass and even
    the fourth moment fails (measured: at s = 0.6, kappa4 drops from 12.58 to 6.34).
    """
    alpha = gennorm_alpha_for_unit_var(s)
    if L is None:
        L = alpha * 40.0 ** (1.0 / s)
    if ngrid is None:
        ngrid = int(min(600001, max(60001, 4000 * L)))
    q = np.linspace(-L, L, ngrid)
    w = gennorm_pdf(q, s, alpha)
    dq = q[1] - q[0]
    ww = w / np.sum(w)          # uniform grid -> sum(w) is the normalizing factor
    # discrete Stieltjes on the normalized uniform grid, weighted by ww
    P = []          # values of the orthogonal polynomials
    a = np.zeros(n)  # Jacobi diagonal
    b = np.zeros(n)  # Jacobi subdiagonal (b[0] unused)
    # inner product weighted by ww
    def ip(u, v):
        return np.sum(ww * u * v)
    p_prev = np.zeros_like(q)
    p_cur = np.ones_like(q) / np.sqrt(ip(np.ones_like(q), np.ones_like(q)))
    for kk in range(n):
        P.append(p_cur)
        a[kk] = ip(q * p_cur, p_cur)
        # next orthogonal polynomial
        v = (q - a[kk]) * p_cur - (b[kk] * p_prev if kk > 0 else 0.0)
        # one extra reorthogonalization to suppress round-off
        for _ in range(2):
            for k2 in range(kk + 1):
                v = v - ip(v, P[k2]) * P[k2]
        nrm = np.sqrt(ip(v, v))
        p_prev, p_cur = p_cur, v / nrm
        if kk + 1 < n:
            b[kk + 1] = nrm
    Jm = np.diag(a) + np.diag(b[1:n], 1) + np.diag(b[1:n], -1)
    ev, evec = np.linalg.eigh(Jm)
    nodes = ev
    weights = evec[0, :] ** 2
    weights = weights / weights.sum()
    # normalize to zero mean and unit variance
    m1 = np.sum(weights * nodes)
    m2 = np.sum(weights * (nodes - m1) ** 2)
    nodes = (nodes - m1) / np.sqrt(m2)
    return nodes, weights


def kurtosis_of_rule(nodes, weights):
    return float(np.sum(weights * nodes ** 4) - 3.0)


def rule_spike(a, b=1.0, w1=0.1):
    """Fixed 5-point symmetric rule {0, ±b, ±a} (weights w0, w1, w1, w2, w2).

    Constraints: total weight 1 and unit variance; the remaining freedom is the
    continuous knob a. The excess kurtosis kappa4 = 2 w1 b^4 + 2 w2 a^4 - 3 spans
    negative to positive. The support is bounded with outer nodes <= 3 sigma,
    whereas the heavy-tailed generalized-normal family (s <= 1) moves nodes out to
    14 sigma, beyond the perturbation domain; the positive-kurtosis side uses this
    family instead.
    """
    w2 = (1.0 - 2.0 * w1 * b ** 2) / (2.0 * a ** 2)
    w0 = 1.0 - 2.0 * w1 - 2.0 * w2
    assert w0 >= 0.0 and w2 >= 0.0, (a, b, w1, w0, w2)
    nodes = np.array([0.0, b, -b, a, -a])
    weights = np.array([w0, w1, w1, w2, w2])
    weights = weights / weights.sum()
    m2 = float(np.sum(weights * nodes ** 2))
    nodes = nodes / np.sqrt(m2)
    return nodes, weights


def rule_table():
    """Shape families used in the experiments: 5 nodes per dimension, covariance matching exactly."""
    out = {}
    nd, wt = gauss_rule_1d(2.0, n=5)
    out["gauss(k4=0)"] = (nd, wt)
    for a in (1.5, 1.8, 2.0, 2.4, 2.8):
        nd, wt = rule_spike(a)
        out["spike(a=%.1f)" % a] = (nd, wt)
    return out


def shapes_table(s_list=(0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 6.0, 25.0)):
    out = {}
    for s in s_list:
        nd, wt = gauss_rule_1d(s, n=5)
        out[f"s={s:g}"] = dict(s=s, nodes=nd.tolist(), weights=wt.tolist(),
                               kurtosis4=kurtosis_of_rule(nd, wt),
                               kurtosis4_analytic=float(gennorm_kurtosis(s)))
    return out


# ------------------------------------------------- 3D ensembles and moments
def build_cloud(nodes, weights, Sigma0, mean=(0.0, 0.0, 0.0)):
    """3D tensor-product cloud with covariance exactly Sigma0 (tensor-product rule -> no cross-correlation)."""
    L = np.linalg.cholesky(Sigma0)
    pts = []
    wts = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            for k, c in enumerate(nodes):
                pts.append([a, b, c])
                wts.append(weights[i] * weights[j] * weights[k])
    Z = np.array(pts)
    W = np.array(wts)
    X = Z @ L.T + np.array(mean)
    return X, W


def weighted_moments(X, W):
    Wn = W / W.sum()
    mu = np.sum(Wn[:, None] * X, axis=0)
    dX = X - mu
    cov = (dX * Wn[:, None]).T @ dX
    # fourth-order cumulant tensor Lambda[p,q,r,s]
    C4 = np.einsum("n,ni,nj,nk,nl->ijkl", Wn, dX, dX, dX, dX)
    Wick = (np.einsum("ij,kl->ijkl", cov, cov) + np.einsum("ik,jl->ijkl", cov, cov)
            + np.einsum("il,jk->ijkl", cov, cov))
    Lam = C4 - Wick
    return mu, cov, Lam


def sym_eig(M):
    ev, evec = np.linalg.eigh(0.5 * (M + M.T))
    o = np.argsort(-ev)
    return ev[o], evec[:, o]


def save_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=float)
    print("saved:", p)
    return p


def load_json(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)
