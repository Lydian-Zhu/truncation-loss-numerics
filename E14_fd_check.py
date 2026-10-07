# -*- coding: utf-8 -*-
"""Variational-equation propagator vs direct finite differences: an error-control check.

Paper location
--------------
The mechanism chapter's setup note (20_mechanism/24_sg, error-control item 2): the
propagator Dphi_t from the variational equation matches the direct finite
difference element by element (relative difference 2.7e-4 at T=3).

Method
------
(i) Variational equation: d_t (Dphi) = J(x(t)) Dphi, Dphi(0)=I, integrated to T
    with RK4 synchronized with the trajectory.
(ii) Direct finite difference: a central difference along each coordinate,
      Dphi_ij ≈ [phi_t(x0 + h e_j) - phi_t(x0 - h e_j)] / (2h).
(iii) Relative difference in the Frobenius-norm sense,
      rel = ||Dphi_var - Dphi_fd||_F / ||Dphi_fd||_F.

Output: results/E14_fd_check.json
"""
import json
import os
import sys

import numpy as np

import n01_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
T_END = 3.0
H = 0.002          # integration step, matching the main experiment
H_FD = 1e-6        # finite-difference step


def flow(x0, T, h):
    """Advance the single point x0 to T with RK4."""
    return C.integrate(np.asarray(x0, float), kappa=0.0, T=T, h=h)


def variational(x0, T, h):
    """Integrate the trajectory and variational equation together; return Dphi(T)."""
    x = np.asarray(x0, float).copy()
    D = np.eye(3)
    n = int(round(T / h))
    for _ in range(n):
        # trajectory: reuse common's RK4 (same flow as the main experiment)
        x_next = C.rk4(x, 0.0, h)
        # variational: same RK4 scheme with right-hand side J(x)D (same intermediate points)
        J0 = C.J_lorenz(x, 0.0)
        x2 = x + 0.5 * h * C.f_lorenz(x, 0.0)
        J2 = C.J_lorenz(x2, 0.0)
        x3 = x + 0.5 * h * C.f_lorenz(x2, 0.0)
        J3 = C.J_lorenz(x3, 0.0)
        x4 = x + h * C.f_lorenz(x3, 0.0)
        J4 = C.J_lorenz(x4, 0.0)
        k1 = J0 @ D
        k2 = J2 @ (D + 0.5 * h * k1)
        k3 = J3 @ (D + 0.5 * h * k2)
        k4 = J4 @ (D + h * k3)
        D = D + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x_next
    return x, D


def finite_difference(x0, T, h, eps=H_FD):
    """Central difference giving Dphi(T)."""
    x0 = np.asarray(x0, float)
    D = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        xp = flow(x0 + e, T, h)
        xm = flow(x0 - e, T, h)
        D[:, j] = (xp - xm) / (2.0 * eps)
    return D


def main():
    rng = np.random.default_rng(20261006)
    # several starting points on (near) the attractor
    x_att = np.array([1.0, 1.0, 20.0])  # near the L63 attractor
    starts = [x_att]
    # add several perturbed points and take the mean relative difference
    for _ in range(4):
        starts.append(x_att + rng.normal(0, 1.0, 3))

    recs = []
    for s in starts:
        _, Dv = variational(s, T_END, H)
        Df = finite_difference(s, T_END, H)
        rel = np.linalg.norm(Dv - Df, "fro") / np.linalg.norm(Df, "fro")
        recs.append(dict(x0=list(map(float, s)), rel_fro=float(rel)))

    rels = np.array([r["rel_fro"] for r in recs])
    out = dict(
        T=T_END, h=H, h_fd=H_FD,
        n_starts=len(recs),
        rel_fro_mean=float(rels.mean()),
        rel_fro_max=float(rels.max()),
        records=recs,
        note="variational-equation propagator vs central finite difference, relative Frobenius difference; corresponds to item 2 of the setup section.",
    )
    C.save_json("E14_fd_check.json", out)
    print("T=%.1f  h=%.4f  n=%d" % (T_END, H, len(recs)))
    print("relative diff (mean) = %.3e   relative diff (max) = %.3e" % (rels.mean(), rels.max()))
    print("wrote results/E14_fd_check.json")


if __name__ == "__main__":
    main()
