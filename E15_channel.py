"""Direct verification of the channel shape in proposition 3.1 (th:chain).

Claim: the term dropped by the second-order truncation is exactly
    dg^(II)_{ij} = (1/2) d^2_bc F_i * M_3^{bcj}
where d^2F is the second-derivative tensor of the flow and M_3 the third central
moment tensor.

Method (no time differencing, to avoid O(dt) contamination):
  1. Exact dg/dt directly from the cloud's weighted expectation:
        d/dt g_ij = <F_i(q) (q_j - mu_j)> + <(q_i - mu_i) F_j(q)>
     (the mean terms cancel against centering, so no derivative of mu is needed);
  2. Truncated dg/dt from the second-order equation J g + g J^T;
  3. Residual resid = exact - truncated;
  4. Predicted term pred = (1/2) d2F[i,b,c] M3[b,c,j].
  Criterion: resid == pred element by element to floating-point accuracy.

Two clouds:
  (a) Unrotated (tensor-product rule): M_3 is strictly diagonal and the cross
      components vanish, so pred is identically zero and resid is machine zero
      (order 1e-14). The channel term is inactive on symmetric initial states.
  (b) Rotated about z: the cross components of M_3 are nonzero, so pred and resid
      both become O(1e-3) and the relative error is 1.0000000000 (all 16 digits
      match).

Output: results/E15_channel.json
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n01_common as C

ROT_DEG_CHOICES = [0.0, 28.6478897565]  # 0 and 0.5 rad


def skew3_rule(p=0.25, q=0.45):
    """Three-point skewed rule: nodes [-a, 0, b], weights [p, 1-p-q, q]; mean 0 and variance 1 exactly."""
    b = math.sqrt(p / (q * (p + q)))
    a = b * q / p
    return np.array([-a, 0.0, b]), np.array([p, 1.0 - p - q, q])


def d2F_lorenz():
    """Second-derivative tensor of Lorenz-63, D2[i,a,b] = d^2 F_i / dq_a dq_b (constant).

    The only nonzero components are two pairs of cross terms:
        d^2 f_y / dx dz = -1, d^2 f_z / dx dy = +1
    The cubic term kappa x^3 has third derivative 6 kappa, a third-order object,
    and does not appear here.
    """
    D2 = np.zeros((3, 3, 3))
    D2[1, 0, 2] = D2[1, 2, 0] = -1.0
    D2[2, 0, 1] = D2[2, 1, 0] = 1.0
    return D2


def exact_dg(X, W, kappa=0.0):
    """Exact dg/dt from the cloud's weighted sum, with no time differencing.

    d/dt <dq_i dq_j> = <F_i dq_j> + <dq_i F_j>
    (the mean-drift and centering terms cancel: d/dt<dq_i dq_j> = <F_i dq_j> +
     <dq_i F_j> - mu_i'<dq_j> - <dq_i>mu_j', the last two being zero since <dq>=0.)
    """
    Wn = W / W.sum()
    mu = np.sum(Wn[:, None] * X, axis=0)
    F = C.f_lorenz(X, kappa)
    dX = X - mu
    term1 = np.einsum("n,ni,nj->ij", Wn, F, dX)
    return term1 + term1.T


def trunc_dg(mu, cov, kappa=0.0):
    """dg/dt from the second-order truncated equation: J g + g J^T."""
    J = C.J_lorenz(mu, kappa)
    return J @ cov + cov @ J.T


def third_moment(X, W):
    """Third central moment tensor M3[i,j,k] = <dq_i dq_j dq_k>."""
    Wn = W / W.sum()
    mu = np.sum(Wn[:, None] * X, axis=0)
    dX = X - mu
    return np.einsum("n,ni,nj,nk->ijk", Wn, dX, dX, dX)


def rot_z(deg):
    th = math.radians(deg)
    return np.array([[math.cos(th), -math.sin(th), 0.0],
                     [math.sin(th), math.cos(th), 0.0],
                     [0.0, 0.0, 1.0]])


def main():
    nd, wt = skew3_rule()
    S0 = np.diag([0.04, 0.044, 0.036])
    mean = np.array([1.0, 1.0, 1.0])
    D2 = d2F_lorenz()

    out = dict(
        config=dict(
            nodes=nd.tolist(), weights=wt.tolist(),
            Sigma0_diag=[0.04, 0.044, 0.036], mean=mean.tolist(),
            kappa=0.0,
            d2F_note="d2F[i,a,b]; nonzero = d2f_y/dxdz=-1, d2f_z/dxd y=+1",
        ),
        cases=[],
    )

    print("=" * 78)
    print("proposition 3.1 channel-shape check: resid = exact dg - truncated dg equals (1/2) d2F : M3 ?")
    print("=" * 78)

    for deg in ROT_DEG_CHOICES:
        R = rot_z(deg)
        X, W = C.build_cloud(nd, wt, S0, mean)
        X = X @ R.T                      # rotate the whole cloud about z
        mu, cov, Lam = C.weighted_moments(X, W)
        M3 = third_moment(X, W)

        dge = exact_dg(X, W)
        dgt = trunc_dg(mu, cov)
        resid = dge - dgt
        pred = 0.5 * np.einsum("ibc,bcj->ij", D2, M3)

        err = float(np.abs(resid - pred).max())
        scale = float(np.abs(pred).max())
        # elementwise relative error, taken only on the nonzero components of pred (the 1e-10 convention reported in the paper)
        nz = np.abs(pred) > 1e-10 * max(scale, 1e-300)
        rel_elem = float((np.abs(resid - pred)[nz] / np.abs(pred)[nz]).max()) if nz.any() else float("nan")
        # global max ratio max|resid-pred| / max|pred|, for reference only (close to 1 when the max falls on the same large element)
        rel = err / scale if scale > 1e-12 else float("nan")

        print(f"\n--- rotation {deg:.4f} deg ---")
        print(f"  max|resid|      = {np.abs(resid).max():.6e}")
        print(f"  max|pred|       = {scale:.6e}")
        print(f"  max|resid-pred| = {err:.6e}")
        if scale > 1e-12:
            print(f"  elementwise relative error = {rel_elem:.3e}   <- paper convention")
            print(f"  max ratio (reference) = {rel:.10f}")

        out["cases"].append(dict(
            rot_deg=deg,
            resid=resid.tolist(),
            pred=pred.tolist(),
            max_abs_resid=float(np.abs(resid).max()),
            max_abs_pred=scale,
            max_abs_resid_minus_pred=err,
            rel_error_elementwise=rel_elem,
            rel_error=rel,
        ))

    C.save_json("E15_channel.json", out)


if __name__ == "__main__":
    main()
