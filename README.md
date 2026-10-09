# Truncation-loss numerics

Code and JSON output for the manuscript

> **How higher-order moments leak into second-order predictability dynamics**
> Jinlin Zhu, School of Physics, Hubei University, Wuhan, China
> ORCID: [0009-0008-9157-3272](https://orcid.org/0009-0008-9157-3272)

Every numerical value reported in the paper is produced by one of these scripts, with no
random sampling: all integrals use fixed deterministic quadrature rules, so the output is
bit-for-bit reproducible for a given environment.

---

## Requirements

- Python ≥ 3.9
- `numpy`
- `sympy` (only for `V01_symbolic.py`)

Figures are **not** drawn with a plotting library: the `F0*` scripts emit **TikZ** source,
which the paper includes directly. No `matplotlib` is needed.

```
pip install numpy sympy
```

## Running

Scripts are run from this directory and write into `results/`:

```
python E04_phase.py          # writes results/E04_phase.json
python V01_symbolic.py       # symbolic checks; writes results/verify_symbolic.log
python F01_fig_mirror.py     # reads results/E06_mirror.json, emits TikZ
```

Order does not matter except for the `F0*` figure scripts, which read the JSON produced by
the corresponding `E*` experiment. The four `F0*` scripts write their TikZ fragments into the
`figures/` directory inside the repository (created automatically on first run).

---

## What each script produces

### Common tools

| File | Role |
|---|---|
| `n01_common.py` | Systems (Lorenz-63, the two companion systems), RK4 integrator, quadrature rules, JSON output helpers |
| `n02_shapes.py` | Construction of the initial shape families (three-point / five-point, skewness- and kurtosis-controlled) |

### Experiments

| Script | Produces (in the paper) |
|---|---|
| `E03_closure.py` | second-order closure check |
| `E04_phase.py` | the mirror pairing (**Table V**, **Table VI**); the real/imaginary contrast $|\Delta\Re\varphi_t|$, $|\Delta\Im\varphi_t|$ (**Table V**); the $\epsilon$ scaling fit ($\kappa=0$ and $\kappa=0.01$); the quadrature cumulant residual (below $10^{-16}$). Sections: `cloud`, `K_t0`, `scaling` |
| `E05_kurtosis.py` | the even-branch pairing (**Table VI**, second row) |
| `E06_mirror.py` | the mirror contrast (**Fig. 2**) and the maximum relative difference $1.5\times10^{-3}$ in its caption; also the four separated initial values of the error-control check |
| `E07_orders.py` | the powers of order-by-order separation (**Fig. 1**) |
| `E08_crosssystem.py` | the cross-system recheck (**Fig. 3**) |
| `E09_c3_coeff.py` | the coefficients $c_3,c_4$ on Lorenz-63 and their ratio, **Eq. (27)** |
| `E10_window.py`, `E11_window_edge.py` | the $(\epsilon,T)$ boundary, **Fig. 5** |
| `E12_zeros.py` | the analytic verification of the zero criterion (**Fig. 4**) |
| `E13_profile.py` | the retained-profile reading |
| `E14_fd_check.py` | the finite-difference check of the propagator (relative difference not exceeding $3\times10^{-8}$) |
| `E15_channel.py` | the direct contrast of the channel shape (**Table IV**), rotated and unrotated sets |
| `E16_step_convergence.py` | the step-convergence check (maximum relative change $1.6\times10^{-11}$) |

### Figures and symbolic verification

| Script | Role |
|---|---|
| `F01_fig_mirror.py` | TikZ for **Fig. 2**, from `results/E06_mirror.json` |
| `F02_fig_crosssystem.py` | TikZ for **Fig. 3** |
| `F03_fig_window.py` | TikZ for **Fig. 5** |
| `F04_fig_zeros.py` | TikZ for **Fig. 4**; checks the $12\to4$ curve collapse |
| `V01_symbolic.py` | SymPy, item-by-item check of the algebraic identities in the text: the cumulant expansion, the odd/even partitioning of $i^n$, the modulus identity, the spectral-decomposition identity of $\dot H_i$, the moments of the three quadrature families, the mirror identity, the $\epsilon$-power accounting, and the constant form of the Lorenz-63 channel equation. This is a determination by symbolic simplification, not a formal proof |

`results/` holds the JSON output of each experiment plus `verify_symbolic.log`.

---

## On reproducibility

- **Deterministic by construction.** No random sampling anywhere; the covariance and the
  higher moments are matched exactly element by element by the quadrature rules, so
  differences between paired states come out as floating-point zeros rather than as small
  numbers buried in noise.
- **Three independent numerical checks** run through all experiments: step convergence
  (`E16`), finite-difference verification of the propagator (`E14`), and repetition from
  four separated initial values on the attractor (`E06`).
- **Symbolic check** of the algebraic identities (`V01`).

## Licence

MIT — see `LICENSE`. You are free to use, modify and redistribute this code with
attribution.
