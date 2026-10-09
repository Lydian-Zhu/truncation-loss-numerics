# -*- coding: utf-8 -*-
r"""Generate the zero-distance criterion figure from E12_zeros.json (publication quality).

Three panels:
  (a) e_2 vs kcut/kstar for all 12 (N,R) curves.  Colour encodes N, marker shape
      encodes R.  Data fact: for a fixed N the R=0.5/1/2 curves are pointwise
      identical (the error is only a function of kcut/kstar; R scales kstar without
      changing the ratio), so the three marker shapes of one colour coincide and the
      panel shows the collapse explicitly.
  (b) The residual N dependence: the same error divided by the N=5 curve.  On a linear
      axis the three N curves are clearly separated, so the size of the remaining
      degree of freedom is readable directly.
  (c) kstar = j_{N/2-1,1}/R vs N, the zero moving outward with dimension, matching the
      Airy asymptotics.

Output: figures/fig_zeros.tex.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results", "E12_zeros.json"), encoding="utf-8") as fh:
    D = json.load(fh)

rat = D["collapse"]["ratios"]
curves = D["collapse"]["curves"]

N_COLOR = {5: "cBlue", 7: "cRed", 9: "cGreen", 11: "cPurple"}
R_MARK = {0.5: "square*", 1.0: "*", 2.0: "triangle*"}
R_MSIZE = {0.5: "1.5pt", 1.0: "1.6pt", 2.0: "1.5pt"}
NS = sorted(N_COLOR)


def xy(yy, nd=5):
    return " ".join("(%.3f,%s)" % (r, "%.*g" % (nd, v))
                    for r, v in zip(rat, yy) if v > 0)


# ---------- panel (a): all 12 curves; the R != 1 ones are forget-plot so that the
# ---------- legend samples the four N values with the R = 1 marker.
plots = []
for N in NS:
    for R in (0.5, 2.0):
        opt = ("%s, line width=0.9pt, forget plot, mark=%s, mark size=%s, "
               "mark options={fill=%s}" % (N_COLOR[N], R_MARK[R], R_MSIZE[R], N_COLOR[N]))
        plots.append(r"\addplot[%s] coordinates {%s};" % (opt, xy(curves["N%d_R%.1f" % (N, R)])))
for N in NS:
    opt = ("%s, line width=0.9pt, mark=%s, mark size=%s, mark options={fill=%s}"
           % (N_COLOR[N], R_MARK[1.0], R_MSIZE[1.0], N_COLOR[N]))
    plots.append(r"\addplot[%s] coordinates {%s};" % (opt, xy(curves["N%d_R1.0" % N])))
plots = "\n".join(plots)
legend_items = ["$N=%d$" % N for N in NS]

# ---------- panel (b): error(N)/error(N=5) ----------
base = curves["N5_R1.0"]
ratio_lines = []
for N, col in ((7, "cRed"), (9, "cGreen"), (11, "cPurple")):
    y = [c / b for c, b in zip(curves["N%d_R1.0" % N], base)]
    ratio_lines.append(
        r"\addplot[%s, line width=0.9pt, mark=*, mark size=1.6pt, "
        r"mark options={fill=%s}] coordinates {%s};" % (col, col, xy(y, nd=5)))
ratio_lines = "\n".join(ratio_lines)
# Panel (b)'s sentence is about the N = 11 curve specifically: its largest
# overshoot over N = 5 (at small kcut/kstar) and its value as kcut/kstar -> 1.
_r11 = [c / b for c, b in zip(curves["N11_R1.0"], base)]
rmax = max(_r11)
rmin = _r11[-1]

# verify: the three R curves at the same N are pointwise identical
same = True
for N in NS:
    a = curves["N%d_R0.5" % N]
    b = curves["N%d_R1.0" % N]
    c = curves["N%d_R2.0" % N]
    if not (all(abs(x - y) < 1e-12 for x, y in zip(a, b))
            and all(abs(x - y) < 1e-12 for x, y in zip(a, c))):
        same = False
assert same, "the three R curves at the same N are not pointwise identical; the caption conclusion does not hold"

# ---------- panel (c): kstar vs N ----------
ns, ks = [], []
for row in D["cases"]:
    if abs(row["R"] - 1.0) < 1e-12:
        ns.append(row["N"])
        ks.append(row["kstar"])
kstar_coords = " ".join("(%d,%.4f)" % (n, k) for n, k in sorted(zip(ns, ks)))
airy = " ".join("(%d,%.4f)" % (n, (n / 2.0 - 1.0) + 1.8558 * (n / 2.0 - 1.0) ** (1.0 / 3.0))
                for n in sorted(ns))

# divergence points: read from the N=5 case (R does not affect the curve shape)
_ref = [r for r in D["cases"] if r["N"] == 5 and abs(r["R"] - 0.5) < 1e-12][0]
div_pts = [(k, _ref["ratios"][k]["e2"]) for k in ("0.50", "0.80", "0.95")]
_div_v = ["$%s$" % (("%.3f" % v) if v < 0.1 else ("%.2f" % v)) for _k, v in div_pts]
_div_k = [(k.rstrip("0").rstrip(".") if k.endswith("0") else k) for k, _v in div_pts]
div_tex = "%s at $\\kcut/\\kstar=%s$" % (
    ", ".join(_div_v[:-1]) + " and " + _div_v[-1],
    ", ".join(_div_k[:-1]) + " and " + _div_k[-1])

body = r"""% ID: fig_zeros
% DEPENDS: th_window
% EMITS: fig:zeros

\begin{figure}[!tbp]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-40pt, height=48mm, ymode=log,
             xmin=0.05, xmax=1.06, ymin=5e-5, ymax=1e1,
             xlabel={$\kcut/\kstar$},
             ylabel={maximum truncation error},
             legend style={font=\scriptsize, draw=none, fill=none,
                           cells={anchor=west}, column sep=3pt,
                           at={(0.5,1.07)}, anchor=south,
                           legend columns=2},
             every axis plot/.append style={publines}]
<<ALL>>
\addplot[pubref, forget plot] coordinates {(1.0,5e-5) (1.0,1e1)};
\node[font=\scriptsize, text=cGray, anchor=south east]
  at (axis cs:0.99,1.4e0) {$\kcut=\kstar$};
\legend{<<LEGEND>>}
\end{axis}
\end{tikzpicture}
\caption{all $(N,R)$ collapse}\label{fig:zeros:a}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-40pt, height=48mm,
             xmin=0.05, xmax=1.06, ymin=0.99, ymax=1.26,
             xlabel={$\kcut/\kstar$},
             ylabel={error ratio to $N=5$},
             legend style={font=\scriptsize, draw=none, fill=none,
                           cells={anchor=west}, column sep=3pt,
                           at={(0.5,1.06)}, anchor=south},
             legend columns=1]
\addplot[pubref, forget plot] coordinates {(0.05,1) (1.06,1)};
\node[font=\scriptsize, text=cGray, anchor=south west]
  at (axis cs:0.44,1.003) {$N=5$};
<<RATIO>>
\legend{$N=7$,$N=9$,$N=11$}
\end{axis}
\end{tikzpicture}
\caption{residual $N$ dependence}\label{fig:zeros:b}
\end{subfigure}

\medskip
\begin{subfigure}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-40pt, height=48mm,
             xlabel={$N$}, ylabel={$\kstar$ ($R=1$)}]
\addplot[cBlue, line width=0.9pt, mark=*, mark size=1.8pt,
        mark options={fill=cBlue}] coordinates {<<KSTAR>>};
\addplot[cRed, line width=0.9pt, densely dashed, mark=triangle*,
        mark size=1.7pt, mark options={fill=cRed}] coordinates {<<AIRY>>};
\end{axis}
\end{tikzpicture}
\caption{zeros move outward with $N$}\label{fig:zeros:c}
\end{subfigure}

\caption{Analytic verification of the zero criterion (Hopf exact solution of the Orszag--McLaughlin model~\cite{hopf1952,marston2005}).
(a) Maximum error of the $n=2$ truncation versus $\kcut/\kstar$, where $\kstar=j_{N/2-1,1}/R$ is the analytically known zero distance.
All $12$ combinations of $N\in\{5,7,9,11\}$ and $R\in\{0.5,1,2\}$ are drawn:
colour marks $N$ and marker shape marks $R$ (circle $R=1$, square $R=0.5$, triangle $R=2$),
At a fixed $N$ the three marker shapes coincide pointwise, so $R$ enters only through the plotted combination $\kcut/\kstar$;
across $N$ the collapse is not exact, and the residual dependence is quantified in panel (b).
The error diverges as $\kcut/\kstar\to1$, reaching
<<DIV>>,
which is the direct content of ``the truncation holds only for $\kcut<\kstar$''.
(b) The residual $N$ dependence: the same error divided by the $N=5$ curve, so that a perfect collapse would be a flat line at $1$.
The separation is monotone in $N$ and bounded -- the $N=11$ curve exceeds $N=5$ by <<RMAX>> at small $\kcut/\kstar$,
falling to <<RMIN>> as $\kcut/\kstar\to1$ -- so $N$ is the only remaining degree of freedom, and a weak one.
(c) $\kstar$ moves outward with $N$ (solid: exact $j_{N/2-1,1}$; dashed: Airy asymptotics
$j_{\nu,1}\sim\nu+1.8558\nu^{1/3}$). The model tends to Gaussian as $N\to\infty$, so the decrease of non-Gaussianity, the outward movement of the zero and the improvement of the second-order truncation
are the same thing.}
\label{fig:zeros}
\end{figure}
"""

FIELDS = {"ALL": plots, "LEGEND": ",".join(legend_items), "RATIO": ratio_lines,
          "KSTAR": kstar_coords, "AIRY": airy, "DIV": div_tex,
          "RMAX": "$%.0f\\%%$" % (100 * (rmax - 1)), "RMIN": "$%.0f\\%%$" % (100 * (rmin - 1))}
for k, v in FIELDS.items():
    body = body.replace("<<" + k + ">>", v)

assert "<<" not in body, "placeholders not all replaced"
assert body.count(r"$") % 2 == 0, "unpaired dollar signs"
assert body.count(r"\begin{axis}") == body.count(r"\end{axis}") == 3, "axis not paired"
assert body.count(r"\begin{subfigure}") == body.count(r"\end{subfigure}") == 3

out = os.path.join(HERE, "figures", "fig_zeros.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write(body)
print("wrote:", out)
print("curves drawn:", len(plots.split("\n")), "(12 = 4 N x 3 R)")
print("R coincidence check: passed")
print("N=11 / N=5 ratio: %.4f (kcut/kstar -> 1) .. %.4f (small kcut/kstar)" % (rmin, rmax))
print("error as kcut/kstar -> 1:", [(r, round(v, 4)) for r, v in div_pts])
print("kstar (R=1):", [(n, round(k, 3)) for n, k in sorted(zip(ns, ks))])
