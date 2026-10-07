# -*- coding: utf-8 -*-
r"""Generate the zero-distance criterion figure from E12_zeros.json (publication quality).

Left: e_2 vs kcut/kstar for the collapse of all 12 (N,R) curves. Data fact: for a
      fixed N the R=0.5/1/2 curves are pointwise identical (the error is only a
      function of kcut/kstar; R scales kstar without changing the ratio), so the 12
      curves reduce to 4 distinct ones. The figure colors these 4 by N and presents
      the "three R curves coincide" fact explicitly.
Right: kstar = j_{N/2-1,1}/R vs N, the zero moving outward with dimension, matching
      the Airy asymptotics.

Output: figures/fig_zeros.tex.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results", "E12_zeros.json"), encoding="utf-8") as fh:
    D = json.load(fh)

rat = D["collapse"]["ratios"]
curves = D["collapse"]["curves"]


def xy(yy, nd=5):
    return " ".join("(%.3f,%s)" % (r, "%.*g" % (nd, v))
                    for r, v in zip(rat, yy) if v > 0)


# take only the 4 curves with R=1.0 (the other 8 are pointwise identical, verified)
N_COLOR = {5: "cBlue", 7: "cRed", 9: "cGreen", 11: "cPurple"}
N_MARK = {5: "*", 7: "square*", 9: "triangle*", 11: "diamond*"}
N_MARKSIZE = {5: "1.5pt", 7: "1.4pt", 9: "1.7pt", 11: "1.6pt"}

plots = []
legend_items = []
for N in sorted(N_COLOR):
    key = "N%d_R1.0" % N
    col = N_COLOR[N]
    opt = ("%s, line width=0.9pt, mark=%s, mark size=%s, mark options={fill=%s}"
           % (col, N_MARK[N], N_MARKSIZE[N], col))
    plots.append(r"\addplot[%s] coordinates {%s};" % (opt, xy(curves[key])))
    legend_items.append("$N=%d$" % N)
plots = "\n".join(plots)

# verify: the three R curves at the same N are pointwise identical
same = True
for N in sorted(N_COLOR):
    a = curves["N%d_R0.5" % N]
    b = curves["N%d_R1.0" % N]
    c = curves["N%d_R2.0" % N]
    if not (all(abs(x - y) < 1e-12 for x, y in zip(a, b))
            and all(abs(x - y) < 1e-12 for x, y in zip(a, c))):
        same = False
assert same, "the three R curves at the same N are not pointwise identical; the caption conclusion does not hold"

# right panel
ns, ks = [], []
for row in D["cases"]:
    if abs(row["R"] - 1.0) < 1e-12:
        ns.append(row["N"])
        ks.append(row["kstar"])
kstar_coords = " ".join("(%d,%.4f)" % (n, k) for n, k in sorted(zip(ns, ks)))
airy = " ".join("(%d,%.4f)" % (n, (n / 2.0 - 1.0) + 1.8558 * (n / 2.0 - 1.0) ** (1.0 / 3.0))
                for n in sorted(ns))

# divergence points: use the exact ratio points already in cases (0.5/0.8/0.95); R does not
# affect the curve shape, so reading them from the N=5 case suffices.
_ref = [r for r in D["cases"] if r["N"] == 5 and abs(r["R"] - 0.5) < 1e-12][0]
div_pts = [(k, _ref["ratios"][k]["e2"]) for k in ("0.50", "0.80", "0.95")]
div_tex = ", ".join("$%s\\to%s$" % (k.rstrip("0").rstrip(".") if k.endswith("0") else k,
                                   ("%.3f" % v) if v < 0.1 else ("%.2f" % v))
                    for k, v in div_pts)

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
                           legend columns=4},
             every axis plot/.append style={publines}]
<<ALL>>
\addplot[pubref, forget plot] coordinates {(1.0,5e-5) (1.0,1e1)};
\node[font=\scriptsize, text=cGray, anchor=south east]
  at (axis cs:0.99,1.4e0) {$\kcut=\kstar$};
\legend{<<LEGEND>>}
\end{axis}
\end{tikzpicture}
\caption{$\kcut/\kstar$ determines everything}\label{fig:zeros:a}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-40pt, height=48mm,
             xlabel={$N$}, ylabel={$\kstar$ ($R=1$)},
             legend style={font=\scriptsize, draw=none, fill=none,
                           at={(0.03,0.97)}, anchor=north west}]
\addplot[cBlue, line width=0.9pt, mark=*, mark size=1.8pt,
        mark options={fill=cBlue}] coordinates {<<KSTAR>>};
\addplot[cRed, line width=0.9pt, densely dashed, mark=triangle*,
        mark size=1.7pt, mark options={fill=cRed}] coordinates {<<AIRY>>};
\legend{exact $j_{N/2-1,1}$, Airy asymptotics}
\end{axis}
\end{tikzpicture}
\caption{zero moves outward with $N$}\label{fig:zeros:b}
\end{subfigure}

\caption{Analytic verification of the zero criterion (Hopf exact solution of the Orszag--McLaughlin model~\cite{hopf1952,marston2005}).
(a) Maximum error of the $n=2$ truncation versus $\kcut/\kstar$, where $\kstar=j_{N/2-1,1}/R$ is the analytically known zero distance.
All $12$ combinations of $N\in\{5,7,9,11\}$ and $R\in\{0.5,1,2\}$ collapse onto four curves distinguished by $N$:
at the same $N$ the three values of $R$ give pointwise identical curves (the error is only a function of $\kcut/\kstar$,
and $R$ only rescales $\kstar$ without changing this ratio), so the only visible degree of freedom is $N$.
Across all $12$ curves the median max/min ratio of the error at a fixed $\kcut/\kstar$ is $1.20$,
i.e. the validity of the truncation is determined only by $\kcut/\kstar$, independently of $N$ and $R$.
The error diverges as $\kcut/\kstar\to1$
(<<DIV>>),
which is the direct content of ``the truncation holds only for $\kcut<\kstar$''.
(b) $\kstar$ moves outward with $N$, in agreement with the Airy asymptotics $j_{\nu,1}\sim\nu+1.8558\nu^{1/3}$.
The model tends to Gaussian as $N\to\infty$, so the decrease of non-Gaussianity, the outward movement of the zero and the improvement of the second-order truncation
are the same thing.}
\label{fig:zeros}
\end{figure}
"""

FIELDS = {"ALL": plots, "LEGEND": ",".join(legend_items),
          "KSTAR": kstar_coords, "AIRY": airy, "DIV": div_tex}
for k, v in FIELDS.items():
    body = body.replace("<<" + k + ">>", v)

assert "<<" not in body, "placeholders not all replaced"
assert body.count(r"$") % 2 == 0, "unpaired dollar signs"
assert body.count(r"\begin{axis}") == body.count(r"\end{axis}")
assert body.count(r"\begin{subfigure}") == body.count(r"\end{subfigure}")

out = os.path.join(HERE, "figures", "fig_zeros.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write(body)
print("wrote:", out)
print("number of curves (deduplicated):", len(N_COLOR), "(of the original 12, the 3 R curves at the same N are pointwise identical)")
print("R coincidence check: passed")
print("error as kcut/kstar -> 1:", [(r, round(v, 4)) for r, v in div_pts])
print("kstar (R=1):", [(n, round(k, 3)) for n, k in sorted(zip(ns, ks))])
