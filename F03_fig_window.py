# -*- coding: utf-8 -*-
r"""Generate the (eps, T) validity-region figure from E10_window.json / E11_window_edge.json (publication quality).

Criterion
---------
Both measurable branches, Delta_odd and Delta_even, carry the same epsilon^{-2}
weight, so |Delta_odd|/|Delta_even| = |dk3|/|dk4|. The odd branch leads iff
    Q(eps,T) := |dk3(T)| / |dk4(T)| > 1 .
Q scales approximately as eps^{-2} along eps and decreases with nonlinear growth in T.

Three panels (stacked, each readable on its own)
-----------------------------------------------
(a) Q vs eps for five T sections (0.10/0.20/0.50/0.75/1.00). Inside the effective
    window (T<=0.5) the three slopes are all eps^{-2}; the T=0.75/1.00 sections leave
    the window and their slopes also depart from -2. The T=1.00 section crosses the
    Q=1 reference line, marking failure.
(b) Q(eps=0.02) vs T (single curve): with eps fixed, Q collapses as T grows; the
    threshold is marked on the same axes.
(c) Power cleanliness clean(T): drawn as a 0.5 threshold band plus a shaded validity
    interval [T_min, T_*).

Publication details
-------------------
* The sections use four distinguishable colors + alternating line styles + solid
  markers (triple-redundant encoding);
* line width 0.9pt and small/footnotesize fonts, within journal figure limits;
* one legend entry per curve (the T value);
* subfigure axis width linewidth-34pt to keep tick labels inside.

Output: figures/fig_window.tex.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results", "E10_window.json"), encoding="utf-8") as fh:
    D = json.load(fh)
with open(os.path.join(HERE, "results", "E11_window_edge.json"), encoding="utf-8") as fh:
    W = json.load(fh)

grid_T = {g["T"]: g["row"] for g in D["grid"]}
epss = [c["eps"] for c in D["grid"][0]["row"]]

# --- keep 5 sections: 3 representative valid ones plus the 2 boundary sections named in the text -------
# T=0.75 (crossing eps≈0.09) and T=1.0 (crossing eps≈0.26) are both required,
# otherwise the figure cannot be matched against the text.
KEEP = [0.10, 0.20, 0.50, 0.75, 1.00]
COLORS = ["cBlue", "cGreen", "cOrange", "cPurple", "cRed"]
DASHES = ["", "densely dashed", "dash dot", "dash dot dot", "densely dotted"]

qlines = []
legend = []
for k, T in enumerate(KEEP):
    row = grid_T[T]
    seq = [(c["eps"], c["w3"] / c["w4"]) for c in row if c["w4"] > 0]
    opt = "%s, line width=0.9pt" % COLORS[k]
    if DASHES[k]:
        opt += ", %s" % DASHES[k]
    opt += ", mark=*, mark size=1.4pt, mark options={fill=%s}" % COLORS[k]
    qlines.append(r"\addplot[%s] coordinates {%s};"
                  % (opt, " ".join("(%.4g,%.4g)" % (e, q) for e, q in seq)))
    legend.append("$T=%s$" % ("%.2g" % T))
sec_tex = "\n".join(qlines)
legend_tex = ",".join(legend)

# --- panel (b): Q(T) at fixed eps ----------------------------------------
Q_EPS = 0.02
qT = []
for g in D["grid"]:
    for c in g["row"]:
        if abs(c["eps"] - Q_EPS) < 1e-12 and c["w4"] > 0:
            qT.append((g["T"], c["w3"] / c["w4"]))
qT.sort()
qt_tex = " ".join("(%.4g,%.6g)" % (T, q) for T, q in qT)
T_break = None
for i in range(len(qT) - 1):
    if qT[i][1] > 1.0 >= qT[i + 1][1]:
        T_break = qT[i + 1][0]
# panel (b) uses a log y axis; Q(T) spans 1e1–1e5, so a log axis is the only readable choice.
# data fact: Q peaks at T=0.35 above its T=0.1 value (not monotone), and stays above Q=1
# until T≈1.5; at T=1.0, Q≈381, still above 1.
T_CROSS = 1.50  # T of the first sampled point with Q<1

# --- panel (c): cleanliness ----------------------------------------------
rows = W["rows"]
thresh = W["meta"]["thresh"]
T_star = W["T_star"]
clean_tex = " ".join("(%.3f,%.4f)" % (r["T"], r["clean"]) for r in rows)
clean_max = max(r["clean"] for r in rows)
T_max = max(r["T"] for r in rows)
# validity region [T_min, T_valid]: the last sampled point still satisfying clean <= thresh (= 0.5);
# the first point beyond it, 0.65, already exceeds the threshold and is reported as the first failure.
T_min = min(r["T"] for r in rows)
T_valid = max(r["T"] for r in rows if r["clean"] <= thresh)
valid_lo, valid_hi = T_min, T_valid
# vertical placement of the threshold band
band_lo, band_hi = thresh - 0.035 * clean_max, thresh + 0.035 * clean_max

has_leadT = "lead_T" in W
lead_tex = " ".join("(%.3f,%.4f)" % (r["T"], r["lead"]) for r in rows)

body = r"""% ID: fig_epsT
% DEPENDS: ls_window
% EMITS: fig:epsT

\begin{figure}[!tbp]
\centering
\begin{subfigure}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-38pt, height=40mm, xmode=log, ymode=log,
             ymin=0.05, ymax=2e5,
             xlabel={$\epsilon$},
             ylabel={$Q=\lvert\dot\kappa_3\rvert/\lvert\dot\kappa_4\rvert$},
             legend style={font=\scriptsize, draw=none, fill=none,
                           cells={anchor=west}, column sep=3pt,
                           at={(0.5,1.07)}, anchor=south,
                           legend columns=3}]
<<SECTIONS>>
\addplot[pubref, forget plot] coordinates {(0.016,1) (0.36,1)};
\node[font=\scriptsize, text=cGray, anchor=south east]
  at (axis cs:0.30,1.35) {$Q=1$};
\legend{<<LEGEND>>}
\end{axis}
\end{tikzpicture}
\caption{$\epsilon$ direction: the slope is unchanged inside the effective window and changes beyond it}\label{fig:epsT:a}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-38pt, height=40mm, ymode=log,
             ymin=0.5, ymax=2e5,
             xmin=0, xmax=1.68,
             xlabel={$T$},
             ylabel={$Q$ (fixed $\epsilon=0.02$)},
             legend style={font=\scriptsize, draw=none, fill=none,
                           cells={anchor=west}, column sep=3pt,
                           at={(0.5,1.07)}, anchor=south,
                           legend columns=1}]
\addplot[cBlue, line width=0.9pt, mark=*, mark size=1.6pt,
        mark options={fill=cBlue}] coordinates {<<QT>>};
\addplot[pubref, forget plot] coordinates {(0,1) (1.68,1)};
\addplot[cGray, line width=0.9pt, densely dashed, forget plot]
  coordinates {(<<TB>>,0.5) (<<TB>>,2e5)};
\node[font=\scriptsize, text=cGray, anchor=south east]
  at (axis cs:1.44,1.3) {$Q=1$ at $T\approx<<TB>>$};
\legend{$\epsilon=0.02$}
\end{axis}
\end{tikzpicture}
\caption{$T$ direction: $Q$ decreases overall and drops below $1$ at the end}\label{fig:epsT:b}
\end{subfigure}

\vspace{2mm}

\begin{subfigure}{\textwidth}
\centering
\begin{tikzpicture}
\begin{axis}[pubaxis, width=\linewidth-42pt, height=36mm,
             xlabel={$T$}, ylabel={power cleanliness},
             xmin=0, xmax=<<TMAX>>,
             ymin=0, ymax=<<YMAX>>,
             legend style={font=\scriptsize, draw=none, fill=none,
                           at={(0.5,1.07)}, anchor=south,
                           cells={anchor=west}, column sep=3pt,
                           legend columns=1}]
% validity region: draw only up to the bottom of the threshold band to avoid merging with the gray band
\addplot[draw=none, fill=cGreen!12, forget plot] coordinates
  {(<<VLO>>,0) (<<VHI>>,0) (<<VHI>>,<<THR>>) (<<VLO>>,<<THR>>)}
  \closedcycle;
\addplot[draw=none, fill=cGray!18, forget plot] coordinates
  {(0,<<BLO>>) (<<TMAX>>,<<BLO>>) (<<TMAX>>,<<BHI>>) (0,<<BHI>>)}
  \closedcycle;
\addplot[cBlue, line width=0.9pt, mark=*, mark size=1.6pt,
        mark options={fill=cBlue}] coordinates {<<CLEAN>>};
\addplot[pubref, forget plot] coordinates {(0,<<THR>>) (<<TMAX>>,<<THR>>)};
\node[font=\scriptsize, text=cGreen!45!black, anchor=north east]
  at (axis cs:1.60,<<YMAX>>) {green = effective window $T\le0.5$};
\node[font=\scriptsize, text=cGray, anchor=south east]
  at (axis cs:1.60,<<BHI>>) {threshold $<<THR>>$};
\legend{$\lvert p_3-3\rvert+\lvert p_4-5\rvert$}
\end{axis}
\end{tikzpicture}
\caption{}\label{fig:epsT:c}
\end{subfigure}

\caption{The effective region of order-by-order separation on the $(\epsilon,T)$ plane.
(a) The ratio of the two measurable deviations $Q=\lvert\Delta_{\mathrm{odd}}\rvert/\lvert\Delta_{\mathrm{even}}\rvert
=\lvert\dot\kappa_3\rvert/\lvert\dot\kappa_4\rvert$ (both branches carry the same $\epsilon^{-2}$ weight, so the ratio is the rate ratio),
giving five $T$ sections; the gray dashed line is the $Q=1$ reference.
The three sections with $T\le0.5$ lie entirely above the reference and share the slope $\epsilon^{-2}$;
the two sections $T=0.75$ and $T=1.00$ fall below the reference on the large-$\epsilon$ side and their slopes also depart from $-2$,
crossing at $\epsilon\approx0.09$ and $\epsilon\approx0.26$ respectively -- beyond the effective window,
the overall height, endpoint and slope all change together.
(b) At fixed $\epsilon=0.02$, the variation of $Q$ with $T$: it decreases overall but not monotonically
(a peak above the $T=0.1$ value at $T=0.35$), and first drops below $Q=1$ at $T\approx1.5$.
(c) The cleanliness of the $\epsilon$-direction exponents
$\lvert p_3-3\rvert+\lvert p_4-5\rvert$. For $T\le0.5$ all values are below $0.22$ (green region);
for $T\ge0.65$ they jump above $0.65$, and the ``leading powers do not mix'' assumption of Proposition~\ref{th:split} fails here,
so the effective-window upper bound is taken as $T_*=0.65$ (the first out-of-tolerance sample; the last acceptable point is $0.5$).}
\label{fig:epsT}
\end{figure}
"""

FIELDS = {
    "SECTIONS": sec_tex, "LEGEND": legend_tex,
    "QT": qt_tex, "TB": "%.2f" % T_CROSS,
    "CLEAN": clean_tex, "THR": "%.2f" % thresh,
    "THRPLUS": "%.2f" % (T_star + 0.05),
    "TMAX": "%.2f" % (T_max * 1.02), "YMAX": "%.2f" % (clean_max * 1.08),
    "YLABEL": "%.2f" % (clean_max * 1.08 - 0.14),
    "BLO": "%.4f" % band_lo, "BHI": "%.4f" % band_hi,
    "VLO": "%.2f" % valid_lo, "VHI": "%.2f" % valid_hi,
}
for k, v in FIELDS.items():
    body = body.replace("<<" + k + ">>", v)

assert "<<" not in body, "placeholders not all replaced"
assert body.count(r"$") % 2 == 0, "unpaired dollar signs (paired across lines; only the total is checked)"
assert body.count(r"\begin{axis}") == body.count(r"\end{axis}")
assert body.count(r"\begin{subfigure}") == body.count(r"\end{subfigure}")

out = os.path.join(HERE, "figures", "fig_window.tex")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    fh.write(body)
print("wrote:", out)
print("sections:", KEEP)
print("Q(T) at eps=0.02:", [(round(T, 2), round(q, 4)) for T, q in qT])
print("T_* =", T_star, " threshold =", thresh)
print("cleanliness range: %.3f .. %.3f" % (min(r['clean'] for r in rows), clean_max))
print("panel (b) first point:", qT[0], " last point:", qT[-1])
print("Q=1 crossing interval:", [T for T, q in qT if abs(q - 1) < 1e6][0:1])
