#!/usr/bin/env python3
"""Generate editable, deterministic TikZ method figures. No simulated result is drawn here.
All examples are algebraic illustrations; the publication captions mark this boundary.
Run with Python 3.11+ and a TeX installation with TikZ/standalone.
"""
from pathlib import Path
import subprocess
import json
import math

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figure_sources'
OUT = ROOT / 'figures'
SRC.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
PRE = r'''\documentclass[tikz,border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet,amsmath,amssymb,fix-cm}
\renewcommand{\familydefault}{\sfdefault}
\usetikzlibrary{arrows.meta,calc,positioning,fit,backgrounds,decorations.pathreplacing,shapes.geometric}
\definecolor{ink}{HTML}{243343}
\definecolor{physical}{HTML}{3478A3}
\definecolor{semantic}{HTML}{B27C26}
\definecolor{feasible}{HTML}{248679}
\definecolor{decode}{HTML}{75609D}
\definecolor{fault}{HTML}{BA5554}
\definecolor{muted}{HTML}{6F7D88}
\tikzset{
  every node/.style={font=\fontsize{9}{10.8}\selectfont,text=ink},
  edge/.style={-{Stealth[length=1.8mm,width=1.25mm]},line width=.6pt,draw=ink},
  param/.style={-{Stealth[length=1.7mm]},line width=.6pt,draw=decode,dashed},
  link/.style={line width=.8pt,draw=physical},
  reject/.style={line width=.7pt,draw=fault,dashed},
  panel/.style={rounded corners=2pt,draw=muted!45,line width=.45pt,fill=white},
  box/.style={rounded corners=2pt,draw=physical!70,line width=.55pt,fill=physical!6,align=center,inner sep=3pt},
  sem/.style={box,draw=semantic!70,fill=semantic!8},
  gate/.style={box,draw=feasible!70,fill=feasible!7},
  dec/.style={box,draw=decode!70,fill=decode!7},
  tinylabel/.style={font=\fontsize{8}{9.4}\selectfont},
  smalllabel/.style={font=\fontsize{8.5}{10}\selectfont},
  heading/.style={font=\bfseries\fontsize{10}{11.5}\selectfont,anchor=north west},
  dot/.style={circle,minimum size=.43cm,inner sep=0pt,draw=physical,fill=white,line width=.8pt,font=\bfseries\fontsize{9}{10}\selectfont},
  receiver/.pic={\draw[draw=physical,line width=.8pt] (0,0)--(0,.58) (-.17,0)--(.17,0) (-.15,.39)--(0,.58)--(.15,.39);
    \draw[draw=physical!70,line width=.55pt] (-.2,.56) arc[start angle=160,end angle=200,radius=.23];
    \draw[draw=physical!70,line width=.55pt] (.2,.40) arc[start angle=-20,end angle=20,radius=.23];},
}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm]
'''
POST = '\\end{tikzpicture}\n\\end{document}\n'

def panel(x,y,w,h,title):
    return rf'\draw[panel] ({x},{y}) rectangle ++({w},{h});'+'\n'+rf'\node[heading] at ({x+.17},{y+h-.15}) {{{title}}};'+'\n'

def bars(x,y,values,labels,width=.35,step=.66,height=1.05,color='semantic'):
    s=''
    for k,(v,l) in enumerate(zip(values,labels)):
        xx=x+k*step
        s+=rf'\fill[{color}!12] ({xx},{y}) rectangle ++({width},{height});'+'\n'
        s+=rf'\fill[{color}!70] ({xx},{y}) rectangle ++({width},{height*v});'+'\n'
        s+=rf'\node[tinylabel,anchor=north] at ({xx+width/2},{y-.04}) {{{l}}};'+'\n'
    return s

# 1. Monitoring task: actual geometry, representation heterogeneity, and global ambiguity.
f1 = panel(0,4.65,8.95,4.35,r'(a) Distributed RF monitoring')
f1 += panel(9.2,4.65,8.95,4.35,r'(b) Heterogeneous report representations')
f1 += panel(0,0,8.95,4.4,r'(c) Local links, global receiver conflict')
f1 += panel(9.2,0,8.95,4.4,r'(d) Receiver-exclusive event partition')
f1 += r'''
\begin{scope}[shift={(.2,4.75)}]
  \draw[fill=physical!3,draw=physical!12] (.2,.35) rectangle (5.9,3.3);
  \draw[draw=physical!12,step=.7cm,line width=.25pt] (.2,.35) grid (5.9,3.3);
  \coordinate (s1) at (2.6,2.1); \coordinate (s2) at (4.15,1.0);
  \foreach \r in {.35,.65,.95} \draw[physical!25,line width=.45pt] (s1) circle (\r);
  \foreach \r in {.3,.6} \draw[semantic!25,line width=.45pt] (s2) circle (\r);
  \node[dot,fill=physical!16] at (s1) {$e_1$};
  \node[dot,draw=semantic,fill=semantic!15] at (s2) {$e_2$};
  \pic at (.85,.68) {receiver}; \pic at (4.95,2.55) {receiver}; \pic at (5.15,.55) {receiver};
  \node[smalllabel,anchor=north] at (.85,.63) {$R_0$};
  \node[smalllabel,anchor=west] at (5.14,2.85) {$R_1$};
  \node[smalllabel,anchor=north] at (5.15,.51) {$R_2$};
  \draw[edge,physical!70] (2.28,1.98)--(1.02,1.18);
  \draw[edge,physical!70] (2.87,2.20)--(4.72,2.88);
  \draw[edge,physical!70] (2.88,1.94)--(4.98,1.00);
  \draw[edge,semantic!80] (3.9,1.00)--(1.05,1.00);
  \node[align=left,text width=2.1cm,anchor=west,smalllabel] at (6.15,2.75)
    {One burst\\$\rightarrow$ one event};
  \node[align=left,text width=2.1cm,anchor=west,smalllabel] at (6.15,1.65)
    {Delay + bias\\+ noise};
  \node[align=left,text width=2.1cm,anchor=west,smalllabel,text=muted] at (6.15,.63)
    {Hidden source\\positions};
\end{scope}
\begin{scope}[shift={(9.38,4.8)}]
  % Every row is a two-line record.  The top banner ends at y=3.07;
  % all record content starts at y<=2.83 (minimum vertical gap 0.24 cm).
  \node[gate,text width=7.6cm,minimum height=.48cm,inner sep=2.8pt] at (4.3,3.34)
    {Same carrier; different units and class order};
  \foreach \yy/\rx/\val/\ord/\prob in {
     2.56/0/{100 MHz}/{1,2,3,4}/{.75,.10,.10,.05},
     1.56/1/{100000 kHz}/{2,3,4,1}/{.10,.10,.05,.75},
      .56/2/{$10^8$ Hz}/{3,4,1,2}/{.10,.05,.75,.10}
  } {
    \node[box,minimum width=2.70cm,minimum height=.67cm,inner sep=2pt] at (1.52,\yy)
      {\textbf{$R_{\rx}$}\quad \val};
    \draw[edge] (2.90,\yy)--(3.19,\yy);
    \node[smalllabel,anchor=west] at (3.32,\yy+.17)
      {Native class order: $(\ord)$};
    \node[smalllabel,anchor=west,text=muted] at (3.32,\yy-.18)
      {$(\prob)$};
  }
\end{scope}
\begin{scope}[shift={(.35,.3)}]
  \node[dot] (a1) at (1.0,2.45) {$a$};
  \node[dot] (d1) at (1.0,.85) {$d$};
  \node[dot,draw=semantic] (b1) at (3.10,1.65) {$b$};
  \node[dot,draw=feasible] (c1) at (5.2,2.45) {$c$};
  \draw[link] (a1)--node[above,sloped,tinylabel] {accepted} (b1);
  \draw[link] (d1)--(b1); \draw[link] (b1)--(c1);
  \draw[reject] (a1)--node[left,tinylabel,text=fault] {same Rx} (d1);
  \node[smalllabel,anchor=south] at (1,2.73) {$R_0$};
  \node[smalllabel,anchor=north] at (1,.57) {$R_0$};
  \node[smalllabel,anchor=north] at (3.1,1.32) {$R_1$};
  \node[smalllabel,anchor=south] at (5.2,2.73) {$R_2$};
  \node[align=left,text width=2.35cm,anchor=west,smalllabel] at (5.85,1.76)
    {Connected set:\\$\{a,b,c,d\}$\\[3pt]\textcolor{fault}{$R_0$ occurs twice.}};
  \node[smalllabel,text=muted,anchor=west] at (.15,.03) {Local evidence does not guarantee a valid partition.};
\end{scope}
\begin{scope}[shift={(9.45,.3)}]
  \draw[draw=feasible,fill=feasible!5,rounded corners=7pt] (.22,.9) rectangle (4.56,2.93);
  \node[dot] (a2) at (1.0,2.1) {$a$};
  \node[dot,draw=semantic] (b2) at (2.42,1.55) {$b$};
  \node[dot,draw=feasible] (c2) at (3.83,2.1) {$c$};
  \draw[link] (a2)--(b2)--(c2);\draw[draw=feasible,dashed,line width=.8pt] (c2)--(a2);
  \node[smalllabel,text=feasible] at (2.4,2.68) {Candidate event $C_1$};
  \draw[draw=decode,fill=decode!5,rounded corners=7pt] (5.0,1.30) rectangle (6.43,2.93);
  \node[dot] at (5.71,2.1) {$d$};
  \node[smalllabel,text=decode] at (5.71,2.68) {$C_2$};
  \node[smalllabel,text=muted,align=left,anchor=west,text width=1.5cm] at (6.73,2.07)
    {Singleton\\retained};
  \node[gate,text width=7.8cm,minimum height=.65cm] at (4.18,.48)
    {Every cluster: distinct receivers + all cross-pair gates};
  \node[tinylabel,text=muted,anchor=west] at (.15,-.05) {Event count is unknown; emitter identity is outside this task.};
\end{scope}
'''

# 2. Full system, separated training/calibration/inference routes.
f2 = panel(0,0,18.15,8.6,r'Physics--semantic association architecture')
f2 += r'''
\draw[rounded corners=2pt,draw=decode!40,fill=decode!3,dashed] (.25,6.5) rectangle (17.9,7.98);
\node[heading,text=decode] at (.4,7.87) {Offline};
\node[box,text width=2.45cm,minimum height=.75cm] (train) at (4.15,7.2) {Train pairs\\all cross-Rx};
\node[dec,text width=2.35cm,minimum height=.75cm] (fit) at (7.66,7.2) {Fit $(\mu,\sigma,\beta)$\\training only};
\node[box,text width=2.35cm,minimum height=.75cm] (cal) at (11.55,7.2) {Cal. scenes\\admitted pairs};
\node[dec,text width=2.35cm,minimum height=.75cm] (temp) at (15.3,7.2) {Fit scalar $T$\\frozen scorer};
\draw[edge] (train)--(fit); \draw[edge] (cal)--(temp);
\node[smalllabel,anchor=west,text=muted] at (.45,7.12) {disjoint scenes};

\draw[panel,fill=physical!3] (.25,1.03) rectangle (3.25,6.05);
\draw[panel,fill=feasible!3] (3.65,1.03) rectangle (6.85,6.05);
\draw[panel,fill=semantic!3] (7.25,1.03) rectangle (12.12,6.05);
\draw[panel,fill=decode!3] (12.52,1.03) rectangle (17.9,6.05);
\node[heading] at (.42,5.87) {1. Align};
\node[heading] at (3.82,5.87) {2. Gate};
\node[heading] at (7.42,5.87) {3. Score};
\node[heading] at (12.69,5.87) {4. Partition};

\node[box,text width=2.5cm,minimum height=.65cm] (records) at (1.75,4.80) {Native records\\receiver ID};
\node[box,text width=2.5cm,minimum height=.73cm] (canonical) at (1.75,3.64) {SI units\\class order};
\node[box,text width=2.5cm,minimum height=.74cm] (obs) at (1.75,2.38) {Aligned report\\$t,f,B,D,\pi$};
\draw[edge] (records)--(canonical); \draw[edge] (canonical)--(obs);
\node[tinylabel,align=center,text width=2.5cm] at (1.75,1.4) {No source position\\or event count};

\node[gate,text width=2.38cm,minimum height=.75cm] (bound) at (5.25,4.75) {Geometry\\bound $h_{ij}$};
\node[gate,text width=2.38cm,minimum height=.75cm] (unc) at (5.25,3.53) {Uncertainty\\$u_t,\ u_f$};
\node[gate,text width=2.38cm,minimum height=.75cm] (gate) at (5.25,2.26) {Admit if\\$z_t\leq k,\ z_f\leq k$};
\draw[edge] (bound.east)--(6.73,4.75)--(6.73,2.79)--(5.96,2.79)--(5.96,2.67);
\draw[edge] (unc)--(gate);
\draw[edge] (obs.east)--(3.45,2.38)|-(gate.west);
\node[tinylabel,align=center,text width=2.7cm] at (5.25,1.32) {Necessary test;\\not event identity};

\node[box,text width=1.70cm,minimum height=.9cm] (phys) at (8.56,4.70) {Physical\\5 features};
\node[sem,text width=1.70cm,minimum height=.9cm] (sem) at (10.78,4.70) {Semantic\\1 feature};
\node[box,text width=3.63cm,minimum height=.52cm] (six) at (9.68,3.40) {Concatenate: $\phi_{ij}\in\mathbb R^6$};
\node[dec,text width=3.63cm,minimum height=.52cm] (score) at (9.68,2.45) {Scaler + scorer $\rightarrow\ell_{ij}$};
\node[dec,text width=3.63cm,minimum height=.52cm] (prob) at (9.68,1.52) {$p^{\mathrm{cal}}_{ij}=\mathrm{sigmoid}(\ell_{ij}/T)$};
\draw[edge] (phys.south)--(8.56,3.8)-|(six.north);
\draw[edge] (sem.south)--(10.78,3.8)-|(six.north);
\draw[edge] (six)--(score); \draw[edge] (score)--(prob);
\draw[edge] (gate.east)--(7.04,2.26)--(7.04,5.31)-|(phys.north);
\draw[edge] (7.04,5.31)-|(sem.north);
\draw[param] (fit.south)--(7.66,6.25)--(11.93,6.25)--(11.93,2.45)--(score.east);
\draw[param] (temp.south)--(15.3,6.24)--(12.06,6.24)--(12.06,1.52)--(prob.east);

\node[dec,text width=4.3cm,minimum height=1.3cm] (signed) at (15.21,4.69) {};
\foreach \rr in {0,...,3} \foreach \cc in {0,...,3} {
 \pgfmathsetmacro{\xx}{13.32+.28*\cc}\pgfmathsetmacro{\yy}{4.21+.28*\rr}
 \fill[physical!30] (\xx,\yy) rectangle ++(.25,.25);
}
\foreach \dd in {0,...,3}{\pgfmathsetmacro{\xx}{13.32+.28*\dd}\pgfmathsetmacro{\yy}{4.21+.28*(3-\dd)}\fill[muted!35] (\xx,\yy) rectangle ++(.25,.25);}
\fill[semantic!40] (14.16,5.05) rectangle ++(.25,.25);
\fill[semantic!40] (13.32,4.21) rectangle ++(.25,.25);
\node[anchor=west,align=left,text width=2.55cm,smalllabel] at (14.6,4.72) {Signed weights $W$\\feasibility mask $G$};
\node[dec,text width=4.3cm,minimum height=.85cm] (merge) at (15.21,3.29) {Best positive feasible merge\\disjoint receivers + all gates};
\draw[edge] (prob.east)--(12.31,1.52)--(12.31,4.69)--(signed.west);
\draw[edge] (signed)--(merge);
\draw[edge] (merge.south)--(15.21,2.48);
\draw[rounded corners=4pt,draw=feasible,fill=feasible!5] (13.1,1.43) rectangle (15.38,2.37);
\foreach \xx/\lab in {13.5/a,14.2/b,14.93/c} \node[dot,minimum size=.34cm,tinylabel] at (\xx,1.92) {$\lab$};
\draw[rounded corners=4pt,draw=decode,fill=decode!5] (15.85,1.43) rectangle (17.2,2.37);
\node[dot,minimum size=.34cm,tinylabel] at (16.51,1.92) {$d$};
\node[tinylabel] at (15.18,1.19) {$\mathcal C=\{C_1,\ldots,C_M\}$, including singletons};
\node[smalllabel,anchor=west,text=muted] at (.45,.39)
 {Online path uses frozen parameters. Gate failures are excluded; calibration does not reopen them.};
'''

# 3. Canonicalization and provenance: three native encodings, exact permutation.
f3 = panel(0,0,18.15,6.85,r'Observation contract and representation invariance')
f3 += r'''
\draw[panel,fill=physical!3] (.25,2.15) rectangle (5.65,6.22);
\draw[panel,fill=semantic!3] (6.05,2.15) rectangle (11.65,6.22);
\draw[panel,fill=feasible!3] (12.05,2.15) rectangle (17.90,6.22);
\node[heading] at (.42,6.04) {(a) Units};
\node[heading] at (6.22,6.04) {(b) Class order};
\node[heading] at (12.22,6.04) {(c) Observable record};
\node[box,text width=1.63cm,minimum height=.5cm] (mhz) at (1.43,5.02) {$100$ MHz};
\node[box,text width=1.63cm,minimum height=.5cm] (khz) at (1.43,4.04) {$100000$ kHz};
\node[box,text width=1.63cm,minimum height=.5cm] (hz) at (1.43,3.06) {$10^8$ Hz};
\node[gate,text width=1.65cm,minimum height=.65cm] (canon) at (4.32,4.04) {$f_i=10^8$ Hz};
\draw[edge] (mhz.east)--node[above,tinylabel] {$\times10^6$} (3.01,5.02)-|(canon.north);
\draw[edge] (khz.east)--node[above,tinylabel] {$\times10^3$} (canon.west);
\draw[edge] (hz.east)--node[below,tinylabel] {$\times1$} (3.01,3.06)-|(canon.south);
\node[tinylabel,text width=4.8cm,align=center,text=muted] at (2.95,2.48) {Unit conversion does not correct receiver bias.};

\node[smalllabel,anchor=west] at (6.3,5.43) {Native order of receiver $R_1$};
\foreach \xx/\lab/\val in {6.65/2/.10,7.94/3/.10,9.23/4/.05,10.52/1/.75}{
  \node[sem,minimum width=.80cm,minimum height=.67cm,tinylabel] at (\xx,4.82) {$q_{\lab}$\\$\val$};}
\foreach \xx/\lab/\val in {6.65/1/.75,7.94/2/.10,9.23/3/.10,10.52/4/.05}{
  \node[gate,minimum width=.80cm,minimum height=.67cm,tinylabel] at (\xx,3.04) {$q_{\lab}$\\$\val$};}
\draw[edge,semantic] (6.65,4.47)--(7.94,3.41);
\draw[edge,semantic] (7.94,4.47)--(9.23,3.41);
\draw[edge,semantic] (9.23,4.47)--(10.52,3.41);
\draw[edge,semantic] (10.52,4.47)--(11.27,4.13)--(11.27,3.72)--(6.05,3.72)--(6.05,3.35)--(6.65,3.35);
% Class-order mapping is shown by edges; no annotation is drawn on top of them.
\node[tinylabel,text=muted] at (8.83,2.49) {Missing semantics: $\pi_i=(1/K,\ldots,1/K)$.};

\node[anchor=north west,align=left,text width=5.15cm,smalllabel] at (12.37,5.45) {
\textbf{Identity:} receiver $s_i$, position $\mathbf r_{s_i}$\\[5pt]
\textbf{Measurements:} $t_i,\ f_i,\ B_i,\ D_i$\\[5pt]
\textbf{Uncertainty:} measurement s.d.s;\\receiver residual bias declarations\\[5pt]
\textbf{Semantics:} canonical vector $\pi_i$};
\node[gate,text width=4.82cm,minimum height=.42cm,tinylabel] at (14.96,2.59) {Observed fields only; no event/source label};

\node[box,text width=3.1cm,minimum height=.77cm] (shared) at (2.3,1.17) {Shared receiver\\residual $b_s$};
\node[box,text width=1.44cm,minimum height=.54cm] (oa) at (5.99,1.55) {report $i$};
\node[box,text width=1.44cm,minimum height=.54cm] (ob) at (5.99,.65) {report $j$};
\draw[edge] (shared.east)--(4.60,1.17)|-(oa.west);
\draw[edge] (4.60,1.17)|-(ob.west);
\node[smalllabel,align=left,anchor=west,text width=3.55cm] at (7.3,1.13)
 {Shared biases make pairs\\statistically dependent.};
\draw[draw=fault,dashed,line width=.7pt] (11.66,.32)--(11.66,1.86);
\node[anchor=west,align=left,text width=5.1cm,smalllabel] at (12.1,1.1)
 {\textcolor{fault}{\textbf{Evaluation-only boundary}}\\True event IDs, clutter flags and scene seeds\\never enter feature construction.};
'''

# 4. Causality/uncertainty gate, exact geometric semantics.
f4 = panel(0,0,5.82,5.45,r'(a) Geometric bound')
f4 += panel(6.15,0,5.82,5.45,r'(b) Declared uncertainty')
f4 += panel(12.3,0,5.82,5.45,r'(c) Joint admission region')
f4 += r'''
\node[dot,fill=physical!12] (src) at (2.9,3.8) {$e$};
\pic at (.9,1.77) {receiver};\pic at (4.8,1.77) {receiver};
\coordinate (ri) at (.9,2.04);\coordinate (rj) at (4.8,2.04);
\draw[link] (src)--node[above left,smalllabel] {$d_i$} (ri);
\draw[link] (src)--node[above right,smalllabel] {$d_j$} (rj);
\draw[<->,physical,line width=.65pt] (.9,1.5)--node[below,smalllabel] {$\|\mathbf r_{s_i}-\mathbf r_{s_j}\|$} (4.8,1.5);
\node[smalllabel,anchor=east] at (.64,2.03) {$R_{s_i}$};
\node[smalllabel,anchor=west] at (5.05,2.03) {$R_{s_j}$};
\node[gate,text width=4.8cm,minimum height=.65cm] at (2.91,.61)
 {$|d_i-d_j|/c\leq h_{ij}$};
\node[tinylabel,text=muted] at (2.91,4.45) {Unknown source $\mathbf x_e$; known receiver baseline};

\draw[draw=physical,fill=physical!8] (8.08,3.47) rectangle (10.0,3.93);
\node[tinylabel] at (9.04,3.7) {propagation};
\draw[draw=feasible,fill=feasible!8] (6.76,2.58) rectangle (11.32,3.04);
\node[tinylabel] at (9.04,2.81) {declared admission interval};
\draw[edge] (6.57,2.14)--(11.64,2.14) node[above left,tinylabel] {$\Delta t$};
\foreach \xx/\lab in {6.76/{-h-ku_t},8.08/{-h},9.04/{0},10.0/{h},11.32/{h+ku_t}} {
  \draw[muted,line width=.5pt] (\xx,2.08)--(\xx,2.2);
  \node[tinylabel,anchor=north] at (\xx,2.0) {$\lab$};
}
\draw[edge,feasible] (8.08,3.43)--(6.76,3.10);
\draw[edge,feasible] (10,3.43)--(11.32,3.10);
\node[gate,text width=4.87cm,minimum height=.65cm] at (9.06,.61)
 {$z_t=\max(0,|\Delta t|-h)/u_t$};
\node[tinylabel,text=muted,align=center] at (9.06,4.45) {Noise is not a new path length.\\The gate is conditional on declared $u_t$.};

\begin{scope}[shift={(13.10,1.35)},x=.69cm,y=.69cm]
  \fill[feasible!12] (0,0) rectangle (3.5,3.5);
  \draw[feasible,line width=.8pt] (0,3.5)--(3.5,3.5)--(3.5,0);
  \draw[edge] (0,0)--(5.3,0) node[below,tinylabel] {$z_t$};
  \draw[edge] (0,0)--(0,4.55) node[left,tinylabel] {$z_f$};
  \node[tinylabel,anchor=north] at (3.5,-.09) {$k$};
  \node[tinylabel,anchor=east] at (-.08,3.5) {$k$};
  \fill[physical] (.8,1.3) circle (.10);\fill[physical] (2.4,2.1) circle (.10);
  \draw[semantic,line width=1pt] (1.6,2.6) ++(-.11,-.11)--++(.22,.22) ++(-.22,0)--++(.22,-.22);
  \fill[fault] (4.35,1.4) circle (.10);
  \node[tinylabel,align=left,anchor=west] at (3.95,3.5) {outside:\\reject};
  \node[tinylabel] at (1.75,.4) {admitted};
\end{scope}
\node[gate,text width=4.85cm,minimum height=.65cm] at (15.21,.61)
 {$g_{ij}=\mathbf 1[z_t\leq k]\,\mathbf 1[z_f\leq k]$};
'''

# 5. Explicit feature construction with probability bars, no invented neural block.
f5 = panel(0,0,18.15,6.4,r'Physical and semantic evidence construction')
f5 += r'''
\draw[panel,fill=physical!3] (.25,1.13) rectangle (8.15,5.77);
\draw[panel,fill=semantic!3] (8.53,1.13) rectangle (17.90,5.77);
\node[heading] at (.42,5.59) {(a) Five physical features};
\node[heading] at (8.70,5.59) {(b) Protocol compatibility};
\node[box,text width=1.97cm,minimum height=1.00cm] (reports) at (1.54,3.57)
 {Two canonical\\reports $o_i,o_j$};
\foreach \yy/\sym/\txt in {4.79/{|\Delta t|/u_t}/{arrival time},3.98/{z_t}/{excess delay},3.17/{z_f}/{carrier},2.36/{|\Delta B|/u_B}/{bandwidth},1.55/{|\Delta D|/u_D}/{duration}} {
  \node[box,text width=2.55cm,minimum height=.52cm] at (4.28,\yy) {$\sym$};
  \node[tinylabel,anchor=west,text width=1.72cm] at (5.83,\yy) {\txt};
}
\draw[edge] (reports.east)--(2.69,3.57)--(2.69,4.79)--(2.86,4.79);
\draw[edge] (2.69,3.57)--(2.69,3.98)--(2.86,3.98);
\draw[edge] (2.69,3.57)--(2.69,3.17)--(2.86,3.17);
\draw[edge] (2.69,3.17)--(2.69,2.36)--(2.86,2.36);
\draw[edge] (2.69,2.36)--(2.69,1.55)--(2.86,1.55);
\node[tinylabel,text=muted,anchor=north] at (1.54,2.89) {declared s.d.s};
\node[tinylabel,text=physical,anchor=south] at (6.57,5.04) {$\log(1+\cdot)$};
\node[smalllabel,anchor=west] at (8.89,4.94) {$\pi_i$};
\node[smalllabel,anchor=west] at (12.0,4.94) {$\pi_j$};
'''
f5 += bars(9.13,3.65,[.7,.1,.1,.1],['1','2','3','4'],width=.39,step=.57,height=1.2)
f5 += bars(12.27,3.65,[.6,.2,.1,.1],['1','2','3','4'],width=.39,step=.57,height=1.2)
f5 += r'''
\node[smalllabel,align=left,anchor=west,text width=2.32cm] at (15.00,4.22)
 {Common class order\\$q=1,\ldots,K$\\$\sum_q\pi_{iq}=1$};
\node[sem,text width=3.33cm,minimum height=.71cm] (dot) at (10.57,2.64)
 {$v_{ij}=\log(K\pi_i^{\mathsf T}\pi_j)$};
\node[sem,text width=3.65cm,minimum height=.71cm] (conc) at (15.11,2.64)
 {Concentrations $r_i,r_j$\\$r_i=1-H(\pi_i)/\log K$};
\draw[semantic,line width=.6pt] (10.08,3.30)--(10.08,3.15)--(15.11,3.15);
\draw[semantic,line width=.6pt] (13.2,3.30)--(13.2,3.15);
\fill[semantic] (13.2,3.15) circle (.025);
\fill[semantic] (10.57,3.15) circle (.025);
\draw[edge,semantic] (10.57,3.15)--(dot.north);
\draw[edge,semantic] (15.11,3.15)--(conc.north);
\node[sem,text width=7.55cm,minimum height=.50cm] (weighted) at (13.20,1.57)
 {$\phi^{\mathrm{sem}}_{ij}=r_i r_j v_{ij}$\qquad uniform input $\Rightarrow 0$};
\draw[edge,semantic] (dot.south)--(10.57,2.03)-|(weighted.north);
\draw[edge,semantic] (conc.south)--(15.11,2.03)-|(weighted.north);
\node[dec,text width=16.80cm,minimum height=.56cm] at (9.08,.49)
 {Concatenate $[\phi^{\mathrm{phys}}_{ij},\phi^{\mathrm{sem}}_{ij}]\in\mathbb R^6$\quad$\longrightarrow$\quad train-only standardization\quad$\longrightarrow$\quad fitted logit $\ell_{ij}$};
'''

# 6. Temperature calibration: numerical illustrative logits, precise invariance.
f6 = panel(0,0,8.85,5.45,r'(a) Confidence changes')
f6 += panel(9.2,0,8.95,5.45,r'(b) All signed gains share one positive scale')
f6 += r'''
\begin{scope}[shift={(1.02,1.16)},x=.78cm,y=2.50cm]
\draw[edge] (-.0,0)--(8.55,0) node[below,tinylabel] {$\ell$};
\draw[edge] (4,0)--(4,1.3) node[left,tinylabel] {$p$};
\draw[muted!40,dashed] (0,.5)--(8,.5);
\node[tinylabel,anchor=east] at (3.89,.5) {$0.5$};
\foreach \xx/\lab in {0/-4,2/-2,4/0,6/2,8/4} \node[tinylabel,anchor=north] at (\xx,-.04) {$\lab$};
\draw[physical,line width=1pt,domain=0:8,samples=81] plot (\x,{1/(1+exp(-(\x-4)))});
\draw[semantic,line width=1pt,domain=0:8,samples=81] plot (\x,{1/(1+exp(-(\x-4)/2))});
\node[tinylabel,text=physical,anchor=west] at (5.8,1.14) {$T=1$};
\node[tinylabel,text=semantic,anchor=west] at (6.45,.75) {$T=2$};
\end{scope}
\node[smalllabel,align=center,text width=7.8cm] at (4.42,.40)
 {Illustration. $T$ is fitted by calibration-set NLL.};
\node[dec,text width=7.30cm,minimum height=.66cm] (ell) at (13.68,4.10)
 {Pair logits\quad $[-2,\;1,\;3]$};
\node[dec,text width=7.30cm,minimum height=.66cm] (scaled) at (13.68,2.93)
 {$T=2$\quad $\Rightarrow$\quad $[-1,\;0.5,\;1.5]$};
\draw[edge] (ell)--node[right,tinylabel] {same signs and order} (scaled);
\node[gate,text width=7.30cm,minimum height=.66cm] (gain) at (13.68,1.75)
 {$\Delta_T(A,B)=\Delta_1(A,B)/T$};
\draw[edge] (scaled)--(gain);
\node[smalllabel,align=center,text width=7.8cm] at (13.68,.66)
 {Fixed gates; unsaturated logits; zero cutoff; fixed ties.\\Clipping and finite tolerance require separate checks.};
'''

# 7. Decoder mechanism and a checked counterexample. Distinctive visual logic, not decorative boxes.
f7 = panel(0,3.9,18.15,3.60,r'(a) Merge rule: signed gain and structural feasibility')
f7 += panel(0,0,6.45,3.60,r'(b) Four-node counterexample')
f7 += panel(6.78,0,5.48,3.60,r'(c) Greedy result: $J=5$')
f7 += panel(12.58,0,5.57,3.60,r'(d) Exact result: $J=8$')
f7 += r'''
\node[dec,text width=3.40cm,minimum height=.85cm] (candidate) at (2.28,5.70)
 {Candidate $A,B$\\signed cross-pair sum};
\node[gate,text width=3.20cm,minimum height=.85cm] (rxcheck) at (7.00,5.70)
 {Receiver sets disjoint?\\$\mathcal S(A)\cap\mathcal S(B)=\varnothing$};
\node[gate,text width=3.20cm,minimum height=.85cm] (allcheck) at (11.63,5.70)
 {Every cross pair admitted?\\$g_{ij}=1,\ i\in A,j\in B$};
\node[dec,text width=2.65cm,minimum height=.85cm] (accept) at (16.00,5.70)
 {Merge only if\\$\Delta(A,B)>0$};
\draw[edge] (candidate)--(rxcheck);\draw[edge] (rxcheck)--(allcheck);\draw[edge] (allcheck)--(accept);
\node[smalllabel,text=fault] at (7.00,4.71) {repeated receiver $\Rightarrow$ veto};
\node[smalllabel,text=fault] at (11.63,4.71) {one failed gate $\Rightarrow$ veto};
\node[tinylabel,text=muted,anchor=west] at (.43,4.19)
 {After a merge: add gains, AND feasibility. Feasibility is guaranteed; global objective optimality is not.};
\node[dot] (n1) at (1.35,2.63) {1};\node[dot] (n2) at (4.88,2.63) {2};
\node[dot] (n3) at (1.35,.93) {3};\node[dot] (n4) at (4.88,.93) {4};
\draw[link] (n1)--node[above,tinylabel] {$5$} (n2);
\draw[link] (n1)--node[left,tinylabel] {$4$} (n3);
\draw[link] (n2)--node[right,tinylabel] {$4$} (n4);
\draw[reject] (n3)--node[below,tinylabel] {$-10$} (n4);
\draw[reject] (n1)--node[pos=.30,above,sloped,tinylabel,fill=white,inner sep=1pt] {$-10$} (n4);
\draw[reject] (n2)--node[pos=.30,above,sloped,tinylabel,fill=white,inner sep=1pt] {$-10$} (n3);
\node[tinylabel,text=muted] at (3.22,.30) {All receivers distinct; all gates admit.};

\draw[draw=decode,fill=decode!7,rounded corners=5pt] (7.36,2.10) rectangle (11.68,2.95);
\node[dot] (g1) at (8.00,2.53) {1};\node[dot] (g2) at (11.02,2.53) {2};
\draw[link] (g1)--node[above,tinylabel] {$5$} (g2);
\node[dot] at (8.00,1.09) {3};\node[dot] at (11.02,1.09) {4};
\node[tinylabel] at (9.50,1.73) {$\Delta(\{1,2\},\{3\})=4-10=-6$};
\node[tinylabel,text=muted] at (9.52,.37) {No remaining positive feasible merge.};

\draw[draw=feasible,fill=feasible!7,rounded corners=5pt] (13.21,.71) rectangle (14.55,2.99);
\draw[draw=feasible,fill=feasible!7,rounded corners=5pt] (16.14,.71) rectangle (17.49,2.99);
\node[dot] (e1) at (13.88,2.53) {1};\node[dot] (e2) at (16.81,2.53) {2};
\node[dot] (e3) at (13.88,1.14) {3};\node[dot] (e4) at (16.81,1.14) {4};
\draw[link] (e1)--node[right,tinylabel] {$4$} (e3);
\draw[link] (e2)--node[right,tinylabel] {$4$} (e4);
\node[tinylabel,text=muted] at (15.35,.33) {Early greedy merge must be split.};
'''

FIGS = {'fig01_task':f1,'fig02_architecture':f2,'fig03_contract':f3,'fig04_gate':f4,
        'fig05_evidence':f5,'fig06_temperature':f6,'fig07_decoder':f7}

def main():
    audit=[]
    for name,body in FIGS.items():
        tex=SRC/(name+'.tex');tex.write_text(PRE+body+POST)
        p=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(SRC),str(tex)],cwd=ROOT,capture_output=True,text=True)
        (SRC/(name+'_build.log')).write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(f'{name} failed: '+p.stdout[-2000:])
        import shutil, fitz
        shutil.copy2(SRC/(name+'.pdf'),OUT/(name+'.pdf'))
        d=fitz.open(OUT/(name+'.pdf'));p0=d[0]
        p0.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUT/(name+'.png'))
        audit.append({'file':name,'width_pt':p0.rect.width,'height_pt':p0.rect.height,
                      'overfull':'Overfull' in (SRC/(name+'.log')).read_text(),'type':'conceptual, deterministic TikZ'})
    (ROOT/'audit'/'diagram_build.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
