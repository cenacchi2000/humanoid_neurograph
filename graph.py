# app/graph.py
from __future__ import annotations
import math, random, time
from typing import List
import numpy as np
import plotly.graph_objects as go

class NCPGraph:
    """
    Simple 3-column NCP-like network with animated 'spark' particles.
    figure(activity, t) -> Plotly Figure
      activity: list[>=6]  sensory s1..s6 in [0,1]
      t: current time (seconds) to move the sparks deterministically
    """
    def __init__(self, seed: int = 123):
        random.seed(seed)
        self.S = 6
        self.I = 10
        self.R = 3

        # fixed node coordinates
        self.xS = [0.0]*self.S
        self.yS = list(np.linspace(0.1, 0.9, self.S))
        self.xI = [0.5]*self.I
        self.yI = list(np.linspace(0.05, 0.95, self.I))
        self.xR = [1.0]*self.R
        self.yR = [0.2, 0.5, 0.8]

        # edges S->I and I->R
        self.edges_SI = [(s, i) for s in range(self.S) for i in range(self.I)]
        self.edges_IR = [(i, r) for i in range(self.I) for r in range(self.R)]

        # pick a subset of edges to host visible sparks
        self.spark_edges = random.sample(self.edges_SI, 24) + random.sample(self.edges_IR, 15)
        self.spark_speeds = {e: 0.4 + 1.2*random.random() for e in self.spark_edges}  # units/sec
        self.spark_phase  = {e: random.random() for e in self.spark_edges}

    def _edge_xy(self, e):
        if len(e) != 2: return ([], [])
        u, v = e
        if isinstance(u, tuple):  # already (x0,y0,x1,y1)
            x0,y0,x1,y1 = u
            return [x0,x1], [y0,y1]

    def _coord_S(self, s): return self.xS[s], self.yS[s]
    def _coord_I(self, i): return self.xI[i], self.yI[i]
    def _coord_R(self, r): return self.xR[r], self.yR[r]

    def figure(self, activity: List[float], t: float | None = None) -> go.Figure:
        t = t if t is not None else time.time()
        sA = activity[:self.S] if activity and len(activity) >= self.S else [0]*self.S

        # edges
        xe, ye = [], []
        for s, i in self.edges_SI:
            x0,y0 = self._coord_S(s); x1,y1 = self._coord_I(i)
            xe += [x0, x1, None]; ye += [y0, y1, None]
        for i, r in self.edges_IR:
            x0,y0 = self._coord_I(i); x1,y1 = self._coord_R(r)
            xe += [x0, x1, None]; ye += [y0, y1, None]

        # animated sparks (density from sensory mean)
        meanA = max(0.02, float(np.mean(sA)))
        xs, ys, ms = [], [], []
        for e in self.spark_edges:
            if e in self.edges_SI:
                s,i = e; x0,y0 = self._coord_S(s); x1,y1 = self._coord_I(i)
                weight = 0.6*meanA + 0.4*sA[s]
            else:
                i,r = e; x0,y0 = self._coord_I(i); x1,y1 = self._coord_R(r)
                weight = 0.6*meanA
            # parameter along edge
            v = ( (t*self.spark_speeds[e] + self.spark_phase[e]) % 1.0 )
            xs.append(x0 + v*(x1-x0)); ys.append(y0 + v*(y1-y0))
            ms.append(4 + 10*weight)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=xe, y=ye, mode="lines", line=dict(width=1.5, color="#E6A47A"),
            hoverinfo="skip", showlegend=False
        ))

        # nodes
        fig.add_trace(go.Scatter(x=self.xS, y=self.yS, mode="markers+text",
                                 marker=dict(size=16, color="#9DA8BD"),
                                 text=[f"s{i+1}" for i in range(self.S)], textposition="top center",
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=self.xI, y=self.yI, mode="markers+text",
                                 marker=dict(size=14, color="#9DA8BD"),
                                 text=[f"i{i+1}" for i in range(self.I)], textposition="top center",
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=self.xR, y=self.yR, mode="markers+text",
                                 marker=dict(size=16, color="#9DA8BD"),
                                 text=[f"r{i+1}" for i in range(self.R)], textposition="top center",
                                 hoverinfo="skip", showlegend=False))

        # sparks
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", marker=dict(size=ms, color="#ff6b6b", opacity=0.9),
            hoverinfo="skip", showlegend=False
        ))

        fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=300)
        return fig
