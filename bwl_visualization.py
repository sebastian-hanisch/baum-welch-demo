"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import bwl_constants as C

CLASS_LABELS = ["bleibt", "1 Zelle", "2 Zellen", "3 Zellen"]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_weights_chart(a):
    """Schrittgewichte je Klasse: Start, Handparameter, gelernt, wahre Verteilung der Trainingsfahrten (jeweils auf größtes Gewicht 1 normiert)."""
    fig = go.Figure()
    for name, w, color in (("Start", a.w_start, C.COLOR_START), ("von Hand", a.w_hand, C.COLOR_HAND), ("gelernt", a.w_learned, C.COLOR_LEARNED),
                           ("wahre Schrittverteilung", a.w_truth, C.COLOR_TRUTH)):
        fig.add_trace(go.Bar(x=CLASS_LABELS, y=w, name=name, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Schrittweite in einer Sekunde")
    fig.update_yaxes(title_text="Gewicht (größtes = 1)", range=[0, 1.05])
    return _base(fig)


def build_ll_chart(a):
    ll = [h[0] for h in a.history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(ll))), y=ll, mode="lines+markers", line=dict(color=C.COLOR_LEARNED, width=3), name="Training, EM"))
    fig.add_hline(y=a.ll_hand, line=dict(color=C.COLOR_HAND, dash="dash"), annotation_text="Handparameter", annotation_position="bottom right")
    fig.add_hline(y=a.ll_truth, line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text="wahre Schrittverteilung", annotation_position="bottom right")
    fig.update_xaxes(title_text="EM-Iteration")
    fig.update_yaxes(title_text="Log-Likelihood der Trainingsfahrten")
    return _base(fig, 300)


def build_noise_chart(rows):
    xs = [r["sigma"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["hand"]["hit"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_HAND, width=3), name="von Hand"))
    for name, color, dash in (("von Hand", "#7a9fc6", "dot"), ("gleich", C.COLOR_LEARNED, "solid"), ("Stehenbleiben bevorzugt", "#2a4d78", "dash")):
        fig.add_trace(go.Scatter(x=xs, y=[r["starts"][name]["hit"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash=dash),
                                 name=f"gelernt, Start {name}"))
    fig.update_xaxes(title_text="GPS-Rauschen (m)")
    fig.update_yaxes(title_text="Trefferquote auf Testfahrten", range=[0, 1.02], tickformat=".0%")
    return _base(fig)


def build_starts_chart(rows):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["start"] for r in rows], y=[r["hit"] for r in rows], marker_color=C.COLOR_LEARNED, name="Trefferquote",
                         text=[f"{100 * r['hit']:.0f} %" for r in rows], textposition="outside"))
    fig.update_xaxes(title_text="Startwerte", type="category")
    fig.update_yaxes(title_text="Trefferquote auf Testfahrten", range=[0, 1.1], tickformat=".0%")
    return _base(fig)


def build_data_chart(rows):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[r["hit"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LEARNED, width=3), name="gelernt"))
    fig.update_xaxes(title_text="Zahl der Trainingsfahrten", type="log")
    fig.update_yaxes(title_text="Trefferquote auf Testfahrten", range=[0, 1.02], tickformat=".0%")
    return _base(fig, 300)
