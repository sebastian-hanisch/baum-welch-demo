"""Baum-Welch (EM) für die Übergangsgewichte eines HMM auf einem Straßengitter, numpy-only. Jede Funktion ist eine Einheit.

Zustände sind Straßenstücke („Zellen“) von 10 m. Die Übergänge sind über Schrittklassen gekoppelt: das Gewicht eines Übergangs hängt nur davon ab, wie viele
Zellen weit er geht (0 = bleiben, 1, 2, 3); je Zeile werden die Gewichte normiert. Gelernt werden diese vier Gewichte; das Messmodell (Gauß um die Zellenmitte
mit bekanntem Rauschen) ist fest.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bwl_constants as C


@dataclass(frozen=True)
class Network:
    coords: np.ndarray    # (N, 2)
    adj: np.ndarray       # (N, N) bool
    hops: np.ndarray      # (N, N) uint8, max_hops + 1 = ferner
    cell: float


@lru_cache(maxsize=4)
def build_network(nodes, spacing, cell=10.0, max_hops=4):
    """Gitter mit nodes x nodes Kreuzungen; jede Straße besteht aus spacing/cell Zellen; Zellen an derselben Kreuzung sind benachbart."""
    n_cells = int(round(spacing / cell))
    coords, ends, links, starts = [], {}, [], []
    for i in range(nodes):
        for j in range(nodes):
            if i + 1 < nodes:
                links.append(((i, j), (i + 1, j)))
            if j + 1 < nodes:
                links.append(((i, j), (i, j + 1)))
    for a, b in links:
        base = len(coords)
        pa, pb = np.array(a, float) * spacing, np.array(b, float) * spacing
        for k in range(n_cells):
            coords.append(pa + (k + 0.5) / n_cells * (pb - pa))
        starts.append(base)
        ends.setdefault(a, []).append(base)
        ends.setdefault(b, []).append(base + n_cells - 1)
    N = len(coords)
    adj = np.zeros((N, N), bool)
    for base in starts:
        for k in range(n_cells - 1):
            adj[base + k, base + k + 1] = adj[base + k + 1, base + k] = True
    for cells in ends.values():
        for u in cells:
            for v in cells:
                if u != v:
                    adj[u, v] = True
    hops = np.full((N, N), max_hops + 1, np.uint8)
    reach = np.eye(N, dtype=bool)
    hops[reach] = 0
    A = adj.astype(np.float32)
    frontier = reach.astype(np.float32)
    seen = reach.copy()
    for h in range(1, max_hops + 1):
        frontier = (frontier @ A) > 0
        new = frontier & ~seen
        hops[new] = h
        seen |= frontier
        frontier = frontier.astype(np.float32)
    return Network(np.array(coords), adj, hops, cell)


@dataclass(frozen=True)
class Structure:
    idx: np.ndarray       # (N, K) Nachbarn j von i (aufgefüllt mit i)
    cls: np.ndarray       # (N, K) Schrittklasse 0..3, -1 bei Auffüllung
    count: np.ndarray     # (N, n_classes): Zahl der Nachbarn je Klasse


def build_structure(net, n_classes=4):
    """Nachbarlisten mit Schrittklassen: alle Zellen innerhalb von n_classes - 1 Schritten."""
    N = len(net.coords)
    lists = [np.flatnonzero(net.hops[i] <= n_classes - 1) for i in range(N)]
    K = max(len(x) for x in lists)
    idx = np.tile(np.arange(N)[:, None], (1, K))
    cls = np.full((N, K), -1, int)
    for i, nb in enumerate(lists):
        idx[i, :len(nb)] = nb
        cls[i, :len(nb)] = net.hops[i, nb]
    count = np.stack([(cls == h).sum(axis=1) for h in range(n_classes)], axis=1)
    return Structure(idx, cls, count)


def hand_weights(n_classes=4, cell=10.0, beta=5.0):
    """Von Hand gesetzt: Gewicht ~ exp(-|Schrittlänge - Zellenlänge| / beta)."""
    return np.exp(-np.abs(np.arange(n_classes) * cell - cell) / beta)


def start_weights(name, rng_seed, n_classes=4):
    if name == "von Hand":
        return hand_weights(n_classes)
    if name == "gleich":
        return np.ones(n_classes)
    if name == "Stehenbleiben bevorzugt":
        return np.array([1.0, 0.2, 0.05, 0.01])[:n_classes]
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(0.1, 1.0, n_classes)


def emission(net, z, sigma):
    d2 = ((net.coords - z) ** 2).sum(axis=1)
    return np.exp(-d2 / (2.0 * sigma ** 2)) + 1e-300


def row_weights(st, w):
    """Gewichte je Übergang (N, K) aus den Klassengewichten w und deren Zeilensummen r."""
    wk = np.where(st.cls >= 0, np.asarray(w, float)[np.maximum(st.cls, 0)], 0.0)
    return wk, wk.sum(axis=1)


def e_step(net, st, w, Z, sigma):
    """E-Schritt für eine Spur: Zellenwahrscheinlichkeiten gamma (T, N), erwartete Schrittzahlen je Klasse N_h, erwartete Besuche G (N) und Log-Likelihood.

    Skalierte Vorwärts-Rückwärts-Rechnung; xi_t(i, k) ~ alpha_t(i) A_ik E_{t+1}(j) beta_{t+1}(j) für den k-ten Nachbarn j von i.
    """
    N, T = len(net.coords), len(Z)
    wk, r = row_weights(st, w)
    E = np.array([emission(net, z, sigma) for z in Z])
    al = np.zeros((T, N))
    c = np.zeros(T)
    al[0] = E[0] / N
    c[0] = al[0].sum()
    al[0] /= c[0]
    for t in range(1, T):
        al[t] = (wk * (al[t - 1] / r)[st.idx]).sum(axis=1) * E[t]
        c[t] = al[t].sum()
        al[t] /= c[t]
    be = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        be[t] = (wk * (E[t + 1] * be[t + 1])[st.idx]).sum(axis=1) / r
        be[t] /= c[t + 1]
    gamma = al * be
    gamma /= gamma.sum(axis=1, keepdims=True)
    n_classes = st.count.shape[1]
    Nh = np.zeros(n_classes)
    G = np.zeros(N)
    for t in range(T - 1):
        xi = (al[t] / r)[:, None] * wk * (E[t + 1] * be[t + 1])[st.idx]
        xi /= xi.sum()
        for h in range(n_classes):
            Nh[h] += xi[st.cls == h].sum()
        G += xi.sum(axis=1)
    return gamma, Nh, G, float(np.log(c).sum())


def m_step(Nh, G, count, w, iters=20):
    """M-Schritt: Klassengewichte, die den erwarteten Vollständig-Likelihood-Wert Q(w) = sum_h N_h log w_h - sum_i G_i log Z_i(w) erhöhen (MM-Iteration).

    Z_i = sum_h count[i, h] w_h. Die Iteration w_h <- N_h / sum_i G_i count[i, h] / Z_i erhöht Q monoton. Das Ergebnis wird auf größtes Gewicht 1 normiert
    (die Zeilennormierung macht nur Verhältnisse wirksam).
    """
    w = np.maximum(np.asarray(w, float), 1e-9)
    n_classes = len(w)
    for _ in range(iters):
        Z = count @ w
        denom = np.array([(G * count[:, h] / Z).sum() for h in range(n_classes)])
        w = np.where(denom > 0, Nh / np.maximum(denom, 1e-300), 0.0)
        w = np.maximum(w, 1e-9)
        w = w / w.max()
    return w


def expected_q(Nh, G, count, w):
    """Erwartete Vollständig-Log-Likelihood Q(w) bis auf Konstanten, für Gegenproben."""
    w = np.maximum(np.asarray(w, float), 1e-12)
    return float((Nh * np.log(w)).sum() - (G * np.log(count @ w)).sum())


@dataclass(frozen=True)
class EmRun:
    w: np.ndarray             # (n_classes,) gelernte Gewichte, größtes = 1
    history: list             # je Iteration (Log-Likelihood der Trainingsspuren VOR dem Schritt, Gewichte NACH dem Schritt)
    loglik: float             # Log-Likelihood der Trainingsspuren unter den End-Gewichten


def run_em(net, st, traces, sigma, w0, n_iter):
    """Baum-Welch über mehrere Trainingsspuren: abwechselnd E-Schritt (alle Spuren) und M-Schritt."""
    w = np.asarray(w0, float) / np.max(w0)
    hist = []
    for _ in range(n_iter):
        Nh = np.zeros(len(w))
        G = np.zeros(len(net.coords))
        ll = 0.0
        for Z in traces:
            _, nh, g, l = e_step(net, st, w, Z, sigma)
            Nh += nh
            G += g
            ll += l
        w = m_step(Nh, G, st.count, w)
        hist.append((ll, w.copy()))
    final = sum(e_step(net, st, w, Z, sigma)[3] for Z in traces)
    return EmRun(w, hist, float(final))
