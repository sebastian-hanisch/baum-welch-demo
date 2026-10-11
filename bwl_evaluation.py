"""Kennzahlen, Verdikte und Sweeps."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import bwl_algorithm as A
import bwl_constants as C
import bwl_scenario as S


@dataclass(frozen=True)
class Settings:
    sigma: float = C.DEFAULT_SIGMA         # wahres (und dem Modell bekanntes) GPS-Rauschen in m
    n_train: int = C.DEFAULT_TRAIN         # Zahl der Trainingsfahrten
    n_iter: int = C.DEFAULT_ITER           # Zahl der EM-Iterationen
    start: str = C.DEFAULT_START           # Startwerte der Schrittgewichte
    seed: int = C.DEFAULT_SEED


@lru_cache(maxsize=2)
def _setup():
    net = A.build_network(C.NODES, C.SPACING, C.CELL, C.MAX_HOPS)
    return net, A.build_structure(net, C.N_CLASSES)


def true_step_distribution(net, paths):
    """Anteil der wahren Schritte je Klasse (0, 1, 2, 3 Zellen weit) in den Trainingsfahrten."""
    counts = np.zeros(C.N_CLASSES)
    for p in paths:
        for a, b in zip(p[:-1], p[1:]):
            counts[min(int(net.hops[a, b]), C.N_CLASSES - 1)] += 1
    return counts / counts.sum()


def train_traces(net, settings, sigma=None):
    sg = settings.sigma if sigma is None else sigma
    sims = [S.simulate(net, settings.seed * 100 + k, sg) for k in range(settings.n_train)]
    return [s.z for s in sims], [s.path for s in sims]


def test_traces(net, sigma):
    return [(S.simulate(net, s, sigma).z, S.simulate(net, s, sigma).path) for s in C.TEST_SEEDS]


def evaluate(net, st, w, tests, sigma):
    """Trefferquote des Glätters (wahrscheinlichstes Stück je Sekunde) und Modell-Sicherheit auf Testfahrten."""
    hits, confs = [], []
    for Z, path in tests:
        gamma, *_ = A.e_step(net, st, w, Z, sigma)
        est = gamma.argmax(axis=1)
        hits.append(np.mean(net.hops[est, path] * net.cell <= C.HIT_RADIUS))
        near = net.hops[est] * net.cell <= C.HIT_RADIUS
        confs.append(np.mean((gamma * near).sum(axis=1)))
    return float(np.mean(hits)), float(np.mean(confs))


def log_likelihood(net, st, w, traces, sigma):
    return float(sum(A.e_step(net, st, w, Z, sigma)[3] for Z in traces))


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    w_start: np.ndarray
    w_hand: np.ndarray
    w_learned: np.ndarray
    w_truth: np.ndarray        # wahre Schrittverteilung der Trainingsfahrten (nur zum Vergleich)
    history: list
    ll_start: float
    ll_hand: float
    ll_learned: float
    ll_truth: float
    hit: dict                  # "hand" / "learned" / "truth" -> (Trefferquote, Modell-Sicherheit) auf den Testfahrten
    ll_improved: bool          # Log-Likelihood der Trainingsspuren nach dem Lernen größer als beim Start


def analyse(settings):
    net, st = _setup()
    Zs, paths = train_traces(net, settings)
    tests = test_traces(net, settings.sigma)
    w0 = A.start_weights(settings.start, settings.seed)
    em = A.run_em(net, st, Zs, settings.sigma, w0, settings.n_iter)
    w_hand = A.hand_weights(C.N_CLASSES, C.CELL, C.STEP_HAND_BETA)
    dist = true_step_distribution(net, paths)
    w_truth = np.maximum(dist / dist.max(), 1e-3)
    ll_start = log_likelihood(net, st, w0 / w0.max(), Zs, settings.sigma)
    ll = {k: log_likelihood(net, st, w, Zs, settings.sigma) for k, w in (("hand", w_hand), ("truth", w_truth))}
    hit = {k: evaluate(net, st, w, tests, settings.sigma) for k, w in (("hand", w_hand), ("learned", em.w), ("truth", w_truth))}
    return Analysis(settings, w0 / w0.max(), w_hand / w_hand.max(), em.w, w_truth, em.history, ll_start, ll["hand"], em.loglik, ll["truth"], hit,
                    em.loglik > ll_start)


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------

def _learn(net, st, Zs, tests, sigma, w0, n_iter):
    em = A.run_em(net, st, Zs, sigma, w0, n_iter)
    hit, conf = evaluate(net, st, em.w, tests, sigma)
    return {"w": em.w, "ll": em.loglik, "hit": hit, "conf": conf}


def sweep_noise(settings):
    """GPS-Rauschen variieren: gelernte Gewichte, Trainings-Log-Likelihood und Testtrefferquote für drei benannte Starts, dazu die Handparameter."""
    net, st = _setup()
    rows = []
    for sg in C.SIGMA_SWEEP:
        Zs, _ = train_traces(net, settings, sg)
        tests = test_traces(net, sg)
        wh = A.hand_weights(C.N_CLASSES, C.CELL, C.STEP_HAND_BETA)
        hit_h, conf_h = evaluate(net, st, wh, tests, sg)
        row = {"sigma": sg, "hand": {"hit": hit_h, "conf": conf_h, "ll": log_likelihood(net, st, wh / wh.max(), Zs, sg)}, "starts": {}}
        for name in ("von Hand", "gleich", "Stehenbleiben bevorzugt"):
            row["starts"][name] = _learn(net, st, Zs, tests, sg, A.start_weights(name, 0), C.SWEEP_ITER)
        rows.append(row)
    return rows


def sweep_starts(settings):
    """Verschiedene Startwerte bei dem Rauschen der Seitenleiste: wo landet das Lernen?"""
    net, st = _setup()
    Zs, _ = train_traces(net, settings)
    tests = test_traces(net, settings.sigma)
    rows = []
    names = [("von Hand", 0), ("gleich", 0), ("Stehenbleiben bevorzugt", 0)] + [("zufällig", 11 + k) for k in range(C.RANDOM_STARTS)]
    for name, s in names:
        w0 = A.start_weights(name, s)
        r = _learn(net, st, Zs, tests, settings.sigma, w0, C.SWEEP_ITER)
        rows.append({"start": name if name != "zufällig" else f"zufällig {s - 10}", "w0": w0 / w0.max(), **r})
    return rows


def sweep_data(settings):
    """Mehr Trainingsfahrten bei dem Rauschen der Seitenleiste und dem Start der Seitenleiste."""
    net, st = _setup()
    tests = test_traces(net, settings.sigma)
    rows = []
    w0 = A.start_weights(settings.start, settings.seed)
    for n in C.DATA_SWEEP:
        Zs, _ = train_traces(net, replace(settings, n_train=n))
        r = _learn(net, st, Zs, tests, settings.sigma, w0, C.SWEEP_ITER)
        rows.append({"n": n, **r})
    return rows


def check_without_direction(settings):
    """Gegenprobe zur Erklärung „fehlende Fahrtrichtung“: dieselbe Aufgabe mit einem Zufallsweg, der umkehren darf (keine Fahrtrichtung in den Daten).

    Gibt Handparameter und gelernte Gewichte samt Trefferquote und Trainings-Log-Likelihood zurück.
    """
    net, st = _setup()
    sims = [S.simulate(net, settings.seed * 100 + k, settings.sigma, allow_reverse=True) for k in range(settings.n_train)]
    tests = [(t.z, t.path) for t in (S.simulate(net, s, settings.sigma, allow_reverse=True) for s in C.TEST_SEEDS)]
    Zs = [s.z for s in sims]
    wh = A.hand_weights(C.N_CLASSES, C.CELL, C.STEP_HAND_BETA)
    em = A.run_em(net, st, Zs, settings.sigma, A.start_weights(settings.start, settings.seed), settings.n_iter)
    return {
        "w": em.w, "ll": em.loglik, "ll_hand": log_likelihood(net, st, wh / wh.max(), Zs, settings.sigma),
        "hit_learned": evaluate(net, st, em.w, tests, settings.sigma)[0], "hit_hand": evaluate(net, st, wh, tests, settings.sigma)[0],
    }
