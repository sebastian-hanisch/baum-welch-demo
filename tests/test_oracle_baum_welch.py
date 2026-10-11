"""Orakel-Test baum-welch-demo: E-Schritt, M-Schritt und das Lernen gegen ANDERE Rechenwege.

1. Erwartete Schrittzahlen, Besuche, Log-Likelihood und Randverteilungen gegen Aufzählung aller Pfade (Mini-Netz),
2. der M-Schritt maximiert die erwartete Vollständig-Likelihood (gegen zufällige Störungen),
3. die Log-Likelihood steigt in keiner EM-Iteration (Monotonie),
4. Konsistenz: bei Daten aus dem Modell selbst mit wenig Rauschen findet EM die erzeugenden Gewichte,
5. die angezeigten Kennzahlen gegen unabhängige Neuberechnung.
"""

import itertools

import numpy as np
import pytest

import bwl_algorithm as A
import bwl_constants as C
import bwl_evaluation as ev
import bwl_scenario as S


def _dense_A(net, w, n_classes=4):
    N = len(net.coords)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            h = int(net.hops[i, j])
            if h <= n_classes - 1:
                W[i, j] = w[h]
    return W / W.sum(axis=1, keepdims=True)


def _enumerate(net, w, Z, sigma):
    """Alle Pfade: Log-Likelihood, Randverteilungen, erwartete Schrittzahlen je Klasse, erwartete Besuche (ohne letzte Zeit)."""
    N, T = len(net.coords), len(Z)
    A_mat = _dense_A(net, w)
    E = np.array([A.emission(net, z, sigma) for z in Z])
    paths = np.array(list(itertools.product(range(N), repeat=T)))
    p = E[0][paths[:, 0]] / N
    for t in range(1, T):
        p = p * A_mat[paths[:, t - 1], paths[:, t]] * E[t][paths[:, t]]
    total = p.sum()
    prob = p / total
    gamma = np.array([np.bincount(paths[:, t], weights=prob, minlength=N) for t in range(T)])
    Nh = np.zeros(4)
    for t in range(T - 1):
        cls = np.minimum(net.hops[paths[:, t], paths[:, t + 1]], 3)
        for h in range(4):
            Nh[h] += prob[cls == h].sum()
    G = sum(np.bincount(paths[:, t], weights=prob, minlength=N) for t in range(T - 1))
    return float(np.log(total)), gamma, Nh, G


def test_e_step_against_enumeration_of_all_paths(mini_net, mini_struct):
    rng = np.random.default_rng(0)
    for seed in range(4):
        w = rng.uniform(0.1, 1.0, 4)
        path = np.array([0, 1, 5, 4])
        Z = mini_net.coords[path] + np.random.default_rng(seed).normal(0, 8, (4, 2))
        gamma, Nh, G, ll = A.e_step(mini_net, mini_struct, w, Z, 8.0)
        ll_ref, gamma_ref, Nh_ref, G_ref = _enumerate(mini_net, w, Z, 8.0)
        assert ll == pytest.approx(ll_ref, abs=1e-8)
        assert gamma == pytest.approx(gamma_ref, abs=1e-9)
        assert Nh == pytest.approx(Nh_ref, abs=1e-9)
        assert G == pytest.approx(G_ref, abs=1e-9)


def _consistent_statistics(rng, n_rows=8):
    """Zufällige, aber zueinander passende Zähler: je Zeile eine Besuchsmasse G_i, verteilt nur auf Klassen, die es dort wirklich gibt (n_h(i) > 0)."""
    count = rng.integers(0, 4, (n_rows, 4)).astype(float)
    count[:, 0] = 1.0                                              # jede Zelle ist ihr eigener „bleibt“-Nachbar
    G = rng.random(n_rows) + 0.2
    Nh = np.zeros(4)
    for i in range(n_rows):
        p = rng.random(4) * (count[i] > 0)
        Nh += G[i] * p / p.sum()
    return count, Nh, G


def test_m_step_maximizes_the_expected_complete_log_likelihood():
    rng = np.random.default_rng(1)
    for _ in range(20):
        count, Nh, G = _consistent_statistics(rng)
        w = A.m_step(Nh, G, count, np.ones(4), iters=600)
        q = A.expected_q(Nh, G, count, w)
        for _ in range(60):
            alt = w * np.exp(rng.normal(0, 0.15, 4))
            assert q >= A.expected_q(Nh, G, count, alt) - 1e-9


def test_em_log_likelihood_never_decreases():
    net = A.build_network(C.NODES, C.SPACING, C.CELL, C.MAX_HOPS)
    st = A.build_structure(net, 4)
    for seed in range(3):
        Zs = [S.simulate(net, 700 + seed * 10 + k, 30.0).z[:25] for k in range(4)]
        em = A.run_em(net, st, Zs, 30.0, np.array([1.0, 0.2, 0.05, 0.01]), 12)
        ll = [h[0] for h in em.history] + [em.loglik]
        assert all(b >= a - 1e-6 for a, b in zip(ll, ll[1:])), (seed, ll)


def _sample_from_model(net, st, w, rng, T):
    wk, r = A.row_weights(st, w)
    c = int(rng.integers(len(net.coords)))
    path = [c]
    for _ in range(T - 1):
        p = wk[c] / r[c]
        c = int(st.idx[c, rng.choice(len(p), p=p)])
        path.append(c)
    return np.array(path)


def test_em_recovers_the_generating_weights_from_model_data():
    """Konsistenz: Daten aus dem Modell, wenig Rauschen (3 m bei 10-m-Zellen), genug Iterationen (EM ist bei „bleibt“ langsam)."""
    net = A.build_network(C.NODES, C.SPACING, C.CELL, C.MAX_HOPS)
    st = A.build_structure(net, 4)
    w_true = np.array([0.2, 1.0, 0.4, 0.1])
    rng = np.random.default_rng(5)
    sigma = 3.0
    Zs = []
    for _ in range(25):
        path = _sample_from_model(net, st, w_true, rng, 60)
        Zs.append(net.coords[path] + sigma * rng.standard_normal((60, 2)))
    ll_true = sum(A.e_step(net, st, w_true / w_true.max(), Z, sigma)[3] for Z in Zs)
    for start in (np.ones(4), np.array([1.0, 0.2, 0.05, 0.01])):
        em = A.run_em(net, st, Zs, sigma, start, 25)
        assert em.w == pytest.approx(w_true / w_true.max(), abs=0.1), start
        assert em.loglik >= ll_true - 0.5, start         # der geschätzte Wert erklärt die Daten mindestens so gut wie der wahre


def test_displayed_numbers_match_independent_recomputation():
    s = ev.Settings(sigma=20.0, n_train=5, n_iter=4, start="gleich", seed=3)
    a = ev.analyse(s)
    net = A.build_network(C.NODES, C.SPACING, C.CELL, C.MAX_HOPS)
    st = A.build_structure(net, 4)
    Zs = [S.simulate(net, s.seed * 100 + k, s.sigma).z for k in range(s.n_train)]
    w_hand = A.hand_weights(4, C.CELL, C.STEP_HAND_BETA)
    ll_hand = sum(A.e_step(net, st, w_hand / w_hand.max(), Z, s.sigma)[3] for Z in Zs)
    assert a.ll_hand == pytest.approx(ll_hand)
    hits = []
    for seed in C.TEST_SEEDS:
        tr = S.simulate(net, seed, s.sigma)
        gamma = A.e_step(net, st, w_hand, tr.z, s.sigma)[0]
        hits.append(np.mean(net.hops[gamma.argmax(axis=1), tr.path] * net.cell <= C.HIT_RADIUS))
    assert a.hit["hand"][0] == pytest.approx(np.mean(hits))
    assert a.ll_improved == (a.ll_learned > a.ll_start)
