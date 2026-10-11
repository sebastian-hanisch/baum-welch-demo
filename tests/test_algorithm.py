"""Einheiten des Lernkerns gegen Handrechnung und Randfälle."""

import numpy as np
import pytest

import bwl_algorithm as A
import bwl_constants as C
import bwl_scenario as S


def test_hand_weights_hand_values():
    w = A.hand_weights(4, 10.0, 5.0)
    assert w == pytest.approx([np.exp(-2.0), 1.0, np.exp(-2.0), np.exp(-4.0)])


def test_start_weights_named_and_random():
    assert A.start_weights("gleich", 0).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert A.start_weights("von Hand", 0) == pytest.approx(A.hand_weights())
    assert A.start_weights("Stehenbleiben bevorzugt", 0)[0] == 1.0
    r1, r2 = A.start_weights("zufällig", 3), A.start_weights("zufällig", 3)
    assert r1.tolist() == r2.tolist() and A.start_weights("zufällig", 4).tolist() != r1.tolist()


def test_structure_counts_on_the_mini_net(mini_net, mini_struct):
    assert len(mini_net.coords) == 8
    assert np.all(mini_struct.count[:, 0] == 1)                     # jede Zelle ist genau einmal sie selbst
    assert np.all(mini_struct.cls[mini_struct.cls >= 0] >= 0)
    for i in range(8):
        for h in range(4):
            assert mini_struct.count[i, h] == int((mini_net.hops[i] == h).sum())


def test_row_weights_hand_example(mini_struct):
    wk, r = A.row_weights(mini_struct, [0.5, 1.0, 0.25, 0.0])
    for i in range(8):
        assert r[i] == pytest.approx(0.5 * mini_struct.count[i, 0] + 1.0 * mini_struct.count[i, 1] + 0.25 * mini_struct.count[i, 2])
    assert np.all(wk[mini_struct.cls < 0] == 0.0)


def test_m_step_hand_example():
    """Zwei Klassen, jede Zelle hat genau je einen Nachbarn jeder Klasse: Z_i = w_0 + w_1. Erwartete Schritte (3; 1), erwartete Besuche gesamt 4.

    Q(w) = 3 log w_0 + 1 log w_1 - 4 log(w_0 + w_1): Maximum bei Verhältnis w_0 : w_1 = 3 : 1 (Handrechnung: dQ/dw_0 = 3/w_0 - 4/(w_0+w_1) = 0).
    """
    count = np.ones((5, 2))
    G = np.full(5, 0.8)
    w = A.m_step(np.array([3.0, 1.0]), G, count, np.array([1.0, 1.0]), iters=200)
    assert w == pytest.approx([1.0, 1.0 / 3.0], abs=1e-4)


def test_m_step_never_decreases_the_expected_log_likelihood():
    rng = np.random.default_rng(0)
    for _ in range(30):
        count = rng.integers(0, 4, (6, 4)).astype(float)
        count[:, 0] = 1.0
        G = rng.random(6) + 0.2
        Nh = np.zeros(4)
        for i in range(6):                                  # passende erwartete Schrittzahlen: nur Klassen, die es in Zeile i gibt
            p = rng.random(4) * (count[i] > 0)
            Nh += G[i] * p / p.sum()
        w0 = rng.random(4) + 0.1
        w1 = A.m_step(Nh, G, count, w0, iters=1)
        assert A.expected_q(Nh, G, count, w1) >= A.expected_q(Nh, G, count, w0) - 1e-9


def test_e_step_posteriors_are_distributions(mini_net, mini_struct):
    rng = np.random.default_rng(1)
    Z = mini_net.coords[[0, 1, 5, 4, 3]] + rng.normal(0, 6, (5, 2))
    gamma, Nh, G, ll = A.e_step(mini_net, mini_struct, [0.2, 1.0, 0.3, 0.1], Z, 6.0)
    assert gamma.shape == (5, 8) and gamma.sum(axis=1) == pytest.approx(np.ones(5)) and np.isfinite(ll)
    assert Nh.sum() == pytest.approx(4.0)                           # T - 1 Schritte insgesamt
    assert G.sum() == pytest.approx(4.0)


def test_simulation_is_reproducible():
    net = A.build_network(C.NODES, C.SPACING, C.CELL, C.MAX_HOPS)
    a, b = S.simulate(net, 7, 20.0), S.simulate(net, 7, 20.0)
    assert np.array_equal(a.path, b.path) and np.array_equal(a.z, b.z)


def test_constants_are_consistent():
    assert C.DEFAULT_START in C.START_CHOICES and C.DEFAULT_TRAIN in C.TRAIN_CHOICES
    assert C.SPACING % int(C.CELL) == 0 and C.N_CLASSES == 4 and C.N_CLASSES - 1 <= C.MAX_HOPS
    assert set(C.DATA_SWEEP) <= set(C.TRAIN_CHOICES) | {40}
