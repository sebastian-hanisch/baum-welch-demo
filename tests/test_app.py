"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_training_size_and_start_does_not_raise():
    import bwl_constants as C

    first = _fresh_app()
    assert len(first.select_slider) == 1 and len(first.selectbox) == 1
    for n in C.TRAIN_CHOICES:
        at = _fresh_app()
        at.select_slider[0].set_value(n).run(timeout=TIMEOUT)
        assert not at.exception, (n, [str(e) for e in at.exception])
    for start in C.START_CHOICES:
        at = _fresh_app()
        at.selectbox[0].set_value(start).run(timeout=TIMEOUT)
        assert not at.exception, (start, [str(e) for e in at.exception])


def test_extreme_combination_does_not_raise():
    at = _fresh_app()
    for s in at.slider:
        s.set_value(s.max)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for s in at.slider:
        s.set_value(s.min)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_experiment_buttons_compute_and_show_their_verdict():
    checks = (("run_noise", "nichts zu gewinnen"), ("run_starts", "Die Likelihood ist flach"), ("run_data", "kein Datenmengenproblem"))
    for key, marker in checks:
        at = _fresh_app()
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
        assert any(marker in i.value for i in at.info), key


def test_app_uses_the_sidebar_values_for_the_right_settings():
    """Regressionstest: angezeigte Kennzahlen gleich analyse(Settings(...)) mit denselben Reglerwerten."""
    import bwl_evaluation as ev

    at = _fresh_app()
    a = ev.analyse(ev.Settings())
    shown = " ".join(str(m.value) for m in at.metric)
    assert f"{100 * a.hit['hand'][0]:.0f} %" in shown and f"{100 * a.hit['learned'][0]:.0f} %" in shown
    at.slider(key="sigma_slider").set_value(10.0).run(timeout=TIMEOUT)
    a2 = ev.analyse(ev.Settings(sigma=10.0))
    shown2 = " ".join(str(m.value) for m in at.metric)
    assert f"{100 * a2.hit['hand'][0]:.0f} %" in shown2
