"""Baum-Welch: Übergänge aus den Daten lernen, und wann das nichts bringt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dasselbe Straßengitter wie im Stück „HMM-Dekodieren“. Dort waren die Übergangsgewichte von Hand gesetzt; hier werden sie aus Trainingsfahrten
gelernt (Baum-Welch, ein EM-Verfahren) und auf Testfahrten gegen die Handparameter verglichen.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import bwl_constants as C
import bwl_evaluation as ev
from bwl_evaluation import Settings, analyse
from bwl_pdf_export import build_pdf
from bwl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bwl_visualization import CLASS_LABELS, build_data_chart, build_ll_chart, build_noise_chart, build_starts_chart, build_weights_chart

st.set_page_config(page_title="Baum-Welch – Sebastian Hanisch", layout="wide")


def de(x, nd=1):
    """Dezimalkomma statt Punkt."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def wfmt(w):
    return "(" + "; ".join(f"{x:.2f}".replace(".", ",") for x in w) + ")"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep_noise(settings):
    return ev.sweep_noise(settings)


@st.cache_data(show_spinner=False)
def _sweep_starts(settings):
    return ev.sweep_starts(settings)


@st.cache_data(show_spinner=False)
def _sweep_data(settings):
    return ev.sweep_data(settings)


def run_experiment(key, fn, settings, label):
    """Teure Sweeps nur auf Knopfdruck: das Ergebnis bleibt sichtbar, solange die Einstellungen gleich bleiben."""
    if st.button(label, key=f"run_{key}"):
        st.session_state[f"done_{key}"] = settings
    if st.session_state.get(f"done_{key}") != settings:
        st.info("Noch nicht gerechnet für diese Einstellungen. Der Knopf startet die Rechnung (einige Sekunden); danach bleibt das Ergebnis stehen.")
        return None
    with st.spinner("Rechne … (je nach Rechner bis etwa 30 Sekunden)"):
        return fn(settings)


st.title("🎓 Baum-Welch: Übergänge aus den Daten lernen, und wann das nichts bringt")
st.markdown(
    """
Im Stück **HMM-Dekodieren** waren die Übergangswahrscheinlichkeiten **von Hand** gesetzt: „ein Fahrzeug fährt meist etwa eine Zelle je Sekunde weiter“. Das ist eine Annahme, die man falsch treffen kann.
**Baum-Welch** lernt solche Gewichte aus Daten: abwechselnd schätzt es mit dem aktuellen Modell, **wie wahrscheinlich welcher Schritt** in den Trainingsfahrten war (E-Schritt), und setzt die Gewichte
so, dass diese Schritte wahrscheinlicher werden (M-Schritt). Es ist ein **EM-Verfahren** und kann die Wahrscheinlichkeit der Daten nie senken. Die Demo prüft, ob das hilft. Sie ist das neunte Stück
der Linie **Zustandsschätzung und Filter**. Im Unterschied zu den Fall-Demos zeigt sie ein einziges Verfahren an einem wachsenden Beispiel.

Weiter unten: drei Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Baum-Welch", expanded=True):
    st.markdown(
        """
Gelernt werden vier Zahlen: das Gewicht eines Schritts über **0, 1, 2 oder 3 Zellen** (bleibt, ein Stück weiter, …). Alle anderen Teile des Modells (Straßennetz, Messmodell, Rauschen) sind bekannt und fest.

1. **E-Schritt:** Mit den aktuellen Gewichten rechnet man auf jeder Trainingsfahrt (Vorwärts-Rückwärts), wie wahrscheinlich in jeder Sekunde welcher Schritt war, und summiert das zu **erwarteten Schrittzahlen**
   je Klasse.
2. **M-Schritt:** Die neuen Gewichte sind die, die diese erwarteten Zahlen am wahrscheinlichsten machen.
3. Wiederholen, bis sich nichts mehr ändert.

**Wichtig:** EM erhöht die **Wahrscheinlichkeit der Trainingsdaten**, nicht die **Trefferquote**. Und es konvergiert gegen einen **stationären Punkt** (höchstens ein lokales Optimum), der vom Start abhängt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Daten**")
    sigma = st.slider("GPS-Rauschen (m)", *bounds("sigma_slider"), key="sigma_slider", step=5.0,
                      help="Streuung der GPS-Position um die wahre Position. Das Modell kennt diesen Wert; gelernt werden nur die Übergänge.")
    n_train = st.select_slider("Trainingsfahrten", C.TRAIN_CHOICES, key="train_select", help="Wie viele Fahrten zu je 60 Sekunden zum Lernen vorliegen.")
    st.markdown("**Das Lernen**")
    start = st.selectbox("Startwerte", C.START_CHOICES, key="start_select", help="Womit das Lernen beginnt: von Hand, alle gleich, Stehenbleiben bevorzugt oder zufällig.")
    n_iter = st.slider("EM-Iterationen", *bounds("iter_slider"), key="iter_slider", help="Wie oft E- und M-Schritt abwechseln.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Fahrten generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"sigma_slider": float(sigma), "train_select": int(n_train), "start_select": str(start), "iter_slider": int(n_iter), "seed_input": int(seed)})

settings = Settings(sigma=float(sigma), n_train=int(n_train), n_iter=int(n_iter), start=str(start), seed=int(seed))
with st.spinner("Lerne … (einige Sekunden)"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trefferquote, Handparameter", pct(a.hit["hand"][0]), delta=f"Modell-Sicherheit {pct(a.hit['hand'][1])}", delta_color="off",
          help="Auf 30 festen Testfahrten (nicht zum Lernen benutzt): Anteil der Sekunden, in denen die Schätzung höchstens 15 m (Netzabstand) neben der Wahrheit liegt.")
m2.metric("Trefferquote, gelernt", pct(a.hit["learned"][0]), delta=f"{100 * (a.hit['learned'][0] - a.hit['hand'][0]):+.0f} Punkte gegen Hand".replace(".", ","), delta_color="normal")
m3.metric("Log-Likelihood Training, gelernt", f"{a.ll_learned:.0f}", delta=f"{a.ll_learned - a.ll_hand:+.0f} gegen Handparameter".replace(".", ","), delta_color="normal",
          help="Wie wahrscheinlich die Trainingsfahrten unter dem Modell sind (je größer, desto wahrscheinlicher).")
m4.metric("Trefferquote, wahre Schrittverteilung", pct(a.hit["truth"][0]), delta=f"Log-Likelihood {a.ll_truth:.0f}", delta_color="off",
          help="Die tatsächliche Verteilung der Schrittweiten der Trainingsfahrten als Gewichte: das, was man „lernen müsste“, wenn die Wahrheit das Ziel ist.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Gewichte je Schrittweite** – das Lernen soll die grüne Verteilung (die wahre) finden.")
    st.plotly_chart(build_weights_chart(a), width="stretch", key="weights_chart")
with c2:
    st.markdown("**Log-Likelihood der Trainingsfahrten** über die EM-Iterationen – sie steigt immer.")
    st.plotly_chart(build_ll_chart(a), width="stretch", key="ll_chart")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a), file_name="baum_welch_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wird das Modell besser, wenn die Daten wahrscheinlicher werden?")
dh = a.hit["learned"][0] - a.hit["hand"][0]
if dh <= -0.03 and a.ll_learned > a.ll_hand:
    st.warning(
        f"**Nein, hier schadet das Lernen.** Die Trainingsfahrten werden wahrscheinlicher (Log-Likelihood {a.ll_learned:.0f} statt {a.ll_hand:.0f} mit den Handparametern), "
        f"die Trefferquote auf den Testfahrten sinkt aber von {pct(a.hit['hand'][0])} auf {pct(a.hit['learned'][0])}. Die wahre Schrittverteilung hat sogar die **niedrigste** "
        f"Likelihood ({a.ll_truth:.0f}) und liegt bei {pct(a.hit['truth'][0])}."
    )
elif abs(dh) < 0.03:
    st.info(f"**Kein Unterschied:** Trefferquote {pct(a.hit['learned'][0])} gelernt gegen {pct(a.hit['hand'][0])} mit den Handparametern. Hier gibt es durch das Lernen nichts zu gewinnen.")
elif dh >= 0.03:
    st.success(f"**Das Lernen hilft hier:** {pct(a.hit['learned'][0])} gegen {pct(a.hit['hand'][0])} mit den Handparametern.")
else:
    st.info(f"**Das Lernen schadet leicht:** {pct(a.hit['learned'][0])} gegen {pct(a.hit['hand'][0])} mit den Handparametern.")
st.caption("Eine einzelne Lernaufgabe zeigt nur, was hier passiert ist; die Experimente unten wiederholen das Lernen bei anderen Bedingungen.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wann gelingt das Lernen?")
st.caption(
    f"Je GPS-Rauschen (5 bis 40 m): drei Lernläufe mit verschiedenen Startwerten und die Handparameter, {settings.n_train} Trainingsfahrten, {C.SWEEP_ITER} Iterationen, "
    f"Trefferquote auf {len(C.TEST_SEEDS)} festen Testfahrten."
)
nrows = run_experiment("noise", _sweep_noise, settings, f"▶ Experiment 1 rechnen ({len(C.SIGMA_SWEEP) * 3} Lernläufe)")
if nrows is not None:
    st.plotly_chart(build_noise_chart(nrows), width="stretch", key="noise_chart")
    lo = nrows[0]
    hi = next(r for r in nrows if r["sigma"] == 30.0)
    st.info(
        f"Bei {lo['sigma']:.0f} m Rauschen lernen alle drei Starts die richtige Struktur (Gewicht fast nur auf „1 Zelle“), die Trefferquote liegt bei {pct(lo['starts']['gleich']['hit'])} und "
        f"die Handparameter sind mit {pct(lo['hand']['hit'])} ebenso gut: **nichts zu gewinnen**. Bei 30 m erreichen die gelernten Gewichte {pct(hi['starts']['gleich']['hit'])} bis "
        f"{pct(hi['starts']['von Hand']['hit'])} gegen {pct(hi['hand']['hit'])} mit den Handparametern, obwohl ihre Trainings-Likelihood **höher** ist "
        f"({hi['starts']['gleich']['ll']:.0f} gegen {hi['hand']['ll']:.0f})."
    )

st.subheader("🔬 Experiment 2: Hängt das Ergebnis vom Start ab?")
st.caption(f"Bei dem Rauschen der Seitenleiste, {settings.n_train} Trainingsfahrten, {C.SWEEP_ITER} Iterationen: sechs Starts (drei benannte, drei zufällige), gleiche Daten.")
srows = run_experiment("starts", _sweep_starts, settings, f"▶ Experiment 2 rechnen ({3 + C.RANDOM_STARTS} Lernläufe)")
if srows is not None:
    st.plotly_chart(build_starts_chart(srows), width="stretch", key="starts_chart")
    tbl = "| Start | gelernte Gewichte (bleibt; 1; 2; 3 Zellen) | Log-Likelihood | Trefferquote |\n|---|---|---|---|\n"
    for r in srows:
        tbl += f"| {r['start']} | {wfmt(r['w'])} | {r['ll']:.0f} | {pct(r['hit'])} |\n"
    st.markdown(tbl)
    lls = [r["ll"] for r in srows]
    hits = [r["hit"] for r in srows]
    st.info(
        f"Die Log-Likelihood liegt bei allen Starts in einem Bereich von nur {max(lls) - min(lls):.0f} Einheiten, die gelernten Gewichte sind aber völlig verschieden, "
        f"und die Trefferquote reicht von {pct(min(hits))} bis {pct(max(hits))}. **Die Likelihood ist flach: die Daten legen die Gewichte kaum fest, der Start entscheidet.**"
    )

st.subheader("🔬 Experiment 3: Helfen mehr Daten?")
st.caption(f"Bei dem Rauschen und dem Start der Seitenleiste, {C.SWEEP_ITER} Iterationen: 5 bis 40 Trainingsfahrten.")
drows = run_experiment("data", _sweep_data, settings, f"▶ Experiment 3 rechnen ({len(C.DATA_SWEEP)} Lernläufe)")
if drows is not None:
    st.plotly_chart(build_data_chart(drows), width="stretch", key="data_chart")
    st.info(
        f"Mit 5 Fahrten liegt die Trefferquote bei {pct(drows[0]['hit'])}, mit {drows[-1]['n']} bei {pct(drows[-1]['hit'])}; die gelernten Gewichte ändern sich kaum "
        f"({wfmt(drows[0]['w'])} gegen {wfmt(drows[-1]['w'])}). **Es ist kein Datenmengenproblem.** Bei diesem Rauschen legen die Messungen die Gewichte kaum fest, "
        "und mehr Daten machen die Likelihood nicht schärfer."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell kann die Wirklichkeit abbilden** | Hier ist das Modell einfach (nur vier Schrittgewichte, keine Richtung). Lernen findet die besten Gewichte **innerhalb** dieses Modells, nicht ein besseres Modell. | Reichere Zustände (Richtung, Geschwindigkeit) |
| **Mehr Wahrscheinlichkeit der Daten heißt besseres Modell** | Die Likelihood misst, wie gut das Modell die **Messungen** erklärt, nicht, wie gut es den **Ort** trifft (Experiment 1). | Prüfung auf Testfahrten mit bekannter Wahrheit |
| **Das Optimum ist eindeutig** | EM endet in einem stationären Punkt (höchstens ein lokales Optimum); bei verrauschten Daten ist die Likelihood flach und der Start entscheidet (Experiment 2). | Mehrere Starts, Vorwissen als Prior |
| **Genug Daten genügen** | Bei fehlangepasstem Modell ändern mehr Daten das Ergebnis kaum (Experiment 3). | – |
| **Das Rauschen ist bekannt** | Hier ist das GPS-Rauschen fest und bekannt; gleichzeitig mitzulernen verschärft die Mehrdeutigkeit. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Zustände $x_t \in \{1,\dots,N\}$ (Straßenstücke), Übergang $A_{ij} = w_{c(i,j)} / Z_i$ mit Schrittklasse $c(i,j) \in \{0,1,2,3\}$ (Zellen Abstand) und
$Z_i = \sum_h n_h(i)\,w_h$ ($n_h(i)$ = Zahl der Zellen der Klasse $h$ um $i$). Messmodell $b_j(z) \propto \exp(-\lVert z - c_j\rVert^2 / 2\sigma^2)$, $\sigma$ bekannt.

**E-Schritt.** Vorwärts-Rückwärts mit den aktuellen $w$: $\xi_t(i,j) \propto \alpha_t(i)\,A_{ij}\,b_j(z_{t+1})\,\beta_{t+1}(j)$; erwartete Schrittzahlen
$N_h = \sum_t \sum_{(i,j):\,c(i,j)=h} \xi_t(i,j)$ und erwartete Besuche $G_i = \sum_t \sum_j \xi_t(i,j)$.

**M-Schritt.** Maximiere $Q(w) = \sum_h N_h \log w_h - \sum_i G_i \log Z_i(w)$. Die Iteration
$$w_h \leftarrow \frac{N_h}{\sum_i G_i\,n_h(i)/Z_i(w)}$$
erhöht $Q$ monoton (MM-Verfahren); die Gewichte sind nur bis auf einen gemeinsamen Faktor bestimmt und werden auf größtes Gewicht 1 normiert.

**Eigenschaft.** Die marginale Log-Likelihood $\log p(z_{1:T} \mid w)$ steigt mit jeder Iteration nicht ab; EM konvergiert gegen einen stationären Punkt (höchstens ein lokales Maximum), nicht notwendig gegen das globale.

**Literatur.** L. E. Baum, T. Petrie, G. Soules, N. Weiss (1970), *A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains*, The Annals of
Mathematical Statistics 41(1), 164–171.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
