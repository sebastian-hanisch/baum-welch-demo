# Baum-Welch: Übergänge aus den Daten lernen, und wann das nichts bringt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-baum-welch-demo.streamlit.app/)** -->

Neuntes Stück der Linie **Zustandsschätzung und Filter** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [HMM-Dekodieren](https://github.com/sebastian-hanisch/hmm-decode-demo) fort und greift dessen benennbare Schwäche auf: **die Übergangsgewichte waren von Hand gesetzt**. Baum-Welch
(ein EM-Verfahren) lernt sie aus Trainingsfahrten. Die Demo prüft, ob das hilft, und zeigt, wann nicht. Im Unterschied zu den Fall-Demos zeigt sie ein einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Von Hand gesetzte Parameter sind angreifbar („woher weiß man, dass 5 m stimmt?“). Lernen aus Daten klingt wie die Lösung. Die Kernfrage: **Wird das Modell besser, wenn die Daten wahrscheinlicher
werden?** Die Antwort in diesem Aufbau ist ein klares Nein, mit einer Ursache, die man kennen sollte.

## Modell

Dasselbe Straßengitter wie im Stück HMM-Dekodieren (7 × 7 Kreuzungen, Straßenabstand 50 m, 420 Zellen à 10 m), dieselbe Zufallsfahrt (eine Zelle je Sekunde, nie sofort zurück), GPS mit bekanntem
Rauschen. Gelernt werden vier Zahlen: das Gewicht eines Schritts über 0, 1, 2 oder 3 Zellen; je Zeile normiert. Der E-Schritt ist die skalierte Vorwärts-Rückwärts-Rechnung mit erwarteten
Schrittzahlen je Klasse, der M-Schritt eine Iteration, die die erwartete Vollständig-Likelihood monoton erhöht. Bewertet wird auf **30 festen Testfahrten**, die nicht zum Lernen dienen.
Formeln im 📐-Expander, Kern in `bwl_algorithm.py`.

## Methodik

Drei Experimente auf Knopfdruck:

1. GPS-Rauschen von 5 bis 40 m: drei Lernläufe mit verschiedenen Startwerten gegen die Handparameter.
2. Sechs Startwerte (drei benannte, drei zufällige) bei gleichen Daten.
3. 5 bis 40 Trainingsfahrten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Standardeinstellung (30 m Rauschen, 10 Trainingsfahrten zu 60 s, 10 Iterationen, Start mit gleichen Gewichten); `tests/test_claims.py` rechnet jede nach.

| Frage | Befund |
|---|---|
| Werden die Trainingsfahrten wahrscheinlicher? | Ja: Log-Likelihood **−826** gelernt gegen **−879** mit den Handparametern. Die tatsächliche Schrittverteilung der Fahrten (fast nur „eine Zelle“) hat sogar die **niedrigste**: **−932**. |
| Wird die Zuordnung besser? | **Nein**: Trefferquote auf Testfahrten **60 %** gelernt gegen **75 %** mit den Handparametern; die wahre Schrittverteilung als Gewichte **72 %**. |
| Wann gelingt das Lernen? | Bei **5 m** Rauschen lernen alle drei Starts die richtige Struktur (Gewicht fast nur auf „eine Zelle“); Trefferquote **99 bis 100 %**, die Handparameter sind mit **99 %** genauso gut: **nichts zu gewinnen**. |
| Und bei viel Rauschen? | Bei 30 m: gelernt **61 %** (Start gleich), **73 %** (Start von Hand), **64 %** (Stehenbleiben bevorzugt) gegen **75 %** Hand; bei 40 m: **49 %** gegen **61 %**. |
| Hängt es vom Start ab? | Ja: sechs Starts, gleiche Daten, Log-Likelihood nur **7 Einheiten** auseinander (−833 bis −826), gelernte Gewichte völlig verschieden, Trefferquote **61 bis 73 %**. |
| Helfen mehr Daten? | Nein: 5 bis 40 Fahrten ergeben **61 bis 62 %** und praktisch dieselben Gewichte. |

## Warum lernt es das Falsche?

Belegt ist, **dass** es so ist, nur zum Teil, **warum**:

- **Die Likelihood ist bei diesem Rauschen flach.** Bei 30 m GPS-Rauschen (drei Zellenlängen) unterscheiden sich „bleibt“, „eine Zelle“ und „zwei Zellen“ in den Messungen kaum: ganz verschiedene Gewichte erklären die Daten gleich gut (Log-Likelihood nur 7 Einheiten auseinander), der Start entscheidet. Bei 5 m Rauschen ist sie scharf und das Lernen gelingt.
- **Maximale Likelihood ist nicht maximale Trefferquote.** Die Likelihood misst, wie gut die **Messungen** erklärt werden, nicht, wie gut der **Ort** getroffen wird.
- **Die naheliegende Erklärung „Modell ohne Fahrtrichtung“ ist nicht bestätigt.** Gegenprobe: mit einem Zufallsweg, der umkehren darf, werden die wahren Gewichte ebenfalls nicht gefunden (gelernt (1,0; 0,85; 0,53; 0,23) statt „nur eine Zelle“), die Handparameter bleiben vorn (70 % gegen 66 %), und EM endet sogar bei einer **niedrigeren** Trainings-Likelihood als die Handparameter (−751 gegen −742): ein schlechteres stationäres Ergebnis nach 10 Iterationen.
- **Konsistenz ist gegeben**, wo das Modell stimmt: stammen die Fahrten aus dem Modell selbst, werden die erzeugenden Gewichte gefunden (Orakel-Test).

## Ehrliche Grenzen

- **Die Wahrheit ist ein Zufallsweg auf einem Gitter.** Es gibt keine echten GPS-Spuren; die Aussage gilt für diese Modellfehlanpassung, nicht für jedes Lernproblem.
- **Die Ursache ist nur teilweise geklärt.** Flache Likelihood bei viel Rauschen ist belegt; ob ein Zustand mit Richtung oder Geschwindigkeit das Lernen verbessern würde, ist nicht gemessen.
- **Nur vier Parameter und festes Rauschen.** Das Rauschen mitzulernen oder allgemeine Übergangsmatrizen sind nicht Teil der Demo.
- **EM braucht Iterationen:** bei kleinem Gewicht auf „bleibt“ konvergiert es langsam; die Demo rechnet 8 bis 10 Iterationen, der Konsistenz-Test 25.
- **Kleine Stichproben:** 30 Testfahrten; einzelne Prozentpunkte sind Zufall.

## Verifikation

- **Kern:** Handgewichte, Startwerte, Nachbarschaftsstruktur, M-Schritt (Handbeispiel mit Verhältnis 3 : 1), Monotonie des M-Schritts, Normierung der Schrittzahlen (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_baum_welch.py`, anderer Rechenweg):
  - **Aufzählung aller Pfade** (Mini-Netz): Log-Likelihood, Randverteilungen, erwartete Schrittzahlen und Besuche des E-Schritts.
  - M-Schritt maximiert die erwartete Vollständig-Likelihood (gegen zufällige Störungen).
  - **Die Log-Likelihood steigt in keiner EM-Iteration.**
  - **Konsistenz:** bei Daten aus dem Modell selbst werden die erzeugenden Gewichte gefunden.
  - Angezeigte Kennzahlen gegen unabhängige Neuberechnung.
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jede Trainingsgröße und jeder Start, Experiment-Knöpfe, Abgleich der Kennzahlen mit `analyse(Settings(...))`) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `bwl_algorithm.py` | Kern: Straßengitter, Schrittklassen, E-Schritt, M-Schritt, EM-Lauf |
| `bwl_scenario.py` | Zufallsfahrt und GPS-Messungen (ein Generator, Regler skalieren denselben Zufall) |
| `bwl_evaluation.py` | Lernen, Bewertung auf Testfahrten, Sweeps |
| `bwl_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `bwl_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `bwl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bwl_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Reichere Zustände** (Richtung, Geschwindigkeit): nicht Teil der Demo.
- **Rauschen und allgemeine Übergangsmatrizen mitlernen**, Priors, mehrere Neustarts mit Auswahl nach Likelihood.
- **Echte Spuren.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
