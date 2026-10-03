# Flappy Bird mit Double DQN und PPO

Dieses Projekt trainiert und vergleicht zwei Reinforcement-Learning-Verfahren in `flappy-bird-gymnasium`: Double DQN mit Prioritized Experience Replay (PER) und PPO. Beide erhalten dieselben 15 direkten Zustandswerte und dasselbe Reward-Mapping.

## Dateien

| Datei | Aufgabe |
| --- | --- |
| `train_dqn.py` | Finale DQN-Variante, ehemals V12; Einzeltraining und Evaluation mit Videos |
| `train_ppo.py` | PPO-Variante, ehemals PPO V2; Einzeltraining und Evaluation mit Videos |
| `double_dqn.py` | Double-DQN-Lernziel |
| `prioritized_double_dqn.py` | Prioritätsbaum, Replay-Sampling und gewichtete Lernupdates |
| `vector_prioritized_double_dqn.py` | PER-Unterstützung für mehrere Spielinstanzen |
| `run_flappy_comparison.py` | Gemeinsames Trainings-, Validierungs- und Testprotokoll |
| `check_flappy_comparison.py` | Protokoll- und Evaluationsprüfungen ohne Training |
| `check_flappy_v12.py` | DQN-Logikprüfung; optional echter Trainings- und Videotest |
| `check_helpers.py` | Enthält den von beiden Prüfskripten benötigten AST-Testhelfer `extracted` |
| `requirements.txt` | Installations-Einstieg; verweist auf `requirements_flappy.txt` |
| `requirements_versions.txt` | Exakte Versionen der direkten Pakete in der lokalen Umgebung zum Abgabezeitpunkt |

Alle Python-Dateien in diesem Ordner zusammen lassen. Die Hilfsmodule werden importiert und müssen nicht einzeln gestartet werden.

Die Versionskennungen V12 und V2 in Ausgaben bleiben zur Zuordnung der Entwicklung erhalten. Alte Modelle und Quellcode-Snapshots tragen weiterhin die ursprünglichen Dateinamen.

## Installation

Entwicklungsumgebung: Windows, Python 3.13 (64 Bit). Eine vorhandene Python-Installation vorausgesetzt, im entpackten Abgabeordner ausführen:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Die Aktivierung der Umgebung ist nicht nötig, wenn der vollständige Interpreterpfad wie oben verwendet wird. Die folgenden Befehle verwenden ebenfalls diesen Pfad. Unter Linux/macOS lautet der entsprechende Pfad `.venv/bin/python`; diese Plattformen wurden für die Abgabe nicht erneut getestet.

`requirements_flappy.txt` enthält die benötigten direkten Bibliotheken einschließlich Video-Unterstützung. Für einen Versuch mit den direkt beobachteten Paketversionen kann statt `requirements.txt` die Datei `requirements_versions.txt` installiert werden. Diese ist kein vollständiger Lock aller indirekten Abhängigkeiten. Die Verfügbarkeit der Pakete auf einem anderen Rechner sowie eine frische Installation wurden bei der Abgabevorbereitung nicht verifiziert.

W&B ist als Paket erforderlich, aber ein Konto ist für die Befehle mit `--wandb-mode disabled` nicht nötig. CPU-Ausführung ist möglich; eine GPU ist nicht erforderlich. Videodateien entstehen mit MoviePy und ImageIO-FFmpeg.

## Schnellprüfung ohne langes Training

```powershell
.\.venv\Scripts\python.exe -c "import torch, gymnasium, flappy_bird_gymnasium, stable_baselines3, wandb, moviepy; import train_dqn, train_ppo; print('Imports OK')"
.\.venv\Scripts\python.exe check_flappy_comparison.py
.\.venv\Scripts\python.exe check_flappy_v12.py --logic-only
.\.venv\Scripts\python.exe run_flappy_comparison.py --plan-only --wandb-mode disabled
```

Die ersten beiden Prüfskripte starten kein langes Training. `--plan-only` zeigt das Protokoll, ohne ein Experiment anzulegen. Anschließend prüft ein kurzer Smoke-Test echtes Training beider Verfahren sowie Speichern, Laden und Evaluation:

```powershell
.\.venv\Scripts\python.exe run_flappy_comparison.py --smoke-test --wandb-mode disabled
```

Der Smoke-Test verwendet je einen Trainingsseed, 512 angeforderte Trainingsschritte, Checkpoints alle 256 Schritte, kleine Batches und maximal 64 Schritte je Evaluationsepisode. Er prüft die Funktion, nicht die Leistungsfähigkeit der Agenten. Ein vollständiger DQN-Funktionstest einschließlich Videoexport ist zusätzlich mit `check_flappy_v12.py` ohne `--logic-only` möglich.

## Gemeinsamen Vergleich starten

```powershell
.\.venv\Scripts\python.exe run_flappy_comparison.py --wandb-mode disabled
```

Der Standard entspricht vier Trainingsruns pro Verfahren, also acht Trainingsjobs insgesamt:

- Trainingsseeds: 100, 200, 300, 400. Jeder Run verwendet acht Spielinstanzen mit den daraus abgeleiteten Seeds.
- Trainingsbudget: 1.500.000 Übergänge insgesamt pro Run, nicht pro Spielinstanz. PPO kann wegen vollständiger Rollouts das Sollbudget leicht überschreiten.
- Gemeinsames Checkpoint-Raster: 50.000 Schritte.
- Validation: 20 Episoden mit Seeds 2000 bis 2019.
- Test: 100 Episoden mit Seeds 3000 bis 3099, getrennt von Training und Validation.
- Evaluation: deterministische Aktionswahl, maximal 100.000 Schritte pro Episode.
- Keine Videos im Vergleichsskript. Die Trainingsjobs laufen nacheinander; acht Spielinstanzen teilen sich jeweils ein Modell.

Zuerst erfolgen Training und Checkpoint-Validierung. Anschließend wird der beste periodische Checkpoint ausschließlich nach dem mittleren Validierungsscore ausgewählt und die Auswahl gesperrt. Auf den Testseeds werden sowohl dieses Modell (`best`) als auch das letzte Modell (`final`) ausgewertet. Die Testdaten wählen keinen Checkpoint aus.

Lange erfolgreiche Evaluationsepisoden können die Gesamtlaufzeit deutlich erhöhen. 100.000 Schritte sind ein Episodenlimit, keine Zeitangabe. Limitabbrüche (`truncated`) müssen von Kollisionen unterschieden werden.

## Einzelne Verfahren starten

```powershell
.\.venv\Scripts\python.exe train_dqn.py --mode train --device cpu --wandb-mode disabled
.\.venv\Scripts\python.exe train_ppo.py --mode train --device cpu --wandb-mode disabled
```

`--mode train` führt nur Training aus. Für Training mit anschließender Evaluation und Videos `--mode both` verwenden. Das ist auch der Standard ohne `--mode`. Die Einzeldateien evaluieren standardmäßig 20 Episoden je Modell und erzeugen Videos im Raster von 150.000 Schritten (DQN) beziehungsweise 250.000 Schritten (PPO). Diese individuellen Auswertungen ersetzen nicht das gemeinsame Vergleichsprotokoll. Weitere Optionen zeigt `--help`.

W&B kann optional mit `--wandb-mode online` verwendet werden; dafür vorher `wandb login` ausführen. Keine Zugangsdaten sind Bestandteil der Abgabe.

## Gemeinsame Repräsentation und Unterschiede

Die 15 Werte beschreiben Vogelmittelpunkt, vertikale Geschwindigkeit, Rotation, Decken- und Bodenabstand sowie je fünf Abstände zu den nächsten beiden noch nicht vollständig passierten Röhrenpaaren. Es handelt sich um direkt gelesene Simulatordaten, nicht um Bilderkennung.

Reward-Mapping: passierte Röhre +10, regulärer Überlebensschritt +0,01, Kollision −10, ursprüngliche Näherungsstrafe 0. Beide Trainingsverfahren normalisieren die Rewards. Evaluationen verwenden die unnormalisierten Rewards und den Röhrenscore.

Gemeinsam: acht Umgebungen, Batchgröße 256, gamma 0,995, zwei versteckte Schichten mit je 128 Neuronen (PPO jeweils für Actor und Critic). DQN verwendet Double DQN, PER, einen Buffer mit 200.000 Übergängen und epsilon-greedy. PPO verwendet 512 Schritte je Umgebung, zehn Epochen pro Rollout und sinkendes Clipping. Beide Lernraten sinken, mit unterschiedlichen Anfangswerten. Die genaue Konfiguration wird pro Run gespeichert.

## Ausgaben und Fortsetzen

Jeder neue Lauf erhält einen eigenen Unterordner in `outputs`. Das Vergleichsskript speichert unter anderem `manifest.json`, einen Quellcode-Snapshot, pro Job Checkpoints, Validierungsdateien, `selection.json`, `test_best.json`, `test_final.json` sowie abschließend `summary.json` und `summary.md`.

```powershell
.\.venv\Scripts\python.exe run_flappy_comparison.py --resume "outputs\COMPARISON_DQN_PPO_<Kennung>"
```

Den tatsächlich ausgegebenen Ordner verwenden. Abgeschlossene Phasen werden übersprungen; bereits gespeicherte Evaluationsepisoden werden wiederverwendet. Unterbrochenes Training wird nicht als exakte Fortsetzung mit Replay-Buffer garantiert. Der Quellcode muss zum Manifest passen. Die spätere Dateiumbenennung und die Anpassung des Standards auf vier Runs verändern diese Hashes. Alte Experimente deshalb nicht ungeprüft mit dem neuen Code fortsetzen und keine historischen Hashes überschreiben. Für bestehende Ergebnisse den jeweiligen Original-Snapshot und das Manifest aufbewahren.

## Abgabe und Nachvollziehbarkeit

Dieses Paket enthält den aktuellen ausführbaren Code und Prüfskripte, keine Modelle, Videos, virtuellen Umgebungen oder Trainingslogs. Für den Nachweis der präsentierten Resultate zusätzlich den zugehörigen Experimentordner beziehungsweise mindestens Manifest, Configs, Auswahl- und Evaluationsdateien, Zusammenfassung und ursprünglichen Quellcode-Snapshot bereitstellen. Modelle bei Bedarf separat beilegen. Neue Runs müssen wegen stochastischen Lernens und möglicher Hardwareunterschiede nicht dieselben Resultate erzeugen.

Prüfstand 03.10.2026: Syntax aller beigefügten Python-Dateien sowie Protokoll- und DQN-Logiktests geprüft. Der echte Torch-/Trainings-Smoke-Test und eine frische Paketinstallation wurden bei dieser Abgabevorbereitung nicht erfolgreich ausgeführt, da die lokale Projekt-Python-Umgebung aus dem Assistenzprozess nicht gestartet werden konnte. Die oben genannten Prüfkommandos sind vor der endgültigen Einreichung auf dem Zielrechner auszuführen.

## Quellen

- Double DQN: van Hasselt, Guez und Silver (2016), https://arxiv.org/abs/1509.06461
- PER: Schaul et al. (2016), https://arxiv.org/abs/1511.05952
- PPO: Schulman et al. (2017), https://arxiv.org/abs/1707.06347
- Spielumgebung: https://github.com/markub3327/flappy-bird-gymnasium
