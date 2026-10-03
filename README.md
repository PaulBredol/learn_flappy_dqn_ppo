# Flappy Bird: Vergleich von DQN und PPO

## Dateien im Hauptordner

| Datei | Aufgabe |
|---|---|
| `train_dqn.py` | Training und Einzelauswertung unserer DQN-Variante; enthält Zustandsaufbereitung, Rewards und Trainingseinstellungen. |
| `train_ppo.py` | Training und Einzelauswertung unserer PPO-Variante mit entsprechender Zustandsaufbereitung und Konfiguration. |
| `double_dqn.py` | Double-DQN-Lernregel: Das Online-Netz wählt die nächste Aktion aus, das Zielnetz bewertet sie. |
| `prioritized_double_dqn.py` | PER: Erfahrungen anhand ihrer Lernfehler priorisieren und die dadurch veränderte Stichprobenauswahl beim Lernen gewichten. Baut auf `double_dqn.py` auf. |
| `vector_prioritized_double_dqn.py` | Anpassung des priorisierten Replay-Buffers für mehrere Spielinstanzen. Verwendet die Klassen aus `prioritized_double_dqn.py`. |
| `run_flappy_comparison.py` | Organisiert den gemeinsamen Vergleich: Training, Checkpoints, Validierung, Auswahl des besten Checkpoints und abschließende Tests. Greift auf beide Trainingsdateien und die DQN-Hilfsmodule zu. |
| `requirements_flappy.txt` | Benötigte Python-Bibliotheken. |
| `outputs/` | Ergebnisse von vier Trainingsläufen pro Verfahren; Details stehen in `outputs/README.md`. |

Die drei DQN-Hilfsmodule sind erforderlich und werden importiert. Sie werden nicht einzeln gestartet.

Der vollständige Vergleich ist rechenintensiv. Zum Nachvollziehen der abgegebenen Ergebnisse genügt zunächst das Lesen von `outputs/summary.md`; erneutes Training ist dafür nicht erforderlich. Weitere Optionen zeigt `python run_flappy_comparison.py --help`.

## Versuchsaufbau

Vier Trainingsseeds (100, 200, 300, 400) pro Verfahren ergeben insgesamt acht Trainingsläufe. Jeder Lauf hat ein Sollbudget von 1,5 Millionen Übergängen und ein Checkpoint-Raster von 50.000 Schritten. PPO kann das Sollbudget wegen vollständiger Rollouts leicht überschreiten. Die acht Spielinstanzen innerhalb eines Laufs trainieren ein gemeinsames Modell; sie sind keine acht unabhängigen Trainingsläufe.

Validierung und abschließender Test verwenden getrennte Seeds. Der Vergleich erzeugt keine Videos. Die Ergebnisse beschreiben unsere konkreten Varianten und Einstellungen, keine allgemeine Überlegenheit eines Verfahrens.
