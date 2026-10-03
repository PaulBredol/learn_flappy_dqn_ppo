# Ergebnisse des DQN/PPO-Vergleichs

Dieser Ordner enthält vier Trainingsläufe pro Verfahren, insgesamt acht Runs. Die Zahl im Dateinamen bezeichnet den Trainingsseed, der die Zufallsinitialisierung steuert.

| Dateien | Bedeutung |
|---|---|
| `dqn_seed_100.zip`, `dqn_seed_200.zip`, `dqn_seed_300.zip`, `dqn_seed_400.zip` | Archivierte Daten der vier DQN-Läufe mit unterschiedlichen Trainingsseeds. |
| `ppo_seed_100.zip`, `ppo_seed_200.zip`, `ppo_seed_300.zip`, `ppo_seed_400.zip` | Archivierte Daten der vier PPO-Läufe mit denselben vier Trainingsseeds. |
| `manifest.json` | Gemeinsamer Versuchsplan mit Trainingsbudget, Seeds, Checkpoint-Abständen und Auswertungseinstellungen. |
| `summary.md` | Lesbare Ergebnistabelle aller acht Runs. Empfohlener Einstieg für den Vergleich. |
| `summary.json` | Maschinenlesbare Gesamtauswertung mit Kennzahlen und Zusammenfassungen über die Trainingsseeds. |

In der Zusammenfassung bezeichnet **Best** den anhand der Validierung ausgewählten Checkpoint und **Final** das Modell am Trainingsende. Beide werden auf getrennten Testseeds ausgewertet. Der **Score** zählt die passierten Röhren. Für den Vergleich sollten alle vier Runs eines Verfahrens berücksichtigt werden.
