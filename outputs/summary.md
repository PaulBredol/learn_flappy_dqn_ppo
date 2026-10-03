# DQN / PPO: Vergleich

Best: ausschließlich durch Validierung ausgewählt. Test: getrennte Seeds.

| Verfahren | Trainingsseed | Test Ø best | Test Median best | Unter 10 | Test Ø final | Training min |
|---|---:|---:|---:|---:|---:|---:|
| dqn | 100 | 19.48 | 13.50 | 34.0% | 23.05 | 34.2 |
| ppo | 100 | 357.78 | 226.00 | 2.0% | 281.40 | 24.1 |
| ppo | 200 | 119.40 | 88.00 | 6.0% | 97.83 | 29.8 |
| dqn | 200 | 174.79 | 109.50 | 7.0% | 55.00 | 38.2 |
| dqn | 300 | 2180.48 | 2654.00 | 0.0% | 2180.58 | 42.6 |
| ppo | 300 | 1014.53 | 834.00 | 3.0% | 139.85 | 22.8 |
| ppo | 400 | 103.25 | 85.00 | 2.0% | 105.91 | 24.8 |
| dqn | 400 | 717.78 | 500.50 | 6.0% | 59.17 | 31.7 |

Aggregierte Mittelwerte und Streuung über Trainingsseeds stehen in summary.json.
Trainingszeit enthält Checkpoint-/Logging-Aufwand, aber keine Auswertung oder Videos.
PPO kann wegen voller Rollouts das Sollbudget leicht überschreiten.
Getrennte Seeds garantieren keine vollständig verschiedenen Röhrenfolgen.
