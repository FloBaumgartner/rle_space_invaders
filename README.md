# PPO-Experiment Suite

## Übersicht

Dieses Projekt enthält verschiedene Implementierungen und Experimente zu Proximal Policy Optimization (PPO) für Reinforcement Learning. Ziel ist der Vergleich mehrerer Ansätze:

* **Baseline**: Einfacher PPO-Agent
* **PPO + RND**: PPO mit Random Network Distillation als intrinsische Belohnung
* **Recurrent PPO**: PPO mit rekurrentem Netzwerk (LSTM)
* **PPO-Impala**: Anlehnung an den IMPALA-Algorithmus
* **Hyperparameter-Tuning**: Automatisiertes Tuning mit Optuna

## Projektstruktur

```text
baseline.py              # Erste Baseline-Implementierung
initial.sh               # Skript: Ausführung und Evaluation der Baseline

ppo_clean_rl.py          # Saubere PPO-Implementierung (5M Schritte)
ppo_rnd.py               # PPO mit RND (5M Schritte)
ppo_recurrent.py         # Recurrent PPO (5M Schritte)
ppo_impala.py            # PPO-Impala-Variante (5M Schritte)

rnd.sh                   # Skript: Starte PPO+RND-Experiment
recurrent.sh             # Skript: Starte Recurrent-PPO-Experiment
impala.sh                # Skript: Starte PPO-Impala-Experiment

hp-optuna.sh             # Skript: Hyperparameter-Tuning mit Optuna
ppo_hp-optuna.py         # PPO-Variante im Tuning-Loop

ppo_eval.py              # Sammle Ergebnisse aus allen Experimenten

eval_metrics.ipynb       # Notebook: Metriken-Berechnung (Auswahl)
eval_metrics_all.ipynb   # Notebook: Erweiterte Metriken für alle Durchläufe
bericht.md               # Zusammenfassung und Analyse der Ergebnisse
```

## Voraussetzungen

* Python 3.7+
* Abhängigkeiten (installierbar via `requirements.txt`):

  * `torch`
  * `gym`
  * `optuna`
  * `numpy`, `pandas`, `matplotlib`
  * weitere Bibliotheken siehe `requirements.txt`

## Installation

```bash
git clone <repository-url>
cd <repository>
pip install -r requirements.txt
```

## Nutzung / Experimente

### Baseline ausführen

```bash
bash initial.sh
```

### PPO-Varianten (jeweils \~5M Schritte)

```bash
bash rnd.sh         # PPO + RND
bash recurrent.sh   # Recurrent PPO
bash impala.sh      # PPO-Impala
```

### Hyperparameter-Tuning mit Optuna

```bash
bash hp-optuna.sh
```


## Ergebnisse & Bericht

Der finale Bericht (`bericht.md`) fasst die Experimente, Metriken und Erkenntnisse zusammen.

---

*Erstellt für das PPO-Experiment-Projekt*
