#!/usr/bin/env python3
"""
plot_results.py

Lädt TensorBoard Event-Dateien und erstellt Plots für die wichtigsten RL-Metriken:
- Training: Episodische Returns, Verluste, Lernrate, Schrittgeschwindigkeit, etc.
- Evaluation: Episodische Returns, Länge, Zeit

Verwendung:
    python eval_initial_plots.py --logdir runs/SpaceInvaders-v5__ppo_clean_rl__1__20250612_123456
    python eval_initial_plots.py --logdir runs/... --save_dir figures/

Abhängigkeiten:
    pip install tensorboard matplotlib
"""
import os
import argparse
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def load_scalars(log_dir):
    # Lade alle Scalar-Tags aus dem Event-Ordner
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            # 0 = keine Begrenzung, lade alle Werte
            event_accumulator.SCALARS: 0,
        }
    )
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    data = {tag: ea.Scalars(tag) for tag in tags}
    return data


def plot_metric(xs, ys, title, xlabel, ylabel, show=True, save_path=None):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot RL metrics from TensorBoard logs.")
    parser.add_argument('--logdir', type=str, required=True,
                        help='Pfad zum TensorBoard-Log-Verzeichnis (runs/ALE/...)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Wenn angegeben, werden die Plots hier als PNGs gespeichert')
    parser.add_argument('--no_show', action='store_true', help='Unterdrücke Anzeige der Plots (nützlich beim Speichern)')
    args = parser.parse_args()

    data = load_scalars(args.logdir)

    # Definiere Metriken
    train_metrics = [
        ('train/episodic_return', 'Global Step', 'Episodic Return'),
        ('charts/learning_rate', 'Global Step', 'Learning Rate'),
        ('losses/value_loss', 'Global Step', 'Value Loss'),
        ('losses/policy_loss', 'Global Step', 'Policy Loss'),
        ('losses/entropy', 'Global Step', 'Entropy'),
        ('losses/old_approx_kl', 'Global Step', 'Old Approx KL'),
        ('losses/approx_kl', 'Global Step', 'Approx KL'),
        ('losses/clipfrac', 'Global Step', 'Clip Fraction'),
        ('losses/explained_variance', 'Global Step', 'Explained Variance'),
        ('charts/StepPerSecond', 'Global Step', 'Steps per Second'),
    ]
    eval_metrics = [
        ('eval/episodic_return', 'Episode', 'Eval Return'),
        ('eval/episodic_length', 'Episode', 'Eval Length'),
        ('eval/episodic_time', 'Episode', 'Eval Time'),
    ]

    # Plot Training-Metriken
    for tag, xlabel, ylabel in train_metrics:
        if tag in data:
            events = data[tag]
            xs = [e.step for e in events]
            ys = [e.value for e in events]
            save_path = None
            if args.save_dir:
                save_path = os.path.join(args.save_dir, tag.replace('/', '_') + '.png')
            plot_metric(xs, ys, tag, xlabel, ylabel, show=not args.no_show, save_path=save_path)

    # Plot Evaluation-Metriken
    for tag, xlabel, ylabel in eval_metrics:
        if tag in data:
            events = data[tag]
            xs = list(range(1, len(events) + 1))
            ys = [e.value for e in events]
            save_path = None
            if args.save_dir:
                save_path = os.path.join(args.save_dir, tag.replace('/', '_') + '.png')
            plot_metric(xs, ys, tag, xlabel, ylabel, show=not args.no_show, save_path=save_path)

    print("Fertig mit Plot-Erstellung!")


if __name__ == '__main__':
    main()