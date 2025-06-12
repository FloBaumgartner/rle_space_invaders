from __future__ import annotations
import argparse, glob
from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read_scalar(event_file: str, tag: str):
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        raise KeyError
    evs = ea.Scalars(tag)
    return [e.step for e in evs], [e.value for e in evs]

def locate_event_files(run_dir: Path):
    files = glob.glob(str(run_dir / "events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No event files under {run_dir}")
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", default="results.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    event_files = locate_event_files(args.run_dir)

    # Debug: list tags
    print("=== TAG DUMP ===")
    for evf in event_files:
        ea = EventAccumulator(evf, size_guidance={"scalars": 0})
        ea.Reload()
        print(f"\nFile: {evf}")
        for tag in ea.Tags()["scalars"]:
            print("  ", tag)
    print("================\n")

    # Choose your actual tag names after inspecting the dump:
    train_tag = "rollout/ep_rew_mean"
    eval_tag  = "eval/episodic_return"

    # Read training
    for evf in event_files:
        try:
            steps_train, values_train = read_scalar(evf, train_tag)
            print("Training data →", evf)
            break
        except KeyError:
            continue
    else:
        raise RuntimeError(f"Training tag not found: {train_tag}")

    # Read evaluation
    for evf in event_files:
        try:
            steps_eval, values_eval = read_scalar(evf, eval_tag)
            print("Eval data →", evf)
            break
        except KeyError:
            continue
    else:
        raise RuntimeError(f"Eval tag not found: {eval_tag}")

    # Plot
    plt.figure()
    plt.plot(steps_train, values_train, label="Training return")
    plt.plot(steps_eval, values_eval, label="Eval return")
    plt.xlabel("Environment steps")
    plt.ylabel("Episodic return")
    plt.title("PPO CleanRL: Training vs Evaluation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")
    if args.show: plt.show()

if __name__ == "__main__":
    main()
