"""
evaluate.py — Evaluation script: sweep n = 1 … 50, measure success rates, plot.

Usage
-----
# Evaluate pre-trained models (run train.py first, or pass --auto_train)
python evaluate.py

# Auto-train any missing models, then evaluate
python evaluate.py --auto_train

# Only evaluate n = 1 … 15, 1000 episodes each
python evaluate.py --n_max 15 --eval_episodes 1000

# Only a single n (same idea as train.py --n), e.g. n=12 only
python evaluate.py --n 12

# Auto-train + evaluate only n=12 if checkpoint missing
python evaluate.py --n 12 --auto_train

# Plot training curves for every n that has a saved history
python evaluate.py --plot_training

Evaluation metric
-----------------
Success Rate = (number of episodes where generated == target) / total episodes

This is evaluated with a fully greedy policy (epsilon = 0).
We use 500 episodes per n by default to get a stable estimate.
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from env import BitSequenceEnv
from network import QNetwork
from train import HPARAMS, train_single, get_episode_count

HIDDEN = HPARAMS["hidden_sizes"]
DEFAULT_EVAL_EPISODES = 500


# ---------------------------------------------------------------------------
def _write_results_incremental(results: dict[int, dict], results_file: str) -> None:
    """
    Persist partial results safely during a long sweep.

    Writes to a temporary file first, then atomically replaces results_file.
    """
    tmp_path = results_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, results_file)


# ---------------------------------------------------------------------------
def evaluate_one(
    n: int,
    save_dir: str = "models",
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    compact_obs: bool = False,
) -> dict:
    """
    Evaluate the saved DQN model for sequence length n.

    The policy is fully greedy (no random exploration).

    Returns:
        dict with keys: n, success_rate, examples (list of up to 5 episodes).
    """
    ckpt_path = os.path.join(save_dir, f"dqn_n{n:02d}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    env = BitSequenceEnv(n=n, reward_shaping=False, compact_obs=compact_obs)
    obs_dim = env.observation_space.shape[0]

    q_net = QNetwork(obs_dim, n_actions=2, hidden_sizes=HIDDEN)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    q_net.load_state_dict(ckpt["q_net"])
    q_net.eval()

    successes = 0
    examples: list[dict] = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                t = torch.FloatTensor(state).unsqueeze(0)
                action = int(q_net(t).argmax(dim=1).item())
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if info["success"]:
            successes += 1

        # Keep a handful of qualitative examples
        if ep < 5:
            examples.append(
                {
                    "target": env.target.astype(int).tolist(),
                    "generated": env.generated.astype(int).tolist(),
                    "success": bool(info["success"]),
                }
            )

    return {
        "n": n,
        "success_rate": successes / num_episodes,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
def plot_success_rates(results: dict, plot_file: str = "success_rate.png") -> None:
    """Bar + line chart of success rate vs sequence length."""
    ns = sorted(n for n in results if results[n]["success_rate"] is not None)
    rates = [results[n]["success_rate"] for n in ns]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Bar chart
    colors = ["#2196F3" if r >= 0.5 else "#FF5722" for r in rates]
    ax.bar(ns, rates, color=colors, alpha=0.6, width=0.8, label="Success rate")
    # Line overlay
    ax.plot(ns, rates, "k-o", markersize=4, linewidth=1.5, label="Trend")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="50% baseline")
    ax.set_xlabel("Sequence length  $n$", fontsize=13)
    ax.set_ylabel("Success rate", fontsize=13)
    ax.set_title("DQN Greedy Success Rate  vs  Sequence Length  (n = 1 … 50)",
                 fontsize=14)
    ax.set_xlim(0, max(ns) + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(0, max(ns) + 1, 5))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotate rate values above bars for n ≤ 20
    for n, r in zip(ns, rates):
        if n <= 20:
            ax.text(n, r + 0.02, f"{r:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Success-rate plot saved: {plot_file}")
    plt.show()


# ---------------------------------------------------------------------------
def plot_training_curves(save_dir: str = "models",
                         plot_file: str = "training_curves.png") -> None:
    """
    Load all saved history_n??.json files and plot smoothed success curves
    for a selection of n values on a single figure.
    """
    selected_ns = [1, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    available = []
    histories = {}
    for n in selected_ns:
        p = os.path.join(save_dir, f"history_n{n:02d}.json")
        if os.path.exists(p):
            with open(p) as f:
                histories[n] = json.load(f)
            available.append(n)

    if not available:
        print("No training history files found. Run train.py first.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(available) - 1)) for i in range(len(available))]

    def smooth(x, w=50):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    for idx, n in enumerate(available):
        ax = axes[idx]
        s = np.array(histories[n]["successes"])
        if len(s) > 50:
            ax.plot(smooth(s), color=colors[idx], linewidth=1.5)
        else:
            ax.plot(s, color=colors[idx], linewidth=1.5)
        ax.set_title(f"n = {n}", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Success", fontsize=9)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DQN Training Success Rate (smoothed, window=50)", fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Training curves plot saved: {plot_file}")
    plt.show()


# ---------------------------------------------------------------------------
def run_sweep(
    n_range=range(1, 51),
    save_dir: str = "models",
    num_eval_episodes: int = DEFAULT_EVAL_EPISODES,
    auto_train: bool = False,
    train_episodes: int | None = None,
    train_reward_shaping: bool = True,
    train_device: str = "cpu",
    train_compact_obs: bool = False,
    eval_compact_obs: bool = False,
    results_file: str = "eval_results.json",
    plot_file: str = "success_rate.png",
) -> dict:
    """
    Evaluate all n in n_range, optionally auto-training missing models.

    For each n:
      - If checkpoint exists → load and evaluate greedily.
      - If checkpoint missing and auto_train=True → train first, then evaluate.
      - If checkpoint missing and auto_train=False → skip and report None.

    Saves results to JSON and generates a success-rate plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    results: dict[int, dict] = {}

    print(f"\nEvaluating DQN over n = {min(n_range)} to {max(n_range)}")
    print(f"Evaluation episodes per n: {num_eval_episodes}")
    print("=" * 60)

    for n in n_range:
        ckpt_path = os.path.join(save_dir, f"dqn_n{n:02d}.pt")

        if not os.path.exists(ckpt_path):
            if auto_train:
                budget = train_episodes if train_episodes is not None else get_episode_count(n)
                print(f"\n[n={n:2d}] No checkpoint — training ({budget} eps)…")
                train_single(
                    n=n,
                    num_episodes=train_episodes,
                    save_dir=save_dir,
                    reward_shaping=train_reward_shaping,
                    verbose=True,
                    device=train_device,
                    compact_obs=train_compact_obs,
                )
            else:
                print(f"[n={n:2d}] SKIP  (no checkpoint; use --auto_train to train)")
                results[n] = {"n": n, "success_rate": None, "examples": []}
                _write_results_incremental(results, results_file)
                continue

        print(f"[n={n:2d}] Evaluating ({num_eval_episodes} eps)...", end="  ", flush=True)
        res = evaluate_one(
            n,
            save_dir=save_dir,
            num_episodes=num_eval_episodes,
            compact_obs=eval_compact_obs,
        )
        results[n] = res
        sr = res["success_rate"]
        print(f"success rate = {sr:.3f}")

        # Show up to 2 qualitative examples
        for ex in res["examples"][:2]:
            mark = "OK" if ex["success"] else "XX"
            print(f"     [{mark}]  target={ex['target']}   generated={ex['generated']}")

        # Persist results after each n so progress isn't lost
        _write_results_incremental(results, results_file)

    print(f"\nResults saved to: {results_file}")

    # Plot
    if any(r["success_rate"] is not None for r in results.values()):
        plot_success_rates(results, plot_file=plot_file)

    return results


# ---------------------------------------------------------------------------
def print_summary(results: dict) -> None:
    """Print a compact ASCII table of success rates."""
    sep = "-" * 46
    print(f"\n{'n':>4} | {'Success Rate':>14} | {'Status'}")
    print(sep)
    for n in sorted(results):
        sr = results[n]["success_rate"]
        if sr is None:
            print(f"{n:>4} | {'N/A':>14} | no checkpoint")
        else:
            bar = "#" * int(sr * 20)
            print(f"{n:>4} | {sr:>14.3f} | {bar}")
    print(sep)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN models for bit-sequence matching."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Evaluate only this sequence length (1–50), like `train.py --n`. "
            "If set, ignores --n_max for the sweep range."
        ),
    )
    parser.add_argument("--n_max", type=int, default=50,
                        help="Evaluate n = 1 … n_max when --n is not set")
    parser.add_argument("--eval_episodes", type=int, default=DEFAULT_EVAL_EPISODES,
                        help="Greedy rollouts per n for success-rate estimate")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--auto_train", action="store_true",
                        help="Train missing models automatically before evaluation")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "When used with --auto_train: override training episode count "
            "(same meaning as train.py --episodes). If omitted, uses the "
            "auto-scaled per-n budget from train.py."
        ),
    )
    parser.add_argument(
        "--no_shaping",
        action="store_true",
        help=(
            "When used with --auto_train: train with sparse 0/1 reward "
            "(same meaning as train.py --no_shaping)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "When used with --auto_train: Torch device for training "
            "(same meaning as train.py --device)."
        ),
    )
    parser.add_argument(
        "--compact_obs",
        action="store_true",
        help=(
            "Match train.py --compact_obs: obs is [target[t], step/n] (dim 2). "
            "Use when evaluating checkpoints trained with compact observations."
        ),
    )
    parser.add_argument("--results_file", type=str, default="eval_results.json")
    parser.add_argument("--plot_file", type=str, default="success_rate.png")
    parser.add_argument("--plot_training", action="store_true",
                        help="Also plot per-n training curves (requires history JSON files)")
    args = parser.parse_args()

    if args.n is not None:
        if not (1 <= args.n <= 50):
            parser.error("--n must be between 1 and 50")
        sweep_range = range(args.n, args.n + 1)
    else:
        sweep_range = range(1, args.n_max + 1)

    results = run_sweep(
        n_range=sweep_range,
        save_dir=args.save_dir,
        num_eval_episodes=args.eval_episodes,
        auto_train=args.auto_train,
        train_episodes=args.episodes,
        train_reward_shaping=not args.no_shaping,
        train_device=args.device,
        train_compact_obs=args.compact_obs,
        eval_compact_obs=args.compact_obs,
        results_file=args.results_file,
        plot_file=args.plot_file,
    )

    print_summary(results)

    if args.plot_training:
        plot_training_curves(
            save_dir=args.save_dir,
            plot_file="training_curves.png",
        )


if __name__ == "__main__":
    main()
