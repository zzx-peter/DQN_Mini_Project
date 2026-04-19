"""
evaluate.py — Evaluate the saved DQN over a configurable range of sequence lengths.

The greedy policy uses observations [target[t], step/n] (dim 2) for every n.

Usage
-----
python evaluate.py                              # default: n_min=1, n_max=50, step=1
python evaluate.py --n_min 5 --n_max 40 --n_step 5

python evaluate.py --n 12                       # single n only (ignores range)

python evaluate.py --auto_train
python evaluate.py --eval_episodes 1000 --plot_training
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from env import BitSequenceEnv
from network import QNetwork
from train import HPARAMS, train_dqn, get_episode_count, CKPT_FILENAME, HISTORY_FILENAME

HIDDEN = HPARAMS["hidden_sizes"]
OBS_DIM = 2
DEFAULT_EVAL_EPISODES = 500


# ---------------------------------------------------------------------------
def _write_results_incremental(results: dict[int, dict], results_file: str) -> None:
    tmp_path = results_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, results_file)


# ---------------------------------------------------------------------------
def evaluate_one(
    n: int,
    save_dir: str = "models",
    num_episodes: int = DEFAULT_EVAL_EPISODES,
) -> dict:
    """
    Greedy evaluation for sequence length n using the unified checkpoint.
    """
    ckpt_path = os.path.join(save_dir, CKPT_FILENAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Unified checkpoint not found: {ckpt_path} (run train.py first)"
        )

    env = BitSequenceEnv(n=n, reward_shaping=False)
    q_net = QNetwork(OBS_DIM, n_actions=2, hidden_sizes=HIDDEN)
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
    ns = sorted(n for n in results if results[n]["success_rate"] is not None)
    rates = [results[n]["success_rate"] for n in ns]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#2196F3" if r >= 0.5 else "#FF5722" for r in rates]
    ax.bar(ns, rates, color=colors, alpha=0.6, width=0.8, label="Success rate")
    ax.plot(ns, rates, "k-o", markersize=4, linewidth=1.5, label="Trend")

    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="50% baseline",
    )
    ax.set_xlabel("Sequence length  $n$", fontsize=13)
    ax.set_ylabel("Success rate", fontsize=13)
    ax.set_title(
        "Unified DQN — Greedy success rate vs sequence length",
        fontsize=14,
    )
    ax.set_xlim(0, max(ns) + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(0, max(ns) + 1, 5))
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for n, r in zip(ns, rates):
        if n <= 20:
            ax.text(n, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Success-rate plot saved: {plot_file}")
    plt.show()


# ---------------------------------------------------------------------------
def plot_training_curves(save_dir: str = "models", plot_file: str = "training_curves.png") -> None:
    hist_path = os.path.join(save_dir, HISTORY_FILENAME)
    if not os.path.exists(hist_path):
        print(f"No training history at {hist_path}. Run train.py first.")
        return

    with open(hist_path) as f:
        hist = json.load(f)

    s = np.array(hist["successes"], dtype=np.float64)
    w = min(200, max(10, len(s) // 20))
    if len(s) >= w:
        smoothed = np.convolve(s, np.ones(w) / w, mode="valid")
    else:
        smoothed = s

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smoothed, color="#1976D2", linewidth=1.2)
    ax.set_title(
        f"DQN training (smoothed success, window={w})  n = {hist.get('n', '?')}",
        fontsize=13,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success (smoothed)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Training curve saved: {plot_file}")
    plt.show()


# ---------------------------------------------------------------------------
def run_sweep(
    n_range=range(1, 51),
    save_dir: str = "models",
    num_eval_episodes: int = DEFAULT_EVAL_EPISODES,
    auto_train: bool = False,
    train_episodes: int | None = None,
    train_n: int = 8,
    train_reward_shaping: bool = True,
    train_device: str = "cpu",
    results_file: str = "eval_results.json",
    plot_file: str = "success_rate.png",
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    results: dict[int, dict] = {}

    ckpt_path = os.path.join(save_dir, CKPT_FILENAME)
    if not os.path.exists(ckpt_path):
        if auto_train:
            budget = train_episodes if train_episodes is not None else get_episode_count(train_n)
            print(f"\nNo {CKPT_FILENAME} — training at n={train_n} ({budget} episodes)…")
            train_dqn(
                n=train_n,
                num_episodes=train_episodes,
                save_dir=save_dir,
                reward_shaping=train_reward_shaping,
                verbose=True,
                device=train_device,
            )
        else:
            print(
                f"SKIP: no checkpoint at {ckpt_path}. "
                "Run train.py or use --auto_train."
            )
            for n in n_range:
                results[n] = {"n": n, "success_rate": None, "examples": []}
            _write_results_incremental(results, results_file)
            return results

    print(f"\nEvaluating DQN from {ckpt_path}")
    lo, hi = min(n_range), max(n_range)
    if isinstance(n_range, range) and n_range.step != 1:
        print(
            f"n from {lo} to {hi} (step {n_range.step}), "
            f"{num_eval_episodes} episodes each"
        )
    else:
        print(f"n = {lo} … {hi}   ({num_eval_episodes} episodes each)")
    print("=" * 60)

    for n in n_range:
        print(f"[n={n:2d}] Evaluating …  ", end="", flush=True)
        res = evaluate_one(n, save_dir=save_dir, num_episodes=num_eval_episodes)
        results[n] = res
        sr = res["success_rate"]
        print(f"success rate = {sr:.3f}")

        for ex in res["examples"][:2]:
            mark = "OK" if ex["success"] else "XX"
            print(f"     [{mark}]  target={ex['target']}   generated={ex['generated']}")

        _write_results_incremental(results, results_file)

    print(f"\nResults saved to: {results_file}")

    if any(r["success_rate"] is not None for r in results.values()):
        plot_success_rates(results, plot_file=plot_file)

    return results


# ---------------------------------------------------------------------------
def print_summary(results: dict) -> None:
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
        description="Evaluate unified DQN (single dqn.pt) over sequence lengths."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only this n (1–50). Overrides --n_min/--n_max/--n_step.",
    )
    parser.add_argument(
        "--n_min",
        type=int,
        default=1,
        help="Sweep: smallest sequence length (default 1)",
    )
    parser.add_argument(
        "--n_max",
        type=int,
        default=50,
        help="Sweep: largest sequence length inclusive (default 50)",
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="Sweep: stride between successive n (default 1)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Greedy rollouts per n",
    )
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument(
        "--auto_train",
        action="store_true",
        help=f"Train if {CKPT_FILENAME} is missing",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="With --auto_train: override training episode count",
    )
    parser.add_argument(
        "--train_n",
        type=int,
        default=8,
        help="With --auto_train: sequence length n used for training",
    )
    parser.add_argument(
        "--no_shaping",
        action="store_true",
        help="With --auto_train: use sparse step reward",
    )
    parser.add_argument("--device", type=str, default="cpu", help="With --auto_train: torch device")
    parser.add_argument("--results_file", type=str, default="eval_results.json")
    parser.add_argument("--plot_file", type=str, default="success_rate.png")
    parser.add_argument(
        "--plot_training",
        action="store_true",
        help=f"Plot {HISTORY_FILENAME} (smoothed success curve)",
    )
    args = parser.parse_args()

    if not (1 <= args.train_n <= 50):
        parser.error("--train_n must be between 1 and 50")

    if args.n is not None:
        if not (1 <= args.n <= 50):
            parser.error("--n must be between 1 and 50")
        sweep_range = range(args.n, args.n + 1)
    else:
        if not (1 <= args.n_min <= args.n_max <= 50):
            parser.error("require 1 <= --n_min <= --n_max <= 50")
        if args.n_step < 1:
            parser.error("--n_step must be >= 1")
        sweep_range = range(args.n_min, args.n_max + 1, args.n_step)
        if len(sweep_range) == 0:
            parser.error("empty n sweep (check n_min, n_max, n_step)")

    results = run_sweep(
        n_range=sweep_range,
        save_dir=args.save_dir,
        num_eval_episodes=args.eval_episodes,
        auto_train=args.auto_train,
        train_episodes=args.episodes,
        train_n=args.train_n,
        train_reward_shaping=not args.no_shaping,
        train_device=args.device,
        results_file=args.results_file,
        plot_file=args.plot_file,
    )

    print_summary(results)

    if args.plot_training:
        plot_training_curves(save_dir=args.save_dir, plot_file="training_curves.png")


if __name__ == "__main__":
    main()
