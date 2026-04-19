"""
train.py — Training script for the DQN bit-sequence agent.

Usage examples
--------------
# Train a single model (n=8, auto episode count)
python train.py --n 8

# Train a single model with explicit episode count
python train.py --n 15 --episodes 15000

# Train all lengths from 1 to 50 sequentially
python train.py --all --n_max 50

# Use sparse reward (harder, but purer RL signal)
python train.py --n 8 --no_shaping

Training strategy
-----------------
- Token-by-token generation: at each of the n timesteps the agent picks 0 or 1.
- Observation: full `[target | generated_so_far | step/n]` or compact `[target[t], step/n]` (see `--compact_obs`).
- Reward shaping: fraction of correct bits (default) or pure sparse 0/1.
- Epsilon decays linearly from 1.0 → 0.05 over the first 60% of episodes,
  then stays fixed at 0.05.  This is done per-episode so the exploration
  budget scales with problem size, not with the number of gradient steps.
- Episode counts scale automatically with n (see get_episode_count()).
"""

import os
import json
import argparse

import numpy as np
import torch

from env import BitSequenceEnv
from agent import DQNAgent


# ---------------------------------------------------------------------------
# Default hyper-parameters (shared with evaluate.py via direct import)
# ---------------------------------------------------------------------------
HPARAMS = dict(
    lr=1e-3,
    gamma=0.99,
    buffer_size=30_000,
    batch_size=128,
    target_update_freq=200,
    hidden_sizes=(256, 256),
    epsilon=1.0,
    epsilon_end=0.05,
)


def get_episode_count(n: int) -> int:
    """
    Heuristic episode budget that grows with sequence length.
    Larger n has an exponentially harder random-match probability,
    so we allocate more episodes for exploration and learning.
    """
    if n <= 5:
        return 3_000
    if n <= 10:
        return 8_000
    if n <= 20:
        return 20_000
    if n <= 30:
        return 40_000
    return 60_000


# ---------------------------------------------------------------------------
def train_single(
    n: int,
    num_episodes: int | None = None,
    save_dir: str = "models",
    reward_shaping: bool = True,
    verbose: bool = True,
    device: str = "cpu",
    compact_obs: bool = False,
) -> dict:
    """
    Train a DQN agent for a fixed sequence length n.

    Args:
        n             : Sequence length (1–50).
        num_episodes  : Override the auto-scaled episode budget.
        save_dir      : Directory where checkpoint and history are saved.
        reward_shaping: Use shaped reward (fraction of correct bits).
        verbose       : Print progress every 10% of training.
        device        : 'cpu' or 'cuda'.
        compact_obs   : If True, observation is `[target[t], step/n]` shape (2,).

    Returns:
        History dict with keys: n, rewards, successes, final_success_rate.
    """
    if num_episodes is None:
        num_episodes = get_episode_count(n)

    env = BitSequenceEnv(n=n, reward_shaping=reward_shaping, compact_obs=compact_obs)
    obs_dim = env.observation_space.shape[0]

    agent = DQNAgent(obs_dim=obs_dim, n_actions=2, device=device, **HPARAMS)

    os.makedirs(save_dir, exist_ok=True)

    # ---- epsilon schedule: linear decay over first 60% of episodes ----------
    eps_start = 1.0
    eps_end = 0.05
    explore_episodes = max(1, int(0.6 * num_episodes))
    # per-episode multiplicative factor so that after explore_episodes steps
    # epsilon equals eps_end
    eps_decay = (eps_end / eps_start) ** (1.0 / explore_episodes)

    rewards_hist: list[float] = []
    success_hist: list[float] = []
    LOG_EVERY = max(1, num_episodes // 10)
    WINDOW = min(200, num_episodes)

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_reward += reward

        rewards_hist.append(ep_reward)
        success_hist.append(float(info["success"]))

        # Decay epsilon per episode (stops decaying after explore_episodes)
        if ep < explore_episodes:
            agent.epsilon = max(eps_end, agent.epsilon * eps_decay)
        else:
            agent.epsilon = eps_end

        if verbose and (ep + 1) % LOG_EVERY == 0:
            sr = np.mean(success_hist[-WINDOW:])
            print(
                f"  n={n:2d} | ep {ep + 1:6d}/{num_episodes} | "
                f"success(last {WINDOW}): {sr:.3f} | "
                f"eps: {agent.epsilon:.4f}"
            )

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, f"dqn_n{n:02d}.pt")
    agent.save(ckpt_path)

    final_sr = float(np.mean(success_hist[-WINDOW:]))
    history = {
        "n": n,
        "rewards": rewards_hist,
        "successes": success_hist,
        "final_success_rate": final_sr,
    }

    # Persist training history (used later by evaluate.py for plots)
    hist_path = os.path.join(save_dir, f"history_n{n:02d}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    return history


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train DQN agent for bit-sequence matching."
    )
    parser.add_argument("--n", type=int, default=8, help="Sequence length (1–50)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all lengths from 1 to --n_max sequentially",
    )
    parser.add_argument("--n_max", type=int, default=50, help="Upper bound when --all")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episode count (default: auto-scaled by n)",
    )
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument(
        "--no_shaping",
        action="store_true",
        help="Use sparse 0/1 reward instead of shaped reward",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device: 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--compact_obs",
        action="store_true",
        help=(
            "Use minimal observation [target at current step, step/n] (dim 2) "
            "instead of full target + prefix (dim 2n+1)."
        ),
    )
    args = parser.parse_args()

    if args.all:
        for n in range(1, args.n_max + 1):
            sep = "=" * 55
            print(f"\n{sep}")
            print(f"  Training n = {n}  (budget: {get_episode_count(n)} episodes)")
            print(sep)
            h = train_single(
                n=n,
                num_episodes=args.episodes,
                save_dir=args.save_dir,
                reward_shaping=not args.no_shaping,
                device=args.device,
                compact_obs=args.compact_obs,
            )
            print(f"  >> n={n:2d}: final success rate = {h['final_success_rate']:.3f}")
    else:
        print(f"Training n = {args.n}")
        h = train_single(
            n=args.n,
            num_episodes=args.episodes,
            save_dir=args.save_dir,
            reward_shaping=not args.no_shaping,
            device=args.device,
            compact_obs=args.compact_obs,
        )
        print(f"\nFinal success rate (last 200 episodes): {h['final_success_rate']:.3f}")


if __name__ == "__main__":
    main()
