"""
train.py — Train one DQN with fixed sequence length n.

Observations are [target at current step, step/n] (dim 2). Checkpoints:

    {save_dir}/dqn.pt
    {save_dir}/history.json
"""

import os
import json
import argparse

import numpy as np

from env import BitSequenceEnv
from agent import DQNAgent


# ---------------------------------------------------------------------------
HPARAMS = dict(
    lr=1e-3,
    gamma=0.99,
    buffer_size=30_000,
    batch_size=1024,
    target_update_freq=200,
    hidden_sizes=(256, 256),
    epsilon=1.0,
    epsilon_end=0.05,
)

OBS_DIM = 2
CKPT_FILENAME = "dqn.pt"
HISTORY_FILENAME = "history.json"


def get_episode_count(n: int) -> int:
    """Default episode budget for sequence length n."""
    if n <= 5:
        return 3_000
    if n <= 10:
        return 8_000
    if n <= 20:
        return 20_000
    if n <= 30:
        return 40_000
    return 60_000


def train_dqn(
    n: int,
    num_episodes: int | None = None,
    save_dir: str = "models",
    reward_shaping: bool = True,
    verbose: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Train DQN at a single sequence length n (same env every episode).

    Returns:
        History dict with keys: n, rewards, successes, final_success_rate.
    """
    assert 1 <= n <= 50
    if num_episodes is None:
        num_episodes = get_episode_count(n)

    env = BitSequenceEnv(n=n, reward_shaping=reward_shaping)
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=2, device=device, **HPARAMS)
    os.makedirs(save_dir, exist_ok=True)

    eps_start = 1.0
    eps_end = 0.05
    explore_episodes = max(1, int(0.6 * num_episodes))
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

        if ep < explore_episodes:
            agent.epsilon = max(eps_end, agent.epsilon * eps_decay)
        else:
            agent.epsilon = eps_end

        if verbose and (ep + 1) % LOG_EVERY == 0:
            sr = np.mean(success_hist[-WINDOW:])
            print(
                f"  n={n:2d} | ep {ep + 1:6d}/{num_episodes} | "
                f"success(last {WINDOW}): {sr:.3f} | eps: {agent.epsilon:.4f}"
            )

    ckpt_path = os.path.join(save_dir, CKPT_FILENAME)
    agent.save(ckpt_path)

    final_sr = float(np.mean(success_hist[-WINDOW:]))
    history = {
        "n": n,
        "rewards": rewards_hist,
        "successes": success_hist,
        "final_success_rate": final_sr,
    }
    hist_path = os.path.join(save_dir, HISTORY_FILENAME)
    with open(hist_path, "w") as f:
        json.dump(history, f)

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN with obs [target[t], step/n] at a fixed sequence length n.",
    )
    parser.add_argument("--n", type=int, default=8, help="Sequence length (1–50)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Training episodes (default: auto by n)",
    )
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument(
        "--no_shaping",
        action="store_true",
        help="Sparse 0/1 per-step reward instead of shaped (1/n)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args()

    if not (1 <= args.n <= 50):
        parser.error("--n must be between 1 and 50")

    print(f"Training n={args.n} (obs dim {OBS_DIM}) → {os.path.join(args.save_dir, CKPT_FILENAME)}")
    h = train_dqn(
        n=args.n,
        num_episodes=args.episodes,
        save_dir=args.save_dir,
        reward_shaping=not args.no_shaping,
        device=args.device,
    )
    print(f"\nFinal success rate (last window): {h['final_success_rate']:.3f}")


if __name__ == "__main__":
    main()
