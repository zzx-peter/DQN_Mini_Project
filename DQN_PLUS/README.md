# DQN Bit-Sequence Generation

A standard **Deep Q-Network (DQN)** that learns to reproduce a random binary
target sequence of length *n* (1 ≤ n ≤ 50), token by token.

Observations are **always** `[target[t], step/n]` (shape **2)** — input size does
not depend on *n*. Training uses **one fixed *n*** per run (`train.py --n`);
evaluation can still sweep many lengths with the same `dqn.pt`.

---

## Project structure

```
DQN_PLUS/
├── env.py          # Custom Gymnasium environment
├── network.py      # Q-Network MLP definition
├── agent.py        # ReplayBuffer + DQNAgent
├── train.py        # Train at one n → models/dqn.pt
├── evaluate.py     # Sweep n = 1 … n_max with that checkpoint
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quick start

### Train

```bash
# Default n=8 (episode budget from get_episode_count(8))
python train.py

python train.py --n 20
python train.py --n 20 --episodes 15000
python train.py --device cuda
python train.py --no_shaping
```

### Evaluate across *n*

```bash
python evaluate.py
# Custom range: n = n_min, n_min+step, … up to n_max (inclusive)
python evaluate.py --n_min 5 --n_max 45 --n_step 5

python evaluate.py --n_max 30 --eval_episodes 1000

# If dqn.pt missing: train at n=8 (override with --train_n), then evaluate
python evaluate.py --auto_train
python evaluate.py --auto_train --train_n 20

python evaluate.py --plot_training
```

---

## Method

### Environment (`env.py`)

| Item | Detail |
|---|---|
| Episode | Fixed *n* timesteps |
| Observation | `[target[t], step/n]` — shape `(2)` |
| Action | Discrete(2): append **0** or **1** |

**Shaping** (default): reward per step = `(1/n)` if the bit matches, else `0`.  
**Sparse** (`--no_shaping`): 1/0 per step.

### DQN

* 2-layer MLP, **input dim 2**; replay 30k; target update every 200 steps;
  Huber loss; Adam lr 1e-3; grad clip 10.

### Training (`train.py`)

One `BitSequenceEnv(n)` for the whole run; each episode calls `reset()`.
Default episode count follows `get_episode_count(n)`.

---

## Evaluation metric

**Success Rate** = greedy rollouts where the full sequence matches the target
(default 500 episodes per evaluated *n*).

---

## Output files

| File | Description |
|---|---|
| `models/dqn.pt` | Checkpoint |
| `models/history.json` | Training curve (includes training `n`) |
| `eval_results.json` | Sweep results |
| `success_rate.png` | Success rate vs *n* |
| `training_curves.png` | Smoothed training success (`--plot_training`) |
