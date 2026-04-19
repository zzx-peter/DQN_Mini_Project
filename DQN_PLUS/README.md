# DQN Bit-Sequence Generation

A standard **Deep Q-Network (DQN)** that learns to reproduce a random binary
target sequence of length *n* (1 ≤ n ≤ 50), token by token.

---

## Project structure

```
mini_project/
├── env.py          # Custom Gymnasium environment
├── network.py      # Q-Network MLP definition
├── agent.py        # ReplayBuffer + DQNAgent
├── train.py        # Training script (CLI)
├── evaluate.py     # Evaluation script — sweeps n=1…50, plots results
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

### Train a single model

```bash
# Train for n=8 (auto episode budget = 8 000)
python train.py --n 8

# Minimal observation [target[t], step/n] (recommended for easier value learning)
python train.py --n 8 --compact_obs

# Train for n=20 with a custom episode count
python train.py --n 20 --episodes 15000

# Use sparse 0/1 reward instead of the default shaped reward
python train.py --n 8 --no_shaping
```

### Train all n = 1 … 50 sequentially

```bash
python train.py --all --n_max 50
```

### Evaluate pre-trained models

```bash
# Evaluate all checkpoints already saved in models/
python evaluate.py

# Auto-train any missing models, then evaluate
python evaluate.py --auto_train --n_max 50

# Only evaluate (or auto-train) a single length, same as train.py --n
python evaluate.py --n 12
python evaluate.py --n 12 --auto_train --compact_obs

# Also plot per-n training curves (requires history JSON files)
python evaluate.py --plot_training
```

---

## Method

### Environment (`env.py`)

| Item | Detail |
|---|---|
| Episode | Fixed n timesteps |
| Observation | **Full (default):** `[target (n), generated_so_far (n), step/n]` — shape `(2n+1)`. **Compact:** `[target[t], step/n]` — shape `(2)` — enable with `--compact_obs` in `train.py` / `evaluate.py`. |
| Action | Discrete(2): append **0** or **1** |
| Reward | See below |

**Rewards (step-level):** Every timestep, reward depends only on whether the bit just placed matches `target` at that index.

**Shaping** (default on): reward per step = `(1/n)` if the bit matches, else `0`. Summing over the episode gives `(#correct_bits)/n`.

**Sparse** (`--no_shaping`): reward per step = `1` if that bit matches, else `0`. Episode return = number of correct bits.

Terminal `info["success"]` is still whether the **full** sequence equals the target (used for metrics).

### DQN (`network.py`, `agent.py`)

* **Q-Network**: 2-layer MLP; input dim is `(2n+1)` full obs, or `(2)` with `--compact_obs`
* **Replay buffer**: 30 000 transitions, sampled uniformly
* **Target network**: hard update every 200 gradient steps
* **Loss**: Huber (SmoothL1)
* **Optimiser**: Adam, lr = 1e-3
* **Gradient clipping**: max-norm 10

### Epsilon schedule (`train.py`)

Epsilon decays **per episode** (not per gradient step) from 1.0 → 0.05
over the first 60 % of training episodes, then stays fixed at 0.05.  
Because the budget scales with n, exploration is proportionally longer for
harder problems.

### Episode budgets

| n | Episodes |
|---|---|
| 1 – 5 | 3 000 |
| 6 – 10 | 8 000 |
| 11 – 20 | 20 000 |
| 21 – 30 | 40 000 |
| 31 – 50 | 60 000 |

---

## Evaluation metric

**Success Rate** = fraction of greedy rollouts in which the generated sequence
exactly matches the target.  Measured over 500 episodes (default) per n.

---

## Output files

After training and evaluation the following files are produced:

| File | Description |
|---|---|
| `models/dqn_n??.pt` | Checkpoint for each n |
| `models/history_n??.json` | Per-episode reward & success history |
| `eval_results.json` | Aggregated evaluation results |
| `success_rate.png` | Success rate vs n bar/line chart |
| `training_curves.png` | Per-n smoothed success curves (with `--plot_training`) |
