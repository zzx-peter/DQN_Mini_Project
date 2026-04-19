# DQN Mini Project

A small project that uses Deep Q-Networks (DQN) to learn bit-wise reproduction of random binary target sequences. The code is split into two versions:

## `DQN` (baseline)

Standard DQN with **`(2n+1)`-dimensional observations** (full target string, generated prefix, normalized timestep). You typically **train one checkpoint per sequence length `n`**; weights are saved as `models/dqn_nXX.pt`.

## `DQN_final` (improved)

The environment uses a **fixed 2D observation** (target bit at the current index and `step/n`), **independent of `n`**. After training once, a single `dqn.pt` can be evaluated across multiple lengths.

## Usage

Each subdirectory has its own detailed `README.md`. Install dependencies and run `train.py` / `evaluate.py` from that folder.

**Note:** I kept the first version (`DQN`) to show the improvement made in Task 3. The CLI arguments for `train.py` and `evaluate.py` differ slightly between the two version. Unlike `DQN`, `DQN_final` has no workflow that trains many different sequence lengths (`--n_max` in `DQN`); therefore uses `--train_n` where needed to pick the training length—the same role as `--n` when training in `DQN`. A few extra convenience flags were added as well; see each subdirectory’s `README.md` for the exact differences.
