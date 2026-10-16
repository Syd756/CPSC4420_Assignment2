# CPSC4420_Assignment2
## Q-Learning on CliffWalking

CPSC 4420 – Assignment 2. A Q-learning agent trained on Gymnasium's `CliffWalking-v1` environment, with experiments on the learning rate, discount factor, exploration rate, and action noise.

## Requirements

- Python [3.14.7]
- gymnasium (with toy-text for rendering)
- numpy
- matplotlib

## Setup

```
pip install "gymnasium[toy-text]" numpy matplotlib
```

## How to Run

```
python Q-Learning_Part_B-G.py
```

The program asks two questions:

1. **Mode**
   - `1`: No rendering (fast; use this to reproduce the plots)
   - `2`: Rendering (opens a window showing the agent; much slower)

> **Note:** In rendering mode (`2`), the program does not close on its own. To stop it, click on the terminal window and press **Ctrl + C**.

2. **Experiment**
   - `1`: Baseline (Part C)
   - `2`: Learning rate α = 0.1, 0.5, 0.9 (Part D)
   - `3`: Discount factor γ = 0.1, 0.5, 0.9 (Part E)
   - `4`: Exploration rate ε = 0.05, 0.25, 0.5 (Part F)
   - `5`: Best agent vs. baseline, with 30-episode greedy evaluation (Part G)

To run the action-noise experiment (Part H):

```
python Q-Learning_Part_H.py
```

All experiments use `np.random.seed(0)`, so results are reproducible.

## Files

- `Q-Learning_Part_B-G.py`: Q-learning agent and experiments for Parts B–G
- `Q-Learning_Part_H.py`: action-noise experiment (Part H)
- `results/`: plots for each experiment
- `cliffwalking_trained_agent.mp4`: video of the trained agent

## Results

Both the baseline and the best configuration learned the optimal path. The best configuration was learned faster and more stably during training.
