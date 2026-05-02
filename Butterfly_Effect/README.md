# Imbalance of Nature — Chaos + ML demo

This small demo generates a Lorenz attractor (chaotic system), trains a small MLP to predict
the next state from the current state, and saves visualizations showing how the learned
model diverges from the true chaotic trajectory.

Quick start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo:

```bash
python imbalance_nature.py
```

Outputs

- `outputs/attractor.png`: 3D plot of the true Lorenz attractor
- `outputs/prediction.png`: comparison of true vs predicted trajectories

Notes

- This is an experimental, illustrative demo — not a production model. Adjust `max_iter` and
  model sizes in `imbalance_nature.py` to change training behaviour.
