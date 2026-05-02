# Gravity Simulation (Newtonian N-body)

This project contains `gravity_sim.py`, an interactive 2D N-body gravity simulator.

Quick start

1. Install dependencies (use the gravity-specific file):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gravity.txt
```

2. Run the simulator:

```bash
python gravity_sim.py
```

Controls

- Space: Pause / resume
- r: Reset to initial conditions
- Mouse wheel: Zoom in/out
- f: Toggle force vectors
- v: Toggle trails
- Esc or close window: Quit

Notes

- Provide initial conditions via the CLI prompt when the program starts. Choose the sample
  Sun/Earth/Moon or enter custom bodies with mass (kg), positions (m), and velocities (m/s).
- Distances are shown in pixels using an initial scale chosen so Earth's orbit is visible; use
  the mouse wheel to zoom.
