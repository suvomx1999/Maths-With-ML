#!/usr/bin/env python3
"""
gravity_ml.py

Train a simple machine-learning model to learn one-step N-body dynamics, then
run an interactive visualization that lets you switch between the true
Newtonian simulation and the ML-predicted rollout.

Approach:
- Generate training data by simulating many short trajectories using the
  Newtonian integrator (compute_forces / integrate) from `gravity_sim.py`.
- Represent the state as the flattened vector of [x,y,vx,vy,mass] for each body.
- Train an `MLPRegressor` to predict the state delta (next_state - state).
- In interactive mode you can toggle between "Physics" and "ML" stepping.

Notes / limitations:
- For simplicity this script trains on a fixed number of bodies (3). Extending
  to variable N requires more advanced architectures (graph nets) and padding.
- This is an educational demo showing how a model can learn short-term
  dynamics; due to chaos and accumulation of errors, ML rollouts will
  diverge from true physics over long times.

Run: python gravity_ml.py

Controls in window: Space=Pause, m=Toggle ML/Physics stepping, r=Reset, Esc=Quit
"""

import os
import math
import copy
from collections import deque

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from Gravitation.gravity_sim import Body, compute_forces, integrate, world_to_screen

try:
    import pygame
except Exception:
    raise SystemExit("pygame is required to run this script. Install requirements-gravity.txt")


def state_vector(bodies):
    """Flatten bodies into a 1D state vector: [x,y,vx,vy,mass] * N"""
    return np.concatenate([np.array([b.pos[0], b.pos[1], b.vel[0], b.vel[1], b.mass]) for b in bodies])


def unflatten_state(vec):
    """Convert flattened vector back to list of Body-like dicts (not full Body objects)."""
    n5 = 5
    n = len(vec) // n5
    bodies = []
    for i in range(n):
        s = vec[i * n5:(i + 1) * n5]
        bodies.append({"x": s[0], "y": s[1], "vx": s[2], "vy": s[3], "m": s[4]})
    return bodies


def make_dataset_from_sim(initial_bodies, dt=60 * 60 * 6.0, steps=200, samples=100):
    """Generate dataset by simulating slight perturbations of initial_bodies.

    Returns X, Y where Y = next_state - state
    """
    X = []
    Y = []
    for s in range(samples):
        bodies = copy.deepcopy(initial_bodies)
        # add small random perturbation to velocities and positions
        for b in bodies:
            b.pos += np.random.randn(2) * 1e7
            b.vel += np.random.randn(2) * 10.0

        for t in range(steps):
            st = state_vector(bodies)
            compute_forces(bodies)
            integrate(bodies, dt)
            st_next = state_vector(bodies)
            X.append(st)
            Y.append(st_next - st)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def train_model(X, Y):
    print("Training MLPRegressor on generated data...")
    model = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, random_state=1)
    model.fit(X, Y)
    return model


def build_bodies_from_flat(vec):
    arr = unflatten_state(vec)
    bodies = []
    for i, b in enumerate(arr):
        bodies.append(Body(f"B{i}", b["m"], b["x"], b["y"], b["vx"], b["vy"]))
    return bodies


def rollout_ml(model, state_vec, steps, dt):
    """Rollout using the learned model for `steps` steps starting from `state_vec`.

    Returns array of state vectors.
    """
    traj = np.empty((steps, len(state_vec)))
    s = state_vec.copy()
    for i in range(steps):
        delta = model.predict(s.reshape(1, -1)).ravel()
        s = s + delta
        traj[i] = s
    return traj


def visualize_comparison(true_bodies_init, model, dt=60 * 60 * 6.0):
    pygame.init()
    size = width, height = 1100, 800
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Gravity ML vs Physics")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    center = np.array([width // 2, height // 2], dtype=float)
    px_per_meter = 200.0 / 1.496e11
    scale = px_per_meter

    # initial states
    true_bodies = copy.deepcopy(true_bodies_init)
    ml_bodies = copy.deepcopy(true_bodies_init)

    paused = False
    use_ml = False
    steps = 0

    # keep trails
    for b in true_bodies + ml_bodies:
        b.trail.clear()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    true_bodies = copy.deepcopy(true_bodies_init)
                    ml_bodies = copy.deepcopy(true_bodies_init)
                    for b in true_bodies + ml_bodies:
                        b.trail.clear()
                    steps = 0
                elif event.key == pygame.K_m:
                    use_ml = not use_ml
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    scale *= 1.2
                else:
                    scale /= 1.2

        if not paused:
            # physics step
            compute_forces(true_bodies)
            integrate(true_bodies, dt)

            # ML step: predict next flattened state deltas
            state_ml = state_vector(ml_bodies)
            delta = model.predict(state_ml.reshape(1, -1)).ravel()
            next_state = state_ml + delta
            # update ml_bodies from predicted next_state
            arr = unflatten_state(next_state)
            for i, bdict in enumerate(arr):
                ml_bodies[i].pos[0] = float(bdict["x"])
                ml_bodies[i].pos[1] = float(bdict["y"])
                ml_bodies[i].vel[0] = float(bdict["vx"])
                ml_bodies[i].vel[1] = float(bdict["vy"])
                # mass stays constant
                # append to trail so ML rollout also shows a path
                ml_bodies[i].trail.append(tuple(ml_bodies[i].pos))

            steps += 1

        # Draw
        screen.fill((5, 5, 20))

        # draw true trails
        for b in true_bodies:
            if len(b.trail) > 1:
                pts = []
                for p in b.trail:
                    sp = world_to_screen(np.array(p), center, scale, size)
                    # validate returned screen point is a 2-tuple of finite numbers
                    try:
                        if hasattr(sp, '__len__') and len(sp) == 2:
                            sx, sy = float(sp[0]), float(sp[1])
                            if math.isfinite(sx) and math.isfinite(sy):
                                pts.append((int(sx), int(sy)))
                    except Exception:
                        continue
                if len(pts) > 1:
                    for i in range(len(pts) - 1):
                        try:
                            pygame.draw.line(screen, b.color, pts[i], pts[i + 1], 2)
                        except Exception:
                            continue

        # draw ml trails (dashed / different color shade)
        for b in ml_bodies:
            if len(b.trail) > 1:
                pts = []
                for p in b.trail:
                    sp = world_to_screen(np.array(p), center, scale, size)
                    try:
                        if hasattr(sp, '__len__') and len(sp) == 2:
                            sx, sy = float(sp[0]), float(sp[1])
                            if math.isfinite(sx) and math.isfinite(sy):
                                pts.append((int(sx), int(sy)))
                    except Exception:
                        continue
                # draw fainter
                darker = tuple(max(0, c - 80) for c in b.color)
                if len(pts) > 1:
                    for i in range(len(pts) - 1):
                        try:
                            pygame.draw.line(screen, darker, pts[i], pts[i + 1], 1)
                        except Exception:
                            continue

        # draw bodies
        for b in true_bodies:
            try:
                pos_px = world_to_screen(b.pos, center, scale, size)
                pos_px = (int(pos_px[0]), int(pos_px[1]))
                r = b.radius(scale=1e8)
                pygame.draw.circle(screen, b.color, pos_px, r)
            except Exception:
                continue

        for b in ml_bodies:
            try:
                pos_px = world_to_screen(b.pos, center, scale, size)
                pos_px = (int(pos_px[0]), int(pos_px[1]))
                r = b.radius(scale=1e8)
                # outline for ml bodies
                pygame.draw.circle(screen, (220, 220, 220), pos_px, r, 1)
            except Exception:
                continue

        mode_text = "ML" if use_ml else "Physics"
        info = f"Mode: {mode_text}   dt={dt:.0f}s   steps={steps}   scale={scale:.2e} px/m"
        screen.blit(font.render(info, True, (220, 220, 220)), (8, 8))
        screen.blit(font.render("Space=Pause  m=Toggle ML/Physics  r=Reset  Esc=Quit", True, (200, 200, 200)), (8, 26))

        pygame.display.flip()
        clock.tick(60)


def main():
    # Build a reference initial system (Sun, Earth, Moon)
    sun = Body("Sun", 1.98847e30, 0.0, 0.0, 0.0, 0.0, color=(255, 200, 50))
    earth_dist = 1.496e11
    earth_speed = 29_780.0
    earth = Body("Earth", 5.97237e24, earth_dist, 0.0, 0.0, earth_speed, color=(100, 149, 237))
    moon_dist = earth_dist + 3.844e8
    moon_speed = earth_speed + 1_022.0
    moon = Body("Moon", 7.342e22, moon_dist, 0.0, 0.0, moon_speed, color=(200, 200, 200))
    initial = [sun, earth, moon]

    # Generate training data
    X, Y = make_dataset_from_sim(initial, dt=60 * 60 * 6.0, steps=150, samples=80)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

    model = train_model(X_train, Y_train)
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Test MSE (state delta): {mse:.6e}")

    visualize_comparison(initial, model)


if __name__ == "__main__":
    main()
