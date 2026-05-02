#!/usr/bin/env python3
"""
imbalance_nature.py

Simple demo: generate a chaotic Lorenz attractor, train an MLP to predict the next state,
and save plots showing how the learned model diverges — a metaphor for "imbalance of nature".

Usage: python imbalance_nature.py
Outputs: outputs/attractor.png, outputs/prediction.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def lorenz(s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = s
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def integrate_lorenz(initial, dt=0.01, steps=5000):
    traj = np.empty((steps, 3), dtype=float)
    s = np.array(initial, dtype=float)
    for i in range(steps):
        # simple RK4 integration for stability
        k1 = lorenz(s)
        k2 = lorenz(s + 0.5 * dt * k1)
        k3 = lorenz(s + 0.5 * dt * k2)
        k4 = lorenz(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = s
    return traj


def make_dataset(traj):
    X = traj[:-1]
    y = traj[1:]
    return X, y


def train_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=0)
    model.fit(X_train, y_train)
    return model


def save_attractor_plot(traj, outpath):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5)
    ax.set_title("Lorenz attractor (true)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def save_prediction_plot(true_traj, pred_traj, outpath):
    t = np.arange(len(true_traj))
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axs[i].plot(t, true_traj[:, i], label="true", lw=1)
        axs[i].plot(t, pred_traj[:, i], label="pred", lw=1)
        axs[i].set_ylabel(labels[i])
        axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("time step")
    fig.suptitle("True vs predicted trajectory (test set)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def rollout_predict(model, x0, steps):
    pred = np.empty((steps, 3), dtype=float)
    s = x0.copy()
    for i in range(steps):
        s = model.predict(s.reshape(1, -1)).ravel()
        pred[i] = s
    return pred


def main():
    rng = np.random.RandomState(0)
    initial = rng.randn(3) * 0.1 + np.array([0.0, 1.0, 1.05])

    print("Generating Lorenz trajectory...")
    traj = integrate_lorenz(initial, dt=0.01, steps=4000)

    X, y = make_dataset(traj)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print("Training MLP model to predict next state...")
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")

    # Prepare outputs
    outdir = os.path.join("outputs")
    os.makedirs(outdir, exist_ok=True)

    print("Saving attractor plot...")
    save_attractor_plot(traj, os.path.join(outdir, "attractor.png"))

    # Pick a random seed from test set and rollout to show divergence
    idx = 0
    x0 = X_test[idx]
    steps = min(800, len(y_test))
    true_rollout = np.vstack([X_test[idx:idx + steps], y_test[idx + steps - 1: idx + steps - 1 + 1]])[:steps]
    pred_rollout = rollout_predict(model, x0, steps)

    print("Saving prediction comparison plot...")
    save_prediction_plot(true_rollout, pred_rollout, os.path.join(outdir, "prediction.png"))

    print("Done. Outputs in the 'outputs' directory.")


if __name__ == "__main__":
    main()
