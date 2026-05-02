#!/usr/bin/env python3
"""
gravity_sim.py

Interactive 2D N-body gravity simulation using Newton's law of universal gravitation.

Features:
- Uses F = G * (m1*m2) / r^2 to compute pairwise gravity
- Bodies have mass, position (x,y), velocity (vx,vy)
- Integrates motion using a symplectic (semi-implicit) Euler step
- Real-time animation with Pygame
- User input via simple CLI to choose sample system or custom bodies
- Controls: pause (Space), reset (r), toggle force vectors (f), toggle trails (v), zoom in/out (mouse wheel), quit (Esc/close)

Run: python gravity_sim.py

Note: Distances and masses can be provided in SI units (meters, kilograms). The simulation
scales positions to pixels for visualization; you can change `px_per_meter` or use the
zoom controls during runtime.
"""

import sys
import math
import copy
from collections import deque

import pygame
import numpy as np

# Physical constant (m^3 kg^-1 s^-2)
G = 6.67430e-11


class Body:
    """A single point-mass body in 2D space."""

    def __init__(self, name, mass, x, y, vx=0.0, vy=0.0, color=(255, 255, 255)):
        self.name = name
        self.mass = float(mass)
        self.pos = np.array([float(x), float(y)], dtype=float)
        self.vel = np.array([float(vx), float(vy)], dtype=float)
        self.force = np.zeros(2, dtype=float)
        self.color = color
        self.trail = deque(maxlen=800)

    def radius(self, scale=1e9):
        """Visual radius for drawing: scale mass to a circle size.
        The scale parameter adjusts visual size without changing physics.
        """
        return max(2, int((self.mass ** (1 / 3)) / scale))


def compute_forces(bodies, softening=1e3):
    """Compute pairwise gravitational forces on each body.

    softening: small term to avoid singularities when distance is tiny (in meters).
    """
    n = len(bodies)
    for b in bodies:
        b.force.fill(0.0)

    for i in range(n):
        for j in range(i + 1, n):
            bi = bodies[i]
            bj = bodies[j]
            r_vec = bj.pos - bi.pos
            r2 = (r_vec ** 2).sum() + softening ** 2
            r = math.sqrt(r2)
            # magnitude of force
            F = G * bi.mass * bj.mass / r2
            # direction
            if r > 0:
                f_vec = F * (r_vec / r)
            else:
                f_vec = np.zeros(2)
            bi.force += f_vec
            bj.force -= f_vec


def integrate(bodies, dt):
    """Symplectic (semi-implicit) Euler integrator:
    v_{t+dt} = v_t + (F/m) * dt
    x_{t+dt} = x_t + v_{t+dt} * dt
    This is more stable for energy behavior than explicit Euler.
    """
    for b in bodies:
        acc = b.force / b.mass
        b.vel += acc * dt
        b.pos += b.vel * dt
        b.trail.append(tuple(b.pos))


def ask_initial_conditions():
    """Prompt user for number of bodies and their initial conditions, or choose sample."""
    print("Gravity simulation — choose initial configuration")
    print("1) Sample: Sun, Earth, Moon (realistic masses/distances)")
    print("2) Custom (enter number of bodies and values)")
    choice = input("Choose 1 or 2 [1]: ").strip() or "1"

    bodies = []
    if choice == "1":
        # Use SI units: meters, kg, seconds. We'll scale to pixels later.
        # Sun
        sun = Body("Sun", 1.98847e30, 0.0, 0.0, 0.0, 0.0, color=(255, 200, 50))
        # Earth
        earth_dist = 1.496e11
        earth_speed = 29_780.0
        earth = Body("Earth", 5.97237e24, earth_dist, 0.0, 0.0, earth_speed, color=(100, 149, 237))
        # Moon
        moon_dist = earth_dist + 3.844e8
        moon_speed = earth_speed + 1_022.0
        moon = Body("Moon", 7.342e22, moon_dist, 0.0, 0.0, moon_speed, color=(200, 200, 200))
        bodies = [sun, earth, moon]
    else:
        try:
            n = int(input("Number of bodies: ").strip())
        except Exception:
            print("Invalid number, defaulting to 2")
            n = 2
        for i in range(n):
            print(f"--- Body {i + 1} ---")
            name = input(f"Name [{i + 1}]: ").strip() or f"Body{i + 1}"
            mass = float(input("Mass (kg) [1e22]: ").strip() or 1e22)
            x = float(input("x (meters) [0]: ").strip() or 0.0)
            y = float(input("y (meters) [0]: ").strip() or 0.0)
            vx = float(input("vx (m/s) [0]: ").strip() or 0.0)
            vy = float(input("vy (m/s) [0]: ").strip() or 0.0)
            color = tuple(map(int, input("color RGB (e.g. 255,0,0) [200,200,200]: ").strip().split(",") if input else (200, 200, 200)))
            bodies.append(Body(name, mass, x, y, vx, vy, color=color))
    return bodies


def world_to_screen(pos, center, scale, screen_size):
    """Convert world coordinates (meters) to screen pixels."""
    sx = center[0] + pos[0] * scale
    sy = center[1] - pos[1] * scale
    return int(sx), int(sy)


def draw_force_vector(surface, body, center, scale, max_len=50):
    # draw a small arrow showing the force direction on the body
    if np.linalg.norm(body.force) < 1e-12:
        return
    f = body.force
    # scale force magnitude to pixels for visualization (arbitrary)
    mag = math.sqrt((f ** 2).sum())
    # avoid extremely long arrows
    length = math.log1p(mag) * (max_len / 10.0)
    dir_vec = f / (mag + 1e-30)
    start = world_to_screen(body.pos, center, scale, surface.get_size())
    end_pos = (start[0] + int(dir_vec[0] * length), start[1] - int(dir_vec[1] * length))
    pygame.draw.line(surface, (255, 100, 100), start, end_pos, 2)
    # arrow head
    pygame.draw.circle(surface, (255, 100, 100), end_pos, 3)


def run_simulation(bodies):
    pygame.init()
    size = width, height = 1100, 800
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Gravity simulation — Newtonian N-body")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    # Visualization parameters
    center = np.array([width // 2, height // 2], dtype=float)
    # pixels per meter (initial). We pick a value to make Earth distance ~200 px
    px_per_meter = 200.0 / 1.496e11
    scale = px_per_meter

    paused = False
    show_forces = False
    show_trails = True

    # time step in seconds per simulation update
    dt = 60 * 60 * 6.0  # 6 hours

    initial_state = copy.deepcopy(bodies)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    bodies = copy.deepcopy(initial_state)
                    paused = False
                elif event.key == pygame.K_f:
                    show_forces = not show_forces
                elif event.key == pygame.K_v:
                    show_trails = not show_trails
            elif event.type == pygame.MOUSEWHEEL:
                # zoom: wheel up increases scale
                if event.y > 0:
                    scale *= 1.2
                else:
                    scale /= 1.2

        if not paused:
            compute_forces(bodies)
            integrate(bodies, dt)

        # draw
        screen.fill((10, 10, 30))

        # draw trails first
        if show_trails:
            for b in bodies:
                if len(b.trail) > 1:
                    pts = [world_to_screen(np.array(p), center, scale, size) for p in b.trail]
                    if len(pts) > 2:
                        pygame.draw.lines(screen, b.color, False, pts, 1)

        # draw bodies
        for b in bodies:
            pos_px = world_to_screen(b.pos, center, scale, size)
            r = b.radius(scale=1e8)
            pygame.draw.circle(screen, b.color, pos_px, r)
            # label
            label = font.render(b.name, True, (220, 220, 220))
            screen.blit(label, (pos_px[0] + r + 3, pos_px[1] - r - 3))
            if show_forces:
                draw_force_vector(screen, b, center, scale)

        # HUD
        lines = [
            f"dt = {dt:.0f} s   scale = {scale:.2e} px/m   paused = {paused}",
            "Controls: Space=Pause, r=Reset, mouse wheel=Zoom, f=Toggle forces, v=Toggle trails, Esc=Quit",
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (200, 200, 200))
            screen.blit(surf, (8, 8 + i * 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    bodies = ask_initial_conditions()
    print("Starting simulation window...")
    run_simulation(bodies)


if __name__ == "__main__":
    main()
