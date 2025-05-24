# Re-import libraries after code execution environment reset
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
N = 50                  # Number of particles
steps = 5000            # Total Monte Carlo steps
L = 1.0                 # Box length (1x1x1 cube)
delta = 0.1             # Max displacement per move
beta = 1.0              # Inverse temperature
k = 1.0                 # Spring constant (toward center)

# --- Initialization ---
positions = np.random.uniform(0, L, size=(N, 3))  # N particles in 3D
energy = lambda x: np.sum((x - 0.5)**2)           # U = sum over particles

# --- Data Storage ---
trajectory = []
energy_log = []

# --- Monte Carlo Loop ---
for step in range(steps):
    # Pick a random particle
    i = np.random.randint(0, N)
    old_pos = positions[i].copy()
    new_pos = old_pos + np.random.uniform(-delta, delta, size=3)

    # Reflect at boundaries
    if np.any(new_pos < 0) or np.any(new_pos > L):
        continue  # Invalid move, skip

    # Compute energy change (local)
    U_old = energy(old_pos)
    U_new = energy(new_pos)
    delta_U = U_new - U_old

    # Accept or reject move
    if delta_U < 0 or np.random.rand() < np.exp(-beta * delta_U):
        positions[i] = new_pos  # Accept

    # Record data
    if step % 10 == 0:
        trajectory.append(positions.copy())
        total_energy = np.sum([energy(p) for p in positions])
        energy_log.append(total_energy)

# --- Visualization ---
trajectory = np.array(trajectory)

# Plot energy vs. time
plt.figure(figsize=(10, 4))
plt.plot(energy_log)
plt.xlabel("Sample Index")
plt.ylabel("Total Energy")
plt.title("Monte Carlo Simulation: Total Energy Over Time (50 particles in 3D)")
plt.grid(True)
plt.show()
