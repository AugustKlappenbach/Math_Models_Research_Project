import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
import IPython.display

# --------------------------
# PDE parameters (constants)
# --------------------------
N = 200  # Grid size
Nsteps = 20000  # Number of time steps
Nplot = 2000  # Plot/visualize every Nplot steps

# PDE coefficients
k = 1
sigma = 2.5
h = 1 # You can set h to any constant or array as needed
dx = 0.8
dt = 0.01
dx2 = dx * dx


# --------------------------
# Derived/Helper functions
# --------------------------
@njit
def fprime(phi):
    """f'(phi) = 8 * phi * (2 phi^2 - 3 phi + 1)."""
    return 8.0 * phi * (2.0 * phi ** 2 - 3.0 * phi + 1.0)


@njit
def gprime(phi):
    """g'(phi) = 30 * (phi -1)^2 * phi^2."""
    return 30.0 * (phi - 1.0) ** 2 * phi ** 2


@njit
def laplacian(phi, i, j):
    """
    2D five-point stencil with periodic boundary conditions.
    (phi_{i+1,j} + phi_{i-1,j} + phi_{i,j+1} + phi_{i,j-1} - 4*phi_{i,j}) / dx^2
    """
    # Since we are inside a @njit function, we can't do Pythonic i+1 mod N directly,
    # so we handle periodic indexing manually:
    im = i - 1 if i > 0 else N - 1
    ip = i + 1 if i < N - 1 else 0
    jm = j - 1 if j > 0 else N - 1
    jp = j + 1 if j < N - 1 else 0

    return (phi[ip, j] + phi[im, j] + phi[i, jp] + phi[i, jm] - 4.0 * phi[i, j]) / dx2


# --------------------------
# Allocations & initial data
# --------------------------
phi = np.zeros((N, N), dtype=np.float64)
phidot = np.zeros((N, N), dtype=np.float64)

N = 200
phi = np.zeros((N, N), dtype=np.float64)

# Make a central square from [N/4 : 3N/4] × [N/4 : 3N/4] = 1
for i in range(N):
    for j in range(N):
        if (N//4 <= i < 3*N//4) and (N//4 <= j < 3*N//4):
            phi[i, j] = 1.0
        else:
            phi[i, j] = 0.0

# --------------------------
# Time stepping function
# --------------------------
@njit
def step(phi, phidot):
    """
    One Euler time-step of:
    phi_t = k[ sigma Lap(phi) - f'(phi) - p h g'(phi) ].

    p is updated each time using the formula:
      p = [ ∫ g'(phi) * k(σ Lap(phi) - f'(phi)) dX ] / [ ∫ h g'(phi) dX ]
    """
    # 1) Compute the integrals needed for p.
    numerator = 0.0
    denominator = 0.0

    for i in range(N):
        for j in range(N):
            lap_phi_ij = laplacian(phi, i, j)
            gp_ij = gprime(phi[i, j])
            # Summation for numerator
            numerator += gp_ij * k * (sigma * lap_phi_ij - fprime(phi[i, j]))
            # Summation for denominator
            denominator += h* gp_ij**2

    # Multiply by dx^2 if you want "actual" integrals,
    # but if h is constant and you are consistent, you can skip or keep it:
    # numerator *= dx2
    # denominator *= dx2

    # 2) Compute p
    #    Avoid zero denominators if g'(phi)==0 everywhere.
    if abs(denominator) < 1e-14:
        p = 0.0
    else:
        p = numerator / denominator

    # 3) Compute phidot = PDE RHS
    for i in range(N):
        for j in range(N):
            lap_phi_ij = laplacian(phi, i, j)
            phidot[i, j] = k * (sigma * lap_phi_ij - fprime(phi[i, j]) - p * h * gprime(phi[i, j]))

    # 4) Update phi
    for i in range(N):
        for j in range(N):
            phi[i, j] += dt * phidot[i, j]

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.imshow(phi, cmap="cool", vmin=0, vmax=1)
plt.colorbar(label="Phi Value")
plt.title("Initial Condition: Square Region")
plt.show()

# --------------------------
# Main time-integration loop
# --------------------------
for step_count in range(1, Nsteps + 1):
    step(phi, phidot)

    # Optional quick visualization
    if step_count % Nplot == 0:
        plt.figure(figsize=(6, 5))
        plt.imshow(phi, cmap="cool", vmin=0 , vmax=1)
        plt.title(f"Step = {step_count}")
        plt.colorbar()
        plt.show()