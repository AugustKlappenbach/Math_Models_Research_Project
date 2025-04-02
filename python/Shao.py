import numpy as np
import matplotlib.pyplot as plt
import random
from numba import njit
import IPython.display

# --------------------------
# PDE parameters (constants)
# --------------------------
N = 200  # Grid size
Nsteps = 200000  # Number of time steps
Nplot = 20000  # Plot/visualize every Nplot steps

# PDE coefficients
k = 1
sigma = 2.5
h = 1 # You can set h to any constant or array as needed
dx = 0.8
dt = 0.01
dx2 = dx * dx
"Constants from Computational Model for Cell Morphodynamics by Shao"
gamma = 1
kappa = 1
alpha =.1
beta =.2
M_A = 1
A_0 =50.25
epsilon=1
tau=2.62
a=.084
b=1.146
c=.0764
e=.107
DV= .382
DW = .0764





def potential_well(phi):
    "also known as g"
    """potential_well(phi) = 30 * (phi -1)^2 * phi^2."""
    return 18*phi*(1-phi)**2


def D_potential_well(phi):
    """derivative of potential well"""
    return 36*phi*(1-3*phi+2*phi**2)


def DD_potential_well(phi):
    "double derivative of potential well"
    return 36*(1-6*phi+6*phi**2)

def DDD_potential_well(phi):
    "returns the triple derivative of the potiential well"
    return 216*(1-2*phi)


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
@njit
def norm_of_gradiant(phi, i, j):
    "compuntes the norm of the gradient"
    im = i - 1 if i > 0 else N - 1
    ip = i + 1 if i < N - 1 else 0
    jm = j - 1 if j > 0 else N - 1
    jp = j + 1 if j < N - 1 else 0
    phi_dx =1/(dx * 2)*(phi[ip, j] - phi[im, j])
    phi_dy = 1/(dx * 2)*(phi[i, jp] - phi[i, jm])
    return np.sqrt(phi_dx**2 + phi_dy**2)

def f_e_bend(lap_phi, bi_lap_phi, lap_dg, ddg, dg):
    #Free energy that comes from bending
    return -kappa*(bi_lap_phi + ddg/epsilon**2 * lap_phi - 1/epsilon**2*lap_dg - dg*ddg/epsilon**4)
def f_e_tention(lap_phi,dg):
    # Free energy that comes from bending
    return gamma*(lap_phi-dg/epsilon**2)
def f_e_Area(phi, grad_norm):
    return M_a*(sum(phi)*dx2 -A_0)
def f_e_protrusion(V, grad_norm):
    return alpha*V*grad_norm
def f_e_retraction(W, grad_norm):
    return -beta*W*grad_norm
def compute_centroid(phi):
    total_phi = np.sum(phi)
    indices = np.indices(phi.shape)
    cx = np.sum(indices[0] * phi) / total_phi
    cy = np.sum(indices[1] * phi) / total_phi
    return cx, cy
def update_w_and_v(phi, V, W):
    W_new = np.copy(W)
    for i in range(N):
        for j in range(N):
            lap_W = laplacian(W, i, j)
            reaction = - (b * V[i,j] * W[i,j]**2 - e * W[i,j])
            W_new[i,j] += dt * (reaction + DW * lap_W)
    return W_new

# --------------------------
# Allocations & initial data
# --------------------------
phi = np.zeros((N, N), dtype=np.float64)
W = np.zeros_like(phi)  # start with zero or some asymmetric blob
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

def step(phi, phidot):
    """
    One Euler time-step of:
    tau*phi_t = k(lapacian - g''/epsilon^2)(laplacian*phi - g'/epsilon^2) + gamma(laplacian phi -g'/epsilon) -M_A(\int phi dX -A_0)|gradient*phi| + (alpha*V - B*W)|gradient*phi|.
    also known as:
    tau*phi_t = F_E_Bend + F_E_Area+F_E_protrusion + F_E_retraction
    (note that F_E means free energy)
    """

    for i in range(N):
        for j in range(N):
            lap_phi_ij = laplacian(phi, i, j)
            bi_lap_phi_ij =laplacian(lap_phi_ij,i,j)
            Dg_ij = D_potential_well(phi[i, j])
            DDg_ij = DD_potential_well(phi[i,j])
            DDDg_ij =DDD_potential_well(phi[i,j])
            norm_gradiant_ij=norm_of_gradiant(phi,i,j)
            if phi[i,j] ==0:
                v=0
                w=0
            else:
                w=update_w_and_v(phi,V,W)
                v=

    # 3) Compute phidot = PDE RHS
    for i in range(N):
        for j in range(N):
            lap_phi_ij = laplacian(phi, i, j)
            phidot[i, j] = k * (sigma * lap_phi_ij - fprime(phi[i, j]) - p * h * gprime(phi[i, j]))

    # 4) Update phi
    for i in range(N):
        for j in range(N):
            phi[i, j] += dt/tau * phidot[i, j]

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
