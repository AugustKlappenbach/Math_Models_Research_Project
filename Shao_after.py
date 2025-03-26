import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import clip


N = 200
Nsteps = 200
Nplot = 10

dx = 0.8
dt = 0.01
dx2 = dx * dx

# Constants
kappa = 1
gamma = 1
alpha = .1
beta = .2
M_A = 1
A_0 = 50.25
epsilon = 1
tau = 2.62
a = .084
b = 1.146
c = .0764
e = .107
DV = .382
DW = .0764


def potential_well(phi):
    return 18 * phi * (1 - phi) ** 2

def D_potential_well(phi):
    phi = np.clip(phi, 0, 5)
    return 36 * phi * (1 - 3 * phi + 2 * phi ** 2)

def DD_potential_well(phi):
    phi = np.clip(phi, 0, 5)
    return 36 * (1 - 6 * phi + 6 * phi ** 2)

def DDD_potential_well(phi):
    phi = np.clip(phi, 0, 5)
    return 216 * (1 - 2 * phi)

@njit(cache=True)
def get_coordinates(c1 , c2):
    im = c1 - 1 if c1 > 0 else N - 1
    ip = c1 + 1 if c1 < N - 1 else 0
    jm = c2 - 1 if c2 > 0 else N - 1
    jp = c2 + 1 if c2 < N - 1 else 0
    return im,ip,jm,jp


@njit
def laplacian(phi, i, j):
    im, ip, jm, jp = get_coordinates(i,j)
    return (phi[ip, j] + phi[im, j] + phi[i, jp] + phi[i, jm] - 4.0 * phi[i, j]) / dx2



@njit
def norm_of_gradient(phi, i, j):
    im, ip, jm, jp = get_coordinates(i,j)

    phi_dx = 1 / (2 * dx) * (phi[ip, j] - phi[im, j])
    phi_dy = 1 / (2 * dx) * (phi[i, jp] - phi[i, jm])
    return np.sqrt(phi_dx ** 2 + phi_dy ** 2)

epsilon_sq = epsilon **2
epsilon_sq_sq = epsilon_sq ** 2

@njit(cache=True)
def f_e_bend(lap_phi, bi_lap_phi, lap_dg, ddg, dg):
    result = -kappa * (bi_lap_phi + ddg / epsilon_sq * lap_phi - 1 / epsilon_sq * lap_dg - dg * ddg / epsilon_sq_sq)
    return clip.clip_num(result, -1e4, 1e4)

@njit(cache=True)
def f_e_tension(lap_phi, dg):
    return gamma * (lap_phi - dg / epsilon ** 2)

def update_w_and_v(phi, V, W):
    V_new = np.copy(V)
    W_new = np.copy(W)
    np.clip(W[0, N-1], 0, 5)
    np.clip(V[0, N-1], 0, 5)
    phiV = phi * V
    phiW = phi * W
    for i in range(N):
        for j in range(N):
            filled = phi[i,j]
            if filled == 0:
                continue

            lapV = laplacian(phiV, i, j)
            lapW = laplacian(phiW, i, j)
            Wij = W[i, j]
            Vij = V[i, j]
            dW = filled * (b * Vij * Wij ** 2 - e * Wij) + DW * lapW
            dV = filled * (a - b * Vij * Wij ** 2 - c * Vij) + DV * lapV
            W_new[i, j] += dt * dW / filled
            V_new[i, j] += dt * dV / filled

    return V_new, W_new

phi = np.zeros((N, N))

# Initialize phi as a circle of radius 4 microns
center = N // 2
radius = int(4 / dx)
for i in range(N):
    for j in range(N):
        if np.sqrt((i - center) ** 2 + (j - center) ** 2) <= radius:
            phi[i, j] = 1.0



def step(phi, phidot, V, W):
    lap_phi = np.zeros_like(phi)
    bi_lap_phi = np.zeros_like(phi)

    for i in range(N):
        for j in range(N):
            lap_phi[i, j] = laplacian(phi, i, j)
    for i in range(N):
        for j in range(N):
            bi_lap_phi[i, j] = laplacian(lap_phi, i, j)

    dg = D_potential_well(phi)
    ddg = DD_potential_well(phi)
    lap_dg = np.zeros_like(phi)
    for i in range(N):
        for j in range(N):
            lap_dg[i, j] = laplacian(dg, i, j)

    V, W = update_w_and_v(phi, V, W)
    area_diff = clip.clip_num(np.sum(phi) * dx2 - A_0, -1e3, 1e3)

    for i in range(N):
        for j in range(N):
            norm_grad = norm_of_gradient(phi, i, j)
            bend = f_e_bend(lap_phi[i, j], bi_lap_phi[i, j], lap_dg[i, j], ddg[i, j], dg[i, j])
            tension = f_e_tension(lap_phi[i, j], dg[i, j])
            area_term = -M_A * area_diff * norm_grad
            raw_active = (alpha * V[i, j] - beta * W[i, j]) * norm_grad
            active = clip.clip_num(raw_active, -1e3, 1e3)
            total = bend + tension + area_term + active
            phidot[i, j] = clip.clip_num(total, -1e5, 1e5)
    phi += dt / tau * phidot
    phi = np.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=0.0)
    return phi, phidot, V, W

plt.figure(figsize=(6, 6))
plt.ion()
plt.imshow(phi, cmap="cool", vmin=0, vmax=1)
plt.colorbar(label="Phi Value")
plt.title("Initial Condition: Circular Region (radius 4 μm)")
plt.draw()
plt.pause(.1)
print("starting")

def main(phi):
    phidot = np.zeros_like(phi)
    W = np.zeros_like(phi)
    W_0 = 0.2
    for i in range(N):
        for j in range(N):
            y = (j - center) * dx  # grid -> physical y
            if y < 0 and phi[i, j] > 0.5:
                W[i, j] = W_0

    V = np.ones_like(phi) * 1.1
    for step_count in range(1, Nsteps + 1):
        phi, phidot, V, W = step(phi, phidot, V, W)

        if step_count % Nplot == 0:
            plt.figure(figsize=(6, 5))
            plt.imshow(phi, cmap="cool", vmin=0, vmax=1)
            plt.title(f"Step = {step_count}")
            plt.colorbar()
            plt.draw()
            plt.pause(.1)


main(phi)