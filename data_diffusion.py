import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# torch.manual_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_trajectory(D, kappa, s_inner, Ns, K, dt, u_L=0, u_R=0, sigma=1):
    """
    Generate a numerical trajectory of the stochastic diffusion process.

    Parameters
    ----------
    D : float
        Diffusion coefficient.
    kappa : float
        Reaction or drift parameter in the governing equation.
    s_inner : array-like
        Spatial grid points in the interior domain (excluding boundaries).
    Ns : int
        Number of spatial discretization points.
    K : int
        Number of temporal discretization steps.
    dt : float
        Time step size.
    u_L : float, optional
        Dirichlet boundary value at the left boundary. Default is 0.
    u_R : float, optional
        Dirichlet boundary value at the right boundary. Default is 0.
    sigma : float, optional
        Standard deviation of the stochastic forcing (noise intensity). Default is 1.

    Returns
    -------
    U : ndarray
        Numerical solution of the equation over space and time.
    """
    ds = s_inner[1] - s_inner[0]  # ∆s
    u0 = -torch.sin(np.pi * s_inner)  # 初始条件 u(0,s) = -sin(pi * s)
    u = u0.clone()
    g = 5 * (s_inner ** 2 - 1)

    alpha = D / ds ** 2
    beta = kappa / (2 * ds)
    diag = (1 / dt + 2 * alpha + 2 * beta) * torch.ones(Ns)
    upper = (-alpha) * torch.ones(Ns - 1)
    lower = (-alpha - 2 * beta) * torch.ones(Ns - 1)

    A = torch.diag(diag) + torch.diag(upper, 1) + torch.diag(lower, -1)
    A_inv = torch.inverse(A)
    U_all = [u0.clone()]
    for k in range(1, K):
        noise = torch.randn(Ns) * dt ** 0.5  # ΔW^k ∼ N(0, dt)
        dd = sigma * noise / dt
        rhs = u / dt + g+dd
        rhs[0] += (alpha + 2 * beta) * u_L
        rhs[Ns - 1] += alpha * u_R
        u_new = A_inv @ rhs

        U_all.append(u_new.clone())
        u = u_new
    U = torch.stack(U_all)

    return U


D = 0.1      # Diffusion coefficient
kappa = 1    # Convection (advection) coefficient
sigma = 1    # Noise intensity

Ns = 100
T = 50
dt = 1 / T
s_min, s_max = -1.0, 1.0
s_grid = torch.linspace(s_min, s_max, Ns + 2)  # Including boundary points
s_inner = s_grid[1:-1]                         # Interior spatial points

N_train = 1000
N_test = 200

# Generate a sample trajectory
u = generate_trajectory(D, kappa, s_inner, Ns, T, dt=dt, sigma=sigma)

# Spatio-temporal visualization
plt.figure(figsize=(5, 5))
plt.imshow(u.T, extent=[0.0, T * dt, s_min, s_max])
plt.colorbar()
plt.show()

# Select representative time instants
time_points_to_plot = [int(T * 0.5), int(T * 0.75), T - 1]

fig, axes = plt.subplots(1, len(time_points_to_plot),
                         figsize=(15, 4), sharey=True)

for i, t_idx in enumerate(time_points_to_plot):
    axes[i].plot(
        s_inner.numpy(),
        u[t_idx].numpy(),
        'r-',
        linewidth=2,
        label='Ground Truth'
    )
    axes[i].set_title(f'State at t = {t_idx * dt:.2f}')
    axes[i].set_xlabel('Spatial coordinate s')
    if i == 0:
        axes[i].set_ylabel('u(t, s)')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()


K = 200; dim_u = 50
U_train, U_test = [], []

for _ in range(N_train):
    u = generate_trajectory(D, kappa, s_inner, Ns, T, dt=dt, sigma=sigma)
    if dim_u == 50:
        u = u[:, ::2]
    elif dim_u == 25:
        u = u[:, ::4]

    U_train.append(u)

for _ in range(N_test):
    u = generate_trajectory(D, kappa, s_inner, Ns, K, dt=dt, sigma=sigma)
    if dim_u == 50:
        u = u[:, ::2]
    elif dim_u == 25:
        u = u[:, ::4]
    U_test.append(u)
U_train = torch.stack(U_train)
u_2d = U_train.reshape(N_train, -1).numpy()

U_test = torch.stack(U_test)
U_test = U_test.reshape(N_test, -1).numpy()
df = pd.DataFrame(u_2d)
df.to_csv("Data/diffusion_equation/u_train_Ns={}_T={}_sigma={}.csv".format(dim_u, K, sigma), index=False)
df3 = pd.DataFrame(U_test)
df3.to_csv("Data/diffusion_equation/u_test_Ns={}_T={}_sigma={}.csv".format(dim_u, K, sigma), index=False)

