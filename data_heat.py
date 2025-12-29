
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class NonlinearHeat1D_SDE:
    """
    One-dimensional nonlinear heat-conduction SPDE with Neumann boundary conditions:
        u_t = ∂x( k(u) ∂x u ) + f(x, t) + q · ξ(x, t)

    Numerical scheme:
        Backward Euler time discretization combined with Picard linearization.
        The stochastic forcing is added once per time step after convergence.

    Noise model:
        ξ(x, t) is spatio-temporally independent Gaussian white noise.
        In the discrete implementation, the noise term is approximated as
        sqrt(dt) · q · N(0, I).
    """

    def __init__(self,
                 L=1.0,
                 M=100,
                 k_func=lambda u: 0.1 + 0.05*u**2,
                 f_func=lambda x,t: np.exp(-t)*np.sin(2*np.pi*x),
                 dt=1e-2,
                 T_final=1.0,
                 q=0.0,
                 max_iter=20,
                 tol=1e-8,
                 seed=None):
        self.L = float(L)
        self.M = int(M)
        self.k_func = k_func
        self.f_func = f_func
        self.dt = float(dt)
        self.T_final = float(T_final)
        self.Nt = int(np.round(self.T_final / self.dt))
        self.q = float(q)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(seed)

        # 网格
        self.x = np.linspace(0.0, self.L, self.M )
        self.dx = self.x[1] - self.x[0]
        # 预分配临时数组
        self._u_new = np.empty(self.M )
        self._u_iter = np.empty(self.M )

    # --- 三对角 Thomas 求解器（输入压缩三对角 A 的三行形式与 rhs） ---
    # --- Thomas solver for tridiagonal systems (inputs are the three diagonals of A and the RHS) ---
    @staticmethod
    def _thomas_solve_tridiag(a_sub, b_main, c_sup, rhs):
        """
        Solve a tridiagonal linear system:
            a_sub[i] * x[i-1] + b_main[i] * x[i] + c_sup[i] * x[i+1] = rhs[i]

        where a_sub[0] is ignored and c_sup[-1] is ignored.

        Parameters
        ----------
        a_sub : ndarray
            Sub-diagonal entries of the tridiagonal matrix.
        b_main : ndarray
            Main diagonal entries of the tridiagonal matrix.
        c_sup : ndarray
            Super-diagonal entries of the tridiagonal matrix.
        rhs : ndarray
            Right-hand-side vector.

        Returns
        -------
        x : ndarray
            Solution vector of the same length as rhs.
        """
        n = rhs.size
        cp = np.zeros(n)
        dp = np.zeros(n)
        x = np.zeros(n)

        cp[0] = c_sup[0] / b_main[0]
        dp[0] = rhs[0] / b_main[0]
        for i in range(1, n):
            denom = b_main[i] - a_sub[i]*cp[i-1]
            cp[i] = c_sup[i] / denom if i < n-1 else 0.0
            dp[i] = (rhs[i] - a_sub[i]*dp[i-1]) / denom

        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i]*x[i+1]
        return x

    def _one_step_picard(self, u_old, t_next):
        """
        Given the solution u_old at the previous time step, perform a single
        Backward Euler time step combined with Picard iteration at time t_next
        to compute the deterministic solution u^{n+1} (without stochastic noise).
        """
        M = self.M
        dx2 = self.dx ** 2
        dt = self.dt
        f = self.f_func
        k_func = self.k_func

        u_iter = self._u_iter
        u_new = self._u_new
        u_iter[:] = u_old

        for _ in range(self.max_iter):

            # u_iter shape: [M+1] → k_half shape: [M]
            k_half = k_func(0.5*(u_iter[:-1] + u_iter[1:]))
            a = -dt/dx2 * k_half[:-1]
            c = -dt/dx2 * k_half[1:]
            b = 1.0 - (a + c)

            rhs = u_old[1:-1] + dt * f(self.x[1:-1], t_next)
            a_sub = np.zeros_like(rhs)
            b_main = np.zeros_like(rhs)
            c_sup = np.zeros_like(rhs)
            a_sub[1:] = a[1:]
            b_main[:] = b
            c_sup[:-1] = c[:-1]

            u_inner = self._thomas_solve_tridiag(a_sub, b_main, c_sup, rhs)
            u_new[1:-1] = u_inner
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
            if np.linalg.norm(u_new - u_iter, np.inf) < self.tol:
                break
            u_iter[:] = u_new

        return u_new.copy()

    def simulate(self, N=1, u0=None, return_time=False):

        N = int(N)
        T = self.Nt
        dim_u = self.M

        if u0 is None:
            u0 = np.sin(np.pi * self.x)
        else:
            u0 = np.asarray(u0, dtype=float)
            assert u0.shape == (dim_u,)

        U_paths = np.zeros((N, T, dim_u), dtype=float)
        t_vec = np.linspace(0.0, self.Nt * self.dt, T)

        for n_traj in range(N):
            if n_traj%10==0:
                print(n_traj)
            u = u0.copy()
            U_paths[n_traj, 0] = u

            for n in range(self.Nt-1):
                t_next = (n + 1) * self.dt
                u_det = self._one_step_picard(u, t_next)
                if self.q > 0.0:
                    eta = np.sqrt(self.dt) * self.q * self.rng.standard_normal(size=dim_u)
                    u_det[1:-1] += eta[1:-1]
                    u_det[0]  = u_det[1]
                    u_det[-1] = u_det[-2]

                u = u_det
                U_paths[n_traj, n+1] = u

        if return_time:
            return U_paths, t_vec, self.x
        else:
            return U_paths
if __name__ == "__main__":
    Ns = 200
    # T = 100
    q = 1
    x = np.linspace(0, 1, Ns)
    solver = NonlinearHeat1D_SDE(
        L=1.0, M=Ns,
        k_func=lambda u: 0.1 + 0.05 * u**2,
        f_func=lambda x, t: 0,
        dt=1e-2, T_final=1.5,
        q=q,
        max_iter=20, tol=1e-8,
        seed=42
    )
    N_train = 1000
    dim_u = int(200 / 4)

    U_train= solver.simulate(u0= 2*np.sin(np.pi * x), N=N_train)  # U1.shape == (N, Nt, M)
    U_train = U_train[:, :, ::4]
    u_2d = U_train.reshape(N_train, -1)

    df = pd.DataFrame(u_2d)
    df.to_csv("Data/Heat_equation_2/u_train_Ns={}_sigma={}.csv".format(dim_u, q), index=False)

    N_test = 20
    U_test = solver.simulate(u0=2*np.sin(np.pi * x), N=N_test)  # U1.shape == (N, Nt, M)
    U_test=U_test[:, :, ::4]
    print(U_test.shape)
    U_test = U_test.reshape(N_test, -1)
    dim_u = int(200 / 4)

    df3 = pd.DataFrame(U_test)
    df3.to_csv("Data/Heat_equation_2/u_test_Ns={}_sigma={}.csv".format(dim_u, q), index=False)

    solver1 = NonlinearHeat1D_SDE(
        L=1.0, M=Ns,
        k_func=lambda u: 0.1 + 0.05 * u ** 2,
        f_func=lambda x, t: 0,
        dt=1e-2, T_final=1.5,
        q=0,
        max_iter=20, tol=1e-8,
        seed=42
    )
    U_plot = solver1.simulate(u0=2 * np.sin(np.pi * x), N=1)  # U1.shape == (N, Nt, M)
    N_time, N_space = U_plot[0].shape
    x = np.linspace(0, 1, N_space)
    t = np.linspace(0, 1.5, N_time)

    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')

    plt.figure(figsize=(8, 5))
    im = plt.pcolormesh(T_grid, X_grid, U_plot[0], shading='auto', cmap='plasma')
    plt.ylabel("Space x", fontsize=12)
    plt.xlabel("Time t", fontsize=12)
    plt.title("Spatiotemporal evolution of u(x,t) under Neumann BC", fontsize=14)
    plt.colorbar(im, label="Temperature u(x,t)")
    plt.tight_layout()
    plt.show()
