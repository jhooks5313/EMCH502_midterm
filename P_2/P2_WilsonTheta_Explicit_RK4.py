"""
Midterm P_2 - EMCH 502
Author: JP Hooks
Explicit CD, Wilson-Theta (implicit), and RK4 for 3-DOF system
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

NDOF = 3

M = np.diag([10.0, 20.0, 30.0])
C = np.array([[0.1, 0.2, 0.1],
              [0.2, 0.1, 0.1],
              [0.1, 0.1, 0.1]])
K = 1e4 * np.array([[ 40.0, -20.0, -10.0],
                     [-20.0,  30.0, -10.0],
                     [-10.0, -10.0,  10.0]])

def f_ext(t):
    f_val = 12.0 * np.sin(30.0 * t) + 5.0 * np.cos(15.0 * t)
    return np.array([f_val, f_val, f_val])

x0 = np.zeros(NDOF)
v0 = np.zeros(NDOF)

# eigenvalue analysis & time-step selection
eigvals = eigvalsh(K, M)
omega_n = np.sqrt(np.abs(eigvals))
omega_max = np.max(omega_n)

print("="*60)
print("Eigenvalues (omega^2):", eigvals)
if np.any(eigvals < 0):
    print("*** WARNING: Negative eigenvalue detected — system may be unstable. ***")
print(f"|omega| (rad/s): {np.sort(omega_n)}")

dt_crit = 2.0 / omega_max
dt = 0.60 * dt_crit
T_end = 0.5
n_steps = int(np.ceil(T_end / dt)) + 1
t = np.linspace(0.0, T_end, n_steps)
dt = t[1] - t[0]
print(f"dt = {dt:.6e} s  |  {n_steps} steps  |  T_end = {T_end} s")
print("="*60)

F = np.zeros((n_steps, NDOF))
for i in range(n_steps):
    F[i, :] = f_ext(t[i])

# explicit central difference
def explicit_cd(M, C, K, F, x0, v0, t):
    dt, n, nd = t[1]-t[0], len(t), M.shape[0]
    x, v, a = np.zeros((n,nd)), np.zeros((n,nd)), np.zeros((n,nd))

    a[0] = np.linalg.solve(M, F[0] - C@v0 - K@x0)
    x[0], v[0] = x0.copy(), v0.copy()
    x_m1 = x0 - dt*v0 + 0.5*dt**2*a[0]

    Me_inv = np.linalg.inv(M/dt**2 + C/(2*dt))
    A1 = 2*M/dt**2 - K
    A2 = M/dt**2 - C/(2*dt)

    for i in range(n-1):
        x_ip1 = Me_inv @ (F[i] + A1@x[i] - A2@x_m1)
        v[i] = (x_ip1 - x_m1)/(2*dt)
        a[i] = (x_ip1 - 2*x[i] + x_m1)/dt**2
        x_m1 = x[i].copy()
        x[i+1] = x_ip1

    v[-1] = (x[-1]-x[-2])/dt
    a[-1] = np.linalg.solve(M, F[-1]-C@v[-1]-K@x[-1])
    return x, v, a

# wilson-theta (implicit)
def wilson_theta(M, C, K, F_func, x0, v0, t, theta=1.4):
    dt, n, nd = t[1]-t[0], len(t), M.shape[0]
    x, v, a = np.zeros((n,nd)), np.zeros((n,nd)), np.zeros((n,nd))

    x[0], v[0] = x0.copy(), v0.copy()
    a[0] = np.linalg.solve(M, F_func(t[0]) - C@v0 - K@x0)
    tau = theta*dt

    Ke = (6/tau**2)*M + (3/tau)*C + K
    P1 = (6/tau**2)*M + (3/tau)*C
    P2 = (6/tau)*M + 2*C
    P3 = 2*M + (tau/2)*C

    for i in range(n-1):
        Fe = F_func(t[i]+tau) + P1@x[i] + P2@v[i] + P3@a[i]
        x_theta = np.linalg.solve(Ke, Fe)
        a_theta = (6/tau**2)*(x_theta-x[i]) - (6/tau)*v[i] - 2*a[i]

        a[i+1] = a[i] + (1/theta)*(a_theta - a[i])
        v[i+1] = v[i] + (dt/2)*(a[i+1] + a[i])
        x[i+1] = x[i] + dt*v[i] + (dt**2/6)*(a[i+1] + 2*a[i])

    return x, v, a

def rk4_mdof(M, C, K, F_func, x0, v0, t):
    dt, n, nd = t[1]-t[0], len(t), M.shape[0]
    M_inv = np.linalg.inv(M)
    q = np.zeros((n, 2*nd))
    q[0, :nd], q[0, nd:] = v0.copy(), x0.copy()

    def qdot(qs, ti):
        vel, disp = qs[:nd], qs[nd:]
        acc = M_inv @ (F_func(ti) - C@vel - K@disp)
        return np.concatenate([acc, vel])

    for i in range(n-1):
        ti, qi = t[i], q[i]
        k1 = qdot(qi,              ti)
        k2 = qdot(qi+dt/2*k1,     ti+dt/2)
        k3 = qdot(qi+dt/2*k2,     ti+dt/2)
        k4 = qdot(qi+dt*k3,       ti+dt)
        q[i+1] = qi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return q[:, nd:], q[:, :nd], None   

# run
theta_run = 1.4
x_cd, _, _ = explicit_cd(M, C, K, F, x0, v0, t)
x_wt, _, _ = wilson_theta(M, C, K, f_ext, x0, v0, t, theta=theta_run)
x_rk, _, _ = rk4_mdof(M, C, K, f_ext, x0, v0, t)

# plot
t_zoom = 0.25
iz = np.searchsorted(t, t_zoom)
dof_labels = ['DOF 1', 'DOF 2', 'DOF 3']
colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(figsize=(14, 6))
for j in range(NDOF):
    ax.plot(t[:iz], x_rk[:iz, j], color=colors[j],
            label=f'RK4 – {dof_labels[j]}', linewidth=1.2)
    ax.plot(t[:iz], x_wt[:iz, j], color=colors[j], linestyle='--',
            label=f'Wilson-θ (θ = {theta_run}) – {dof_labels[j]}', linewidth=1.0)
    ax.plot(t[:iz], x_cd[:iz, j], color=colors[j], linestyle=':',
            label=f'Explicit CD – {dof_labels[j]}', linewidth=0.8)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Displacement x (m)')
ax.set_title('Displacement vs Time — Method Comparison', fontweight='bold')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f'P2_displacement_comparison_theta-{theta_run}.png', dpi=200)
plt.show()
