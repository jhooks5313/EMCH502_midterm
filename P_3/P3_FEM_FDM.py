"""
Midterm P_3 — EMCH 502
Author: JP Hooks
Wave Equation on LxW domain - FEM and FDM
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# parameters
L, W    = 1.0, 1.0
alpha_v = 1.0
T_end   = 1.0
n_t     = 100
dt      = T_end / n_t

Nx, Ny  = 15, 15
n_nodes = Nx * Ny
n_elem  = (Nx - 1) * (Ny - 1)

print("="*60)
print(f"Mesh: {Nx}x{Ny} nodes, {n_elem} Q4 elems | dt = {dt:.4e} s | {n_t} steps")
print("="*60)

# structured Q4 mesh
x1_arr = np.linspace(0, L, Nx)
x2_arr = np.linspace(0, W, Ny)

coords = np.zeros((n_nodes, 2))
for j in range(Ny):
    for i in range(Nx):
        coords[i + j * Nx] = [x1_arr[i], x2_arr[j]]

elems = np.zeros((n_elem, 4), dtype=int)
for ej in range(Ny - 1):
    for ei in range(Nx - 1):
        e  = ei + ej * (Nx - 1)
        n0 = ei + ej * Nx
        elems[e] = [n0, n0 + 1, n0 + 1 + Nx, n0 + Nx]

# FEM assembly
gp = 1.0 / np.sqrt(3)
GP = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
GW = [1.0, 1.0, 1.0, 1.0]

def shape_Q4(xi, eta):
    N      = np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                       (1+xi)*(1+eta), (1-xi)*(1+eta)]) / 4.0
    dNdxi  = np.array([-(1-eta),  (1-eta),  (1+eta), -(1+eta)]) / 4.0
    dNdeta = np.array([-(1-xi),  -(1+xi),   (1+xi),   (1-xi)]) / 4.0
    return N, dNdxi, dNdeta

K_gl = np.zeros((n_nodes, n_nodes))
M_gl = np.zeros((n_nodes, n_nodes))

for e in range(n_elem):
    idx = elems[e]
    xe, ye = coords[idx, 0], coords[idx, 1]
    Ke = np.zeros((4, 4))
    Me = np.zeros((4, 4))
    for (xi_g, eta_g), wg in zip(GP, GW):
        N, dNdxi, dNdeta = shape_Q4(xi_g, eta_g)
        J    = np.array([[dNdxi @ xe,  dNdxi @ ye],
                         [dNdeta @ xe, dNdeta @ ye]])
        detJ = np.linalg.det(J)
        Jinv = np.linalg.inv(J)
        dNdx = Jinv[0, 0] * dNdxi + Jinv[0, 1] * dNdeta
        dNdy = Jinv[1, 0] * dNdxi + Jinv[1, 1] * dNdeta
        B    = np.vstack([dNdx, dNdy])
        Ke  += (B.T @ B) * detJ * wg
        Me  += np.outer(N, N) * detJ * wg
    for a in range(4):
        for b in range(4):
            K_gl[idx[a], idx[b]] += Ke[a, b]
            M_gl[idx[a], idx[b]] += Me[a, b]

# BCs: wave: x1=0 => 10 mm, x1=L => 1 mm, x2=0 => 0, x2=W => 0
bc = {}
for n in range(n_nodes):
    x1, x2 = coords[n]
    is_bc, val = False, 0.0
    if abs(x1) < 1e-12:
        val, is_bc = 10.0, True
    elif abs(x1 - L) < 1e-12:
        val, is_bc = 1.0, True
    if abs(x2) < 1e-12:
        val, is_bc = 0.0, True
    elif abs(x2 - W) < 1e-12:
        val, is_bc = 0.0, True
    if is_bc:
        bc[n] = val

# partition into interior/boundary DOFs
bc_ids  = sorted(bc.keys())
int_ids = sorted(set(range(n_nodes)) - set(bc_ids))
ii = np.ix_(int_ids, int_ids)
ib = np.ix_(int_ids, bc_ids)
U_b = np.array([bc[d] for d in bc_ids])

Kii = K_gl[ii]
Mii = M_gl[ii]
Fbc = -K_gl[ib] @ U_b

Me = alpha_v * Mii
Ke = Kii
Ce = np.zeros_like(Ke)
nd_i = len(int_ids)

# eigenvalue analysis & stability check
eigv   = eigvalsh(Ke, Me)
om_max = np.sqrt(np.max(np.abs(eigv)))
dt_cr  = 2.0 / om_max
print(f"Interior DOFs: {nd_i} | omega_max = {om_max:.4f} | dt_crit = {dt_cr:.6e}")

# explicit central difference
def explicit_cd(M, C, K, F, x0, v0, dt, nt):
    nd = M.shape[0]; n = nt + 1
    x, v, a = [np.zeros((n, nd)) for _ in range(3)]
    a[0] = np.linalg.solve(M, F - C @ v0 - K @ x0)
    x[0], v[0] = x0.copy(), v0.copy()
    x_m1 = x0 - dt * v0 + 0.5 * dt**2 * a[0]
    Me_inv = np.linalg.inv(M / dt**2 + C / (2 * dt))
    A1 = 2 * M / dt**2 - K
    A2 = M / dt**2 - C / (2 * dt)
    for i in range(n - 1):
        x_ip1 = Me_inv @ (F + A1 @ x[i] - A2 @ x_m1)
        v[i]  = (x_ip1 - x_m1) / (2 * dt)
        a[i]  = (x_ip1 - 2 * x[i] + x_m1) / dt**2
        x_m1  = x[i].copy()
        x[i + 1] = x_ip1
    v[-1] = (x[-1] - x[-2]) / dt
    a[-1] = np.linalg.solve(M, F - C @ v[-1] - K @ x[-1])
    return x, v, a

# implicit wilson-theta 
def wilson_theta(M, C, K, F, x0, v0, dt, nt, theta=1.4):
    nd = M.shape[0]; n = nt + 1
    x, v, a = [np.zeros((n, nd)) for _ in range(3)]
    x[0], v[0] = x0.copy(), v0.copy()
    a[0] = np.linalg.solve(M, F - C @ v0 - K @ x0)
    tau = theta * dt
    K_e = (6 / tau**2) * M + (3 / tau) * C + K
    P1 = (6 / tau**2) * M + (3 / tau) * C
    P2 = (6 / tau) * M + 2 * C
    P3 = 2 * M + (tau / 2) * C
    for i in range(n - 1):
        Fe = F + P1 @ x[i] + P2 @ v[i] + P3 @ a[i]
        xt = np.linalg.solve(K_e, Fe)
        at = (6 / tau**2) * (xt - x[i]) - (6 / tau) * v[i] - 2 * a[i]
        a[i+1] = a[i] + (1 / theta) * (at - a[i])
        v[i+1] = v[i] + (dt / 2) * (a[i+1] + a[i])
        x[i+1] = x[i] + dt * v[i] + (dt**2 / 6) * (a[i+1] + 2 * a[i])
    return x, v, a

# RK4
def rk4_mdof(M, C, K, F, x0, v0, dt, nt):
    nd = M.shape[0]; n = nt + 1
    Mi = np.linalg.inv(M)
    q  = np.zeros((n, 2 * nd))
    q[0, :nd], q[0, nd:] = v0.copy(), x0.copy()
    def dq(qs):
        vel, dis = qs[:nd], qs[nd:]
        return np.concatenate([Mi @ (F - C @ vel - K @ dis), vel])
    for i in range(n - 1):
        qi = q[i]
        k1 = dq(qi)
        k2 = dq(qi + dt / 2 * k1)
        k3 = dq(qi + dt / 2 * k2)
        k4 = dq(qi + dt * k3)
        q[i+1] = qi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return q[:, nd:], q[:, :nd], None

# reconstruct field from interior and boundary
def recon(u_int, int_ids, bc_ids, U_b):
    nt = u_int.shape[0]
    u  = np.zeros((nt, n_nodes))
    for k in range(nt):
        u[k, int_ids] = u_int[k]
        u[k, bc_ids]  = U_b
    return u

# run
t_arr = np.linspace(0, T_end, n_t + 1)
u0i = np.zeros(nd_i)
v0i = np.zeros(nd_i)
theta_run = 1.4

xw_wt, _, _ = wilson_theta(Me, Ce, Ke, Fbc, u0i, v0i, dt, n_t, theta_run)
xw_rk, _, _ = rk4_mdof(Me, Ce, Ke, Fbc, u0i, v0i, dt, n_t)

stable_cd = dt <= dt_cr
if stable_cd:
    xw_cd, _, _ = explicit_cd(Me, Ce, Ke, Fbc, u0i, v0i, dt, n_t)
else:
    xw_cd = None
    print("!! dt > dt_crit — Explicit CD skipped (unstable)")

uw_wt = recon(xw_wt, int_ids, bc_ids, U_b)
uw_rk = recon(xw_rk, int_ids, bc_ids, U_b)
if stable_cd:
    uw_cd = recon(xw_cd, int_ids, bc_ids, U_b)

# plot contour snapshots
snap_idx = [0, 25, 50, 75, 100]
X1g, X2g = np.meshgrid(x1_arr, x2_arr)

fig, axes = plt.subplots(1, len(snap_idx), figsize=(22, 4),
                         constrained_layout=True)
for col, si in enumerate(snap_idx):
    Z = uw_rk[si].reshape(Ny, Nx)
    cf = axes[col].contourf(X1g, X2g, Z, levels=30, cmap='viridis')
    plt.colorbar(cf, ax=axes[col], shrink=0.85)
    axes[col].set_title(f't = {t_arr[si]:.2f} s')
    axes[col].set_xlabel('x1')
    if col == 0:
        axes[col].set_ylabel('x2')
fig.suptitle('Wave Equation — Temporal Evolution of u(x1, x2, t)',
             fontweight='bold', fontsize=14)
plt.savefig('P3_spatial_evolution.png', dpi=200)
plt.show()

# plot method comparison at centre node
center = (Nx // 2) + (Ny // 2) * Nx
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(t_arr, uw_rk[:, center],       label='RK4',                           linewidth=1.4)
ax2.plot(t_arr, uw_wt[:, center], '--',  label=f'Wilson-θ (θ={theta_run})',     linewidth=1.0)
if stable_cd:
    ax2.plot(t_arr, uw_cd[:, center], ':', label='Explicit CD', linewidth=0.9)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('u (mm)')
ax2.set_title('Wave Equation — u at centre node, FDM method comparison',
              fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('P3_wave_method_comparison.png', dpi=200)
plt.show()

# plot 3d surface at t = 0, 0.5, 1 sec
surf_times = [0.0, 0.5, 1.0]
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 6),
                            subplot_kw={'projection': '3d'})
for ax, ts in zip(axes3, surf_times):
    si = int(round(ts / dt))
    Z = uw_rk[si].reshape(Ny, Nx)
    ax.plot_surface(X1g, X2g, Z, cmap='viridis',
                    edgecolor='k', linewidth=0.2, alpha=0.9)
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('u (mm)')
    ax.set_title(f't = {ts:.1f} s')
fig3.suptitle('Wave Equation — 3D Surface Evolution', fontweight='bold')
plt.tight_layout()
plt.savefig('P3_surface_final.png', dpi=200)
plt.show()

