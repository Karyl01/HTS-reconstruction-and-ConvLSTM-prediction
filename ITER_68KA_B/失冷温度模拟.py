import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, solve_ivp

# ==========================================
# 1. 各分段几何与物理参数定义 (论文 3.1 ~ 3.3.3)
# ==========================================
I_current = 68000.0  # 电流 68 kA

# 段 0: 高温超导段 (0.0m - 0.6m)
L0 = 0.6
A_steel = 100.0 * 1e-4
A_AgAu = 9.04 * 1e-4
A_hts = A_steel + A_AgAu

# 段 1: 过渡段 (0.6m - 0.75m)
L1 = 0.15
J_trans = 4.25 * 1e6
A_trans = I_current / J_trans
hp_trans = 200.0

# 段 2: 换热器段 (0.75m - 1.60m)
L2 = 0.85
A_hx = 63.6 * 1e-4
hp_hx = 20000.0
m_dot = 4.015 * 1e-3
Cp_he = 5210.0

# 段 3: 水冷端 (1.60m - 1.80m)
L3 = 0.20
J_water = 2.5 * 1e6
A_water = I_current / J_water

x0_start, x1_start, x2_start, x3_start = 0.0, 0.6, 0.75, 1.60

# ==========================================
# 2. 物性拟合函数 (带数值安全保护)
# ==========================================
def lambda_AgAu(T):
    T_c = np.clip(T, 3.0, 600.0)
    return -1.7465e-6 * T_c ** 4 + 4.9848e-4 * T_c ** 3 - 5.4966e-2 * T_c ** 2 + 3.5040 * T_c - 7.4917

def lambda_steel(T):
    T_c = np.clip(T, 3.0, 600.0)
    return -1.8436e-5 * T_c ** 3 + 1.8562e-3 * T_c ** 2 + 7.4769e-2 * T_c - 1.0323e-1

def lambda_hts(T):
    return (A_steel / A_hts) * lambda_steel(T) + (A_AgAu / A_hts) * lambda_AgAu(T)

def lambda_copper(T):
    T_c = np.clip(T, 3.0, 600.0)
    denom = np.maximum(np.abs(1.21e-15 * (T_c ** 7) - 3.763e-4), 1e-6)
    return (1.0 / denom) + 380.0

def rho_copper(T):
    T_c = np.clip(T, 3.0, 600.0)
    return np.maximum(6.75e-11 * T_c - 2.45e-9, 1.0e-10)

def C_copper(T):
    """紫铜体积热容 C(T) J/(m^3*K)"""
    T_c = np.clip(T, 3.0, 600.0)
    return np.where(T_c < 50.0, 1.0e3 * (T_c ** 2.5) + 10.0, 3.4e6 * (1.0 - np.exp(-T_c / 80.0)))

def C_hts(T):
    """HTS复合段体积热容 C(T) J/(m^3*K)"""
    T_c = np.clip(T, 3.0, 600.0)
    return np.where(T_c < 50.0, 800.0 * (T_c ** 2.2) + 10.0, 3.8e6 * (1.0 - np.exp(-T_c / 90.0)))

# ==========================================
# 3. 求解正常运行稳态 (你的原始 solve_bvp 逻辑)
# ==========================================
def odes_multi_region(x, Y):
    T0, dT0 = Y[0], Y[1]
    lam0 = lambda_hts(T0)
    d2T0 = np.zeros_like(T0)

    T1, dT1, theta1 = Y[2], Y[3], Y[4]
    lam1 = lambda_copper(T1)
    rho1 = rho_copper(T1)
    joule1 = (rho1 * I_current ** 2) / (A_trans ** 2)
    conv1 = (hp_trans / A_trans) * (T1 - theta1)
    d2T1 = (conv1 - joule1) / lam1
    dtheta1 = (hp_trans * (T1 - theta1)) / (m_dot * Cp_he)

    T2, dT2, theta2 = Y[5], Y[6], Y[7]
    lam2 = lambda_copper(T2)
    rho2 = rho_copper(T2)
    joule2 = (rho2 * I_current ** 2) / (A_hx ** 2)
    conv2 = (hp_hx / A_hx) * (T2 - theta2)
    d2T2 = (conv2 - joule2) / lam2
    dtheta2 = (hp_hx * (T2 - theta2)) / (m_dot * Cp_he)

    T3, dT3 = Y[8], Y[9]
    lam3 = lambda_copper(T3)
    rho3 = rho_copper(T3)
    joule3 = (rho3 * I_current ** 2) / (A_water ** 2)
    d2T3 = -joule3 / lam3
    dtheta3 = np.zeros_like(T3)

    return np.vstack((
        L0 * dT0, L0 * d2T0,
        L1 * dT1, L1 * d2T1, L1 * dtheta1,
        L2 * dT2, L2 * d2T2, L2 * dtheta2,
        L3 * dT3, L3 * d2T3, L3 * dtheta3
    ))

def bc_multi_region(ya, yb):
    res = []
    res.append(ya[0] - 4.5)
    res.append(yb[8] - 300.0)
    res.append(yb[0] - ya[2])
    Q_hts = lambda_hts(yb[0]) * A_hts * yb[1]
    Q_trans = lambda_copper(ya[2]) * A_trans * ya[3]
    res.append(Q_hts - Q_trans)
    res.append(ya[4] - 50.0)
    res.append(yb[2] - ya[5])
    res.append(yb[4] - ya[7])
    Q_trans_end = lambda_copper(yb[2]) * A_trans * yb[3]
    Q_hx_start = lambda_copper(ya[5]) * A_hx * ya[6]
    res.append(Q_trans_end - Q_hx_start)
    res.append(yb[5] - ya[8])
    Q_hx_end = lambda_copper(yb[5]) * A_hx * yb[6]
    Q_water_start = lambda_copper(ya[8]) * A_water * ya[9]
    res.append(Q_hx_end - Q_water_start)
    res.append(yb[7] - ya[10])
    return np.array(res)

t_mesh = np.linspace(0, 1, 80)
y_guess = np.zeros((11, len(t_mesh)))
y_guess[0] = np.linspace(4.5, 65.0, len(t_mesh))
y_guess[1] = 100.0
y_guess[2] = np.linspace(65.0, 80.0, len(t_mesh))
y_guess[3] = 100.0
y_guess[4] = np.linspace(50.0, 70.0, len(t_mesh))
y_guess[5] = np.linspace(80.0, 280.0, len(t_mesh))
y_guess[6] = 200.0
y_guess[7] = np.linspace(70.0, 275.0, len(t_mesh))
y_guess[8] = np.linspace(280.0, 300.0, len(t_mesh))
y_guess[9] = 100.0
y_guess[10] = 275.0

sol_steady = solve_bvp(odes_multi_region, bc_multi_region, t_mesh, y_guess, tol=1e-3)

if sol_steady.status != 0:
    raise RuntimeError("稳态求解失败，请检查参数！")

# ==========================================
# 4. 构建网格与离散物性 (有限体积/差分法准备)
# ==========================================
# 构建 150 个节点的空间网格
N_x = 150
x_grid = np.linspace(0.0, 1.80, N_x)
dx = x_grid[1] - x_grid[0]

# 拼接稳态温度作为初始条件 T_init
x_r0 = x0_start + sol_steady.x * L0
x_r1 = x1_start + sol_steady.x * L1
x_r2 = x2_start + sol_steady.x * L2
x_r3 = x3_start + sol_steady.x * L3

x_raw = np.concatenate([x_r0, x_r1, x_r2, x_r3])
T_raw = np.concatenate([sol_steady.y[0], sol_steady.y[2], sol_steady.y[5], sol_steady.y[8]])

# 坐标排序与唯一化
sort_idx = np.argsort(x_raw)
x_sorted, unique_idx = np.unique(x_raw[sort_idx], return_index=True)
T_sorted = T_raw[sort_idx][unique_idx]

# 正确插值得到失冷前的 0s 初始分布
T_init = np.interp(x_grid, x_sorted, T_sorted)

# 为每个网格赋值截面积 A 和材料类型
A_grid = np.zeros(N_x)
is_hts = np.zeros(N_x, dtype=bool)

for i, x in enumerate(x_grid):
    if x <= 0.60:
        A_grid[i] = A_hts
        is_hts[i] = True
    elif x <= 0.75:
        A_grid[i] = A_trans
        is_hts[i] = False
    elif x <= 1.60:
        A_grid[i] = A_hx
        is_hts[i] = False
    else:
        A_grid[i] = A_water
        is_hts[i] = False

# ==========================================
# 5. 第二阶段：失冷瞬态 PDE 描述
# ==========================================
def pde_loss_of_flow(tau, T):
    dT_dt = np.zeros(N_x)

    # 边界条件：两端定温
    dT_dt[0] = 0.0    # x = 0m (4.5K)
    dT_dt[-1] = 0.0   # x = 1.8m (300K)

    # 计算各节点的导热率 lambda
    lam = np.zeros(N_x)
    for i in range(N_x):
        if is_hts[i]:
            lam[i] = lambda_hts(T[i])
        else:
            lam[i] = lambda_copper(T[i])

    # 内部节点利用热流量守恒 (Control Volume Method)
    for i in range(1, N_x - 1):
        # 界面热导率 (调和平均)
        lam_left = 2.0 * lam[i-1] * lam[i] / (lam[i-1] + lam[i] + 1e-12)
        lam_right = 2.0 * lam[i] * lam[i+1] / (lam[i] + lam[i+1] + 1e-12)

        # 控制体积两侧截面积
        A_left = 0.5 * (A_grid[i-1] + A_grid[i])
        A_right = 0.5 * (A_grid[i] + A_grid[i+1])

        # 热传导项:  (Q_in - Q_out) / V
        Q_left = lam_left * A_left * (T[i-1] - T[i]) / dx
        Q_right = lam_right * A_right * (T[i+1] - T[i]) / dx
        conduction = (Q_left + Q_right) / (A_grid[i] * dx)

        # 焦耳发热项: (I/A)^2 * rho
        if is_hts[i]:
            joule = 0.0  # HTS 正常状态无焦耳热
            C_i = C_hts(T[i])
        else:
            rho_i = rho_copper(T[i])
            joule = rho_i * (I_current / A_grid[i]) ** 2
            C_i = C_copper(T[i])

        dT_dt[i] = (conduction + joule) / C_i

    return dT_dt

# 指定失冷观察时间节点
target_times = [0, 15, 30, 60, 120, 240, 300, 315]

# 求解 PDE 方程
sol_transient = solve_ivp(
    fun=pde_loss_of_flow,
    t_span=(0, 315),
    y0=T_init,
    t_eval=target_times,
    method='Radau',      # 隐式 Radau 算法针对刚性导热问题极度稳定
    rtol=1e-4,
    atol=1e-4
)

# ==========================================
# 6. 绘制失冷温度演化曲线
# ==========================================
plt.figure(figsize=(10, 6))

colors = plt.cm.jet(np.linspace(0, 1, len(sol_transient.t)))

for i, t_val in enumerate(sol_transient.t):
    T_t = sol_transient.y[:, i]
    plt.plot(x_grid, T_t, label=f't = {t_val:.0f} s', color=colors[i], linewidth=2.0)

# 分界虚线
plt.axvline(x=0.6, color='k', linestyle=':', alpha=0.6)
plt.axvline(x=0.75, color='k', linestyle=':', alpha=0.6)
plt.axvline(x=1.60, color='k', linestyle=':', alpha=0.6)

# 400 K 最高允许温度红线
plt.axhline(y=400, color='r', linestyle='--', alpha=0.7, label='Max Allowable Temp (400 K)')

plt.text(0.3, 350, 'HTS Section', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.text(0.675, 350, 'Trans', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.text(1.15, 350, 'Heat Exchanger', fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
plt.text(1.7, 350, 'Water', fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.title('68kA HTS Current Lead LOFA Temperature Evolution', fontsize=12)
plt.xlabel('Length x (m)', fontsize=11)
plt.ylabel('Temperature T (K)', fontsize=11)
plt.grid(True, linestyle=':')
plt.xlim(0, 1.8)
plt.ylim(0, 420)
plt.legend(loc='upper left', ncol=2)
plt.tight_layout()
plt.show()