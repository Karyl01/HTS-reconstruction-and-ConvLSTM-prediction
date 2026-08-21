import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

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

# 各段起点绝对坐标
x0_start, x1_start, x2_start, x3_start = 0.0, 0.6, 0.75, 1.60


# ==========================================
# 2. 物性拟合函数 (论文公式 4, 5, 8, 9, 10)
# ==========================================
def lambda_AgAu(T):
    return -1.7465e-6 * T ** 4 + 4.9848e-4 * T ** 3 - 5.4966e-2 * T ** 2 + 3.5040 * T - 7.4917


def lambda_steel(T):
    return -1.8436e-5 * T ** 3 + 1.8562e-3 * T ** 2 + 7.4769e-2 * T - 1.0323e-1


def lambda_hts(T):
    return (A_steel / A_hts) * lambda_steel(T) + (A_AgAu / A_hts) * lambda_AgAu(T)


def lambda_copper(T):
    T_safe = np.maximum(T, 1.0)
    denom = 1.21e-15 * (T_safe ** 7) - 3.763e-4
    return (1.0 / denom) + 380.0


def rho_copper(T):
    return 6.75e-11 * T - 2.45e-9


# ==========================================
# 3. 归一化多区域 ODE 函数组 (Region ODEs)
# 每一个区域的自变量 x 均映射为 t in [0, 1]
# ==========================================
def odes_multi_region(x, Y):
    """
    Y 的维度为 (11, N):
    Region 0 (HTS, 2个状态):     Y[0]=T0, Y[1]=dT0/dx
    Region 1 (Trans, 3个状态):   Y[2]=T1, Y[3]=dT1/dx, Y[4]=theta1
    Region 2 (HX, 3个状态):      Y[5]=T2, Y[6]=dT2/dx, Y[7]=theta2
    Region 3 (Water, 3个状态):   Y[8]=T3, Y[9]=dT3/dx, Y[10]=theta3 (占位)
    """
    # Region 0: HTS
    T0, dT0 = Y[0], Y[1]
    lam0 = lambda_hts(T0)
    d2T0 = np.zeros_like(T0)

    # Region 1: Trans
    T1, dT1, theta1 = Y[2], Y[3], Y[4]
    lam1 = lambda_copper(T1)
    rho1 = rho_copper(T1)
    joule1 = (rho1 * I_current ** 2) / (A_trans ** 2)
    conv1 = (hp_trans / A_trans) * (T1 - theta1)
    d2T1 = (conv1 - joule1) / lam1
    dtheta1 = (hp_trans * (T1 - theta1)) / (m_dot * Cp_he)

    # Region 2: HX
    T2, dT2, theta2 = Y[5], Y[6], Y[7]
    lam2 = lambda_copper(T2)
    rho2 = rho_copper(T2)
    joule2 = (rho2 * I_current ** 2) / (A_hx ** 2)
    conv2 = (hp_hx / A_hx) * (T2 - theta2)
    d2T2 = (conv2 - joule2) / lam2
    dtheta2 = (hp_hx * (T2 - theta2)) / (m_dot * Cp_he)

    # Region 3: Water
    T3, dT3 = Y[8], Y[9]
    lam3 = lambda_copper(T3)
    rho3 = rho_copper(T3)
    joule3 = (rho3 * I_current ** 2) / (A_water ** 2)
    d2T3 = -joule3 / lam3
    dtheta3 = np.zeros_like(T3)  # 水冷段不考虑对流

    # 乘上各区域长度 L_i 完成导数映射 dt/dx -> d/dt
    return np.vstack((
        L0 * dT0, L0 * d2T0,  # R0
        L1 * dT1, L1 * d2T1, L1 * dtheta1,  # R1
        L2 * dT2, L2 * d2T2, L2 * dtheta2,  # R2
        L3 * dT3, L3 * d2T3, L3 * dtheta3  # R3
    ))


# ==========================================
# 4. 界面与边界条件 (刚好 11 个方程)
# ya 为 t=0 处的向量, yb 为 t=1 处的向量
# ==========================================
def bc_multi_region(ya, yb):
    res = []

    # [1-2] 外部固定温度边界
    res.append(ya[0] - 4.5)  # (1) x=0: HTS冷端 4.5 K
    res.append(yb[8] - 300.0)  # (2) x=1.8m: 水冷端热端 300 K

    # [3-5] 界面 1 (x = 0.6 m): HTS -> 过渡段
    res.append(yb[0] - ya[2])  # (3) 温度连续
    Q_hts = lambda_hts(yb[0]) * A_hts * yb[1]
    Q_trans = lambda_copper(ya[2]) * A_trans * ya[3]
    res.append(Q_hts - Q_trans)  # (4) 热流量连续
    res.append(ya[4] - 50.0)  # (5) 入口气氦温度 theta(0) = 50 K

    # [6-8] 界面 2 (x = 0.75 m): 过渡段 -> 换热器段
    res.append(yb[2] - ya[5])  # (6) 温度连续
    res.append(yb[4] - ya[7])  # (7) 氦气温度连续
    Q_trans_end = lambda_copper(yb[2]) * A_trans * yb[3]
    Q_hx_start = lambda_copper(ya[5]) * A_hx * ya[6]
    res.append(Q_trans_end - Q_hx_start)  # (8) 热流量连续

    # [9-11] 界面 3 (x = 1.60 m): 换热器段 -> 水冷段
    res.append(yb[5] - ya[8])  # (9) 温度连续
    Q_hx_end = lambda_copper(yb[5]) * A_hx * yb[6]
    Q_water_start = lambda_copper(ya[8]) * A_water * ya[9]
    res.append(Q_hx_end - Q_water_start)  # (10) 热流量连续
    res.append(yb[7] - ya[10])  # (11) 氦气出口绑定至水冷段占位变量 (补齐第11个条件)

    return np.array(res)


# ==========================================
# 5. 网格初始化与求解
# ==========================================
t_mesh = np.linspace(0, 1, 60)

# 初始化状态矩阵 11 x 60
y_guess = np.zeros((11, len(t_mesh)))

# 给定平滑合理的初猜分布
y_guess[0] = np.linspace(4.5, 65.0, len(t_mesh))  # T0
y_guess[1] = 100.0  # dT0/dx
y_guess[2] = np.linspace(65.0, 80.0, len(t_mesh))  # T1
y_guess[3] = 100.0  # dT1/dx
y_guess[4] = np.linspace(50.0, 70.0, len(t_mesh))  # theta1

y_guess[5] = np.linspace(80.0, 280.0, len(t_mesh))  # T2
y_guess[6] = 200.0  # dT2/dx
y_guess[7] = np.linspace(70.0, 275.0, len(t_mesh))  # theta2

y_guess[8] = np.linspace(280.0, 300.0, len(t_mesh))  # T3
y_guess[9] = 100.0  # dT3/dx
y_guess[10] = 275.0  # theta3 占位初猜

# 调用 solve_bvp 求解
sol = solve_bvp(
    fun=odes_multi_region,
    bc=bc_multi_region,
    x=t_mesh,
    y=y_guess,
    tol=1e-2,
    max_nodes=3000
)

# ==========================================
# 6. 还原物理坐标并绘图 (复现论文图 6)
# ==========================================
if sol.status == 0:
    print("求解成功！正在生成整体分布图...")

    t_sol = sol.x

    # 还原物理真实坐标 x (m)
    x_r0 = x0_start + t_sol * L0
    x_r1 = x1_start + t_sol * L1
    x_r2 = x2_start + t_sol * L2
    x_r3 = x3_start + t_sol * L3

    x_all = np.concatenate([x_r0, x_r1, x_r2, x_r3])
    T_all = np.concatenate([sol.y[0], sol.y[2], sol.y[5], sol.y[8]])
    theta_all = np.concatenate([np.nan * x_r0, sol.y[4], sol.y[7], np.nan * x_r3])

    plt.figure(figsize=(9, 5.5))
    plt.plot(x_all, T_all, 'r-', linewidth=2, label='HTS Lead Temp T(K)')
    plt.plot(x_all, theta_all, 'b--', linewidth=1.5, label=r'Helium Temp $\theta$(K)')

    # 分界虚线
    plt.axvline(x=0.6, color='k', linestyle=':', alpha=0.6)
    plt.axvline(x=0.75, color='k', linestyle=':', alpha=0.6)
    plt.axvline(x=1.60, color='k', linestyle=':', alpha=0.6)

    plt.text(0.3, 150, 'HTS Section', fontsize=10, ha='center')
    plt.text(0.675, 180, 'Trans', fontsize=9, ha='center')
    plt.text(1.15, 200, 'Heat Exchanger', fontsize=10, ha='center')
    plt.text(1.7, 230, 'Water-cooled', fontsize=9, ha='center')

    plt.title('68kA HTS Current Lead Temperature Profile (Paper Fig. 6)', fontsize=11)
    plt.xlabel('Length L (m)', fontsize=11)
    plt.ylabel('Temperature (K)', fontsize=11)
    plt.grid(True, linestyle=':')
    plt.xlim(0, 1.8)
    plt.ylim(0, 350)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("求解失败：", sol.message)