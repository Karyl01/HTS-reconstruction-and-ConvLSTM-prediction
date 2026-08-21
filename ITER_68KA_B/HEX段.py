import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. 物理参数与几何定义 (论文 3.1 & 3.3.1 节)
# ==========================================
I_current = 68000.0  # 电流 I = 68 kA
A_hx = 63.6 * 1e-4  # 换热器截面积：63.6 cm² -> m²
L_hx = 0.8  # 换热器长度：80 cm = 0.8 m
hp = 20000.0  # 对流换热系数与湿周乘积 hp = 20000 W/(m·K)
m_dot = 4.015 * 1e-3  # 氦气质量流量：4.015 g/s -> kg/s
Cp_he = 5.21 * 1e3  # 氦气比热：5.21 J/(g·K) -> 5210 J/(kg·K)

# 边界条件 (x = 0 处)
T_cold_hx = 65.0  # 换热器冷端（中界）温度 T(0) = 65 K
theta_in = 50.0  # 入口冷氦气温度 theta(0) = 50 K
Q0_load = 7.0  # 来自高温超导段的冷端热负荷 Q0 = 7 W


# ==========================================
# 2. 材料物性拟合公式 (论文公式 4 & 5)
# ==========================================
def lambda_copper(T):
    """铜热导率近似公式 (公式 4)"""
    # 限制 T 不低于 1K 防止分母异常
    T_safe = max(T, 1.0)
    denom = 1.21e-15 * (T_safe ** 7) - 3.763e-4
    return (1.0 / denom) + 380.0


def d_lambda_dT_copper(T):
    """铜热导率对温度 T 的导数 (用于展开二阶微分项)"""
    T_safe = max(T, 1.0)
    denom = 1.21e-15 * (T_safe ** 7) - 3.763e-4
    # d/dT (1/u + 380) = - (1/u^2) * du/dT
    du_dT = 7.0 * 1.21e-15 * (T_safe ** 6)
    return -du_dT / (denom ** 2)


def rho_copper(T):
    """铜电阻率线性近似公式 (公式 5)"""
    return 6.75e-11 * T - 2.45e-9


# ==========================================
# 3. 降阶常微分方程组定义 (一阶 ODEs)
# ==========================================
def hx_system_odes(x, Y):
    """
    状态向量 Y:
    Y[0] = T     (换热器铜体温度)
    Y[1] = dT/dx (换热器轴向温度梯度)
    Y[2] = theta (冷氦气温度)
    """
    T = Y[0]
    dT_dx = Y[1]
    theta = Y[2]

    # 获取当前温度下的物性值
    lam = lambda_copper(T)
    dlam_dT = d_lambda_dT_copper(T)
    rho = rho_copper(T)

    # 1. dy0/dx = dT/dx
    dT_dx_val = dT_dx

    # 2. dy1/dx = d²T/dx²
    # 由方程 (1): d/dx [lam * dT/dx] + rho*I²/A² - hp/A*(T - theta) = 0
    # 展开: lam * d²T/dx² + (dlam/dT)*(dT/dx)² + rho*I²/A² - hp/A*(T - theta) = 0
    joule_heat = (rho * (I_current ** 2)) / (A_hx ** 2)
    conv_cool = (hp / A_hx) * (T - theta)

    d2T_dx2_val = (conv_cool - joule_heat - dlam_dT * (dT_dx ** 2)) / lam

    # 3. dy2/dx = dtheta/dx
    # 由方程 (2): hp*(T - theta) = m_dot * Cp * dtheta/dx
    dtheta_dx_val = (hp * (T - theta)) / (m_dot * Cp_he)

    return [dT_dx_val, d2T_dx2_val, dtheta_dx_val]


# ==========================================
# 4. 初值与数值求解 (Shooting / IVP Method)
# ==========================================
# 在 x = 0 (换热器冷端) 设定初始向量
lam_0 = lambda_copper(T_cold_hx)
dT_dx_0 = Q0_load / (lam_0 * A_hx)  # 由 Q0 = lam * A * dT/dx 推导

Y0 = [T_cold_hx, dT_dx_0, theta_in]

# 沿轴向 0 到 0.8 m 进行推进求解
x_eval = np.linspace(0, L_hx, 500)

sol = solve_ivp(
    fun=hx_system_odes,
    t_span=(0, L_hx),
    y0=Y0,
    t_eval=x_eval,
    method='Radau'  # 换热器方程具有刚性(Stiff)特征，建议使用 Radau 或 BDF
)

# 提取求解结果
T_sim = sol.y[0]
dT_dx_sim = sol.y[1]
theta_sim = sol.y[2]

# 计算沿轴向的热负荷 dQ/dt = lam(T) * A * dT/dx (公式 3)
heat_flux = np.array([lambda_copper(T) * A_hx * dT for T, dT in zip(T_sim, dT_dx_sim)])

# ==========================================
# 5. 结果绘图 (复现论文图 3 & 图 4)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 图 3 复现: 温度分布曲线
ax1.plot(sol.t, T_sim, 'b-', label='T (Copper Temp)')
ax1.plot(sol.t, theta_sim, 'r--', label=r'$\theta$ (Helium Temp)')
ax1.set_title('Temperature Distribution along HX (Paper Fig. 3)')
ax1.set_xlabel('L (m)')
ax1.set_ylabel('T (K)')
ax1.grid(True, linestyle=':')
ax1.legend()

# 图 4 复现: 热负荷分布曲线
ax2.plot(sol.t, heat_flux, 'g-')
ax2.set_title('Heat Flux Distribution along HX (Paper Fig. 4)')
ax2.set_xlabel('L (m)')
ax2.set_ylabel('Heat flux (W)')
ax2.grid(True, linestyle=':')

plt.tight_layout()
plt.show()