import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp

# ==========================================
# 1. 物理参数与几何定义 (论文 3.2 节)
# ==========================================
L_hts = 0.6  # 高温超导段长度，单位：m
T_cold = 4.5 # 冷端温度，单位：K
T_hot = 65.0 # 热端（中界）温度，单位：K

A_steel = 100.0 * 1e-4  # 不锈钢截面积：100 cm² -> m²
A_AgAu = 9.04 * 1e-4    # 超导带材基体截面积：9.04 cm² -> m²
A_total = A_steel + A_AgAu # 总截面积

# ==========================================
# 2. 物性拟合公式定义 (论文公式 8, 9, 10)
# ==========================================
def lambda_AgAu(T):
    """Ag-5.3wt.%Au 热导率拟合公式 (公式 8)"""
    a = -1.7465e-6
    b = 4.9848e-4
    c = -5.4966e-2
    d = 3.5040
    e = -7.4917
    return a * (T**4) + b * (T**3) + c * (T**2) + d * T + e

def lambda_steel(T):
    """不锈钢 热导率拟合公式 (公式 9)"""
    a = -1.8436e-5
    b = 1.8562e-3
    c = 7.4769e-2
    d = -1.0323e-1
    return a * (T**3) + b * (T**2) + c * T + d

def lambda_eff(T):
    """高温超导段复合材料有效热导率 (公式 10)"""
    return (A_steel / A_total) * lambda_steel(T) + (A_AgAu / A_total) * lambda_AgAu(T)

def total_thermal_conductance_integrand(T):
    """用于计算总热流量 Q0 的被积函数: A_AgAu*lambda_AgAu + A_steel*lambda_steel"""
    return A_AgAu * lambda_AgAu(T) + A_steel * lambda_steel(T)

# ==========================================
# 3. 步骤一：求全长轴向热负荷 Q0 (公式 7)
# ==========================================
# 对温度 T 从 4.5K 积分到 65K
Q0_integral, _ = quad(total_thermal_conductance_integrand, T_cold, T_hot)
Q0 = (1.0 / L_hts) * Q0_integral  # 全长常数传导热负荷 (W)

print(f"=== 计算结果 ===")
print(f"高温超导段 4.5K 端的冷端热负荷 Q0 = {Q0:.4f} W")

# ==========================================
# 4. 步骤二：求解一阶常微分方程 dT/dx = Q0 / (A_total * lambda_eff(T))
# ==========================================
def dT_dx(x, T):
    """温度梯度微分方程"""
    # T 为标量或单元素数组
    T_val = T[0] if isinstance(T, (list, np.ndarray)) else T
    return [Q0 / (A_total * lambda_eff(T_val))]

# 设定空间网格点 (0 到 0.6 m)
x_eval = np.linspace(0, L_hts, 100)

# 使用 solve_ivp 从 x=0 (T=4.5K) 开始向右积分
sol = solve_ivp(
    fun=dT_dx,
    t_span=(0, L_hts),
    y0=[T_cold],
    t_eval=x_eval,
    method='RK45'
)

# ==========================================
# 5. 绘图 (复现论文图 5)
# ==========================================
plt.figure(figsize=(7, 5))
plt.plot(sol.t, sol.y[0], 'b--', label='HTS Temperature Profile')
plt.title('HTS Segment Temperature Distribution (Paper Fig. 5)')
plt.xlabel('L (m)')
plt.ylabel('T (K)')
plt.grid(True, linestyle=':')
plt.xlim(0, 0.6)
plt.ylim(0, 70)
plt.legend()
plt.tight_layout()
plt.show()