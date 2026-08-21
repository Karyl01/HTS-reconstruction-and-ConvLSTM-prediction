import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


# ==============================================================================
# Step 1: 超导与基质材料物性参数计算模块 (Material Properties)
# ==============================================================================
class HTSMaterialProperties:
    """
    高温超导引线非线性物性参数拟合类
    """

    @staticmethod
    def C_func(T):
        """体积热容 C(T) = rho_m * c_p(T) [J/(m^3·K)]"""
        return torch.clamp(1.5e3 * (T ** 2.0) + 1.0e5, min=1.0e4, max=3.0e6)

    @staticmethod
    def lambda_func(T):
        """热导率 lambda(T) [W/(m·K)]"""
        return torch.clamp(10.0 + 0.15 * T - 0.0002 * (T ** 2), min=1.0, max=100.0)

    @staticmethod
    def dlambda_dT_func(T):
        """热导率对温度的导数 d(lambda)/dT"""
        return 0.15 - 0.0004 * T

    @staticmethod
    def rho_func(T, T_crit=90.0):
        """
        等效电阻率 rho(T) [Ohm·m]
        当 T < T_crit 时超导无电阻；当 T >= T_crit 时转为基质金属等效电阻率
        """
        rho_matrix = 1.0e-8 * (1.0 + 0.004 * (T - T_crit))
        return torch.where(T < T_crit, torch.zeros_like(T), rho_matrix)


# ==============================================================================
# Step 2: 1D-Point2TempNet 网络架构 (采用 SiLU 保证二阶导数平滑)
# ==============================================================================
class PointNetEncoder1D(nn.Module):
    def __init__(self, in_dim=2, feature_dim=512):
        super(PointNetEncoder1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        global_feat = torch.max(x, 2)[0]  # 全局最大池化聚合测点特征 [B, feature_dim]
        return global_feat


class DeconvDecoder1D(nn.Module):
    def __init__(self, feature_dim=512, out_channels=1, target_len=256):
        super(DeconvDecoder1D, self).__init__()
        self.target_len = target_len
        assert target_len % 8 == 0, "target_len 必须是 8 的倍数！"

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256 * (target_len // 8)),
            nn.BatchNorm1d(256 * (target_len // 8)),
            nn.SiLU(inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(64, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        scale_len = self.target_len // 8
        x = self.fc(x)
        x = x.view(-1, 256, scale_len)
        out = self.deconv(x)
        out = torch.clamp(out, min=4.5, max=300.0)
        return out


class Point2TempNet1D(nn.Module):
    def __init__(self, in_dim=2, feature_dim=512, target_len=256):
        super(Point2TempNet1D, self).__init__()
        self.encoder = PointNetEncoder1D(in_dim=in_dim, feature_dim=feature_dim)
        self.decoder = DeconvDecoder1D(feature_dim=feature_dim, target_len=target_len)

    def forward(self, sparse_points):
        feat = self.encoder(sparse_points)
        T_pred = self.decoder(feat)
        return T_pred


# ==============================================================================
# Step 3: 完整的 PINN 损失函数类 (包含 L_pde, L_ic, L_bc, L_data)
# ==============================================================================
class HTS_PINN_Loss(nn.Module):
    def __init__(self, dx, dt, A, T_L=4.5, T_H=300.0,
                 w_pde=0.001, w_ic=50.0, w_bc=50.0, w_data=1.0):
        super(HTS_PINN_Loss, self).__init__()
        self.dx = dx
        self.dt = dt
        self.A = A
        self.T_L = T_L
        self.T_H = T_H

        self.w_pde = w_pde
        self.w_ic = w_ic
        self.w_bc = w_bc
        self.w_data = w_data

        self.props = HTSMaterialProperties()

        # 1. 一阶中心差分卷积核 (用于计算 dT/dx)
        self.k_d1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.k_d1.weight.data = torch.tensor([[[-0.5 / dx, 0.0, 0.5 / dx]]], dtype=torch.float32)
        self.k_d1.weight.requires_grad = False

        # 2. 二阶中心差分卷积核 (用于计算 d^2T/dx^2)
        self.k_d2 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.k_d2.weight.data = torch.tensor([[[1.0 / (dx ** 2), -2.0 / (dx ** 2), 1.0 / (dx ** 2)]]],
                                             dtype=torch.float32)
        self.k_d2.weight.requires_grad = False

    def compute_pde_residual(self, T_curr, T_next, Current_I):
        """
        根据你推导的残差公式计算 R(x,t)
        R = C(T)*(dT/dt) - lambda(T)*(d2T/dx2) - (d(lambda)/dT)*(dT/dx)^2 - (rho(T)/A^2)*I^2
        """
        dT_dt = (T_next - T_curr) / self.dt
        dT_dx = self.k_d1(T_curr)
        d2T_dx2 = self.k_d2(T_curr)

        C_val = self.props.C_func(T_curr)
        lambda_val = self.props.lambda_func(T_curr)
        dlambda_dT_val = self.props.dlambda_dT_func(T_curr)
        rho_val = self.props.rho_func(T_curr)

        # 对应你推导的非线性 PDE 残差
        residual = (C_val * dT_dt
                    - lambda_val * d2T_dx2
                    - dlambda_dT_val * (dT_dx ** 2)
                    - (rho_val / (self.A ** 2)) * (Current_I ** 2))

        # 残差缩放因子：将物理残差量级压制到与数据 Loss 相当，防止梯度爆炸
        res_scale = 1.0e-5
        residual_scaled = residual * res_scale

        # 去除左右边界的差分毛刺，取内部网格点计算残差
        return residual_scaled[:, :, 1:-1]

    def forward(self, T_pred_seq, T_0_true, full_targets=None, Current_I=0.0, pde_weight_multiplier=1.0):
        """
        参数:
            T_pred_seq: 模型预测的时空温度序列 [B, Time_steps, 1, N_x]
            T_0_true: 初始时刻真实温度场 [B, 1, N_x]
            full_targets: 全局真实温度监督数据 [B, Time_steps, 1, N_x] (可选)
            Current_I: 当前时刻或恒定的输运电流值
            pde_weight_multiplier: 动态 Warmup 权重系数
        """
        if len(T_pred_seq.shape) == 3:
            T_pred_seq = T_pred_seq.unsqueeze(1) # 若单时间步自动扩充维度

        Time_steps = T_pred_seq.shape[1]

        # --- 1. PDE 物理残差损失 (L_pde) ---
        loss_pde = 0.0
        if Time_steps > 1:
            for t in range(Time_steps - 1):
                T_curr = T_pred_seq[:, t:t + 1, :, :]
                T_next = T_pred_seq[:, t + 1:t + 2, :, :]
                res = self.compute_pde_residual(T_curr, T_next, Current_I)
                loss_pde += torch.mean(res ** 2)
            loss_pde = loss_pde / (Time_steps - 1)
        else:
            loss_pde = torch.tensor(0.0, device=T_pred_seq.device)

        # --- 2. 初始条件损失 (L_ic) ---
        T_pred_t0 = T_pred_seq[:, 0:1, :, :]
        loss_ic = torch.mean((T_pred_t0 - T_0_true) ** 2)

        # --- 3. 边界条件损失 (L_bc) ---
        left_pred = T_pred_seq[:, :, :, 0]    # 左端冷端 x = 0
        right_pred = T_pred_seq[:, :, :, -1]  # 右端温端 x = L
        loss_bc = torch.mean((left_pred - self.T_L) ** 2) + torch.mean((right_pred - self.T_H) ** 2)

        # --- 4. 测量数据监督损失 (L_data) ---
        if full_targets is not None:
            loss_data = torch.mean((T_pred_seq - full_targets) ** 2)
        else:
            loss_data = torch.tensor(0.0, device=T_pred_seq.device)

        # --- 5. 总加权损失 (L_total) ---
        effective_w_pde = self.w_pde * pde_weight_multiplier
        total_loss = (effective_w_pde * loss_pde
                      + self.w_ic * loss_ic
                      + self.w_bc * loss_bc
                      + self.w_data * loss_data)

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_pde": loss_pde.item(),
            "loss_ic": loss_ic.item(),
            "loss_bc": loss_bc.item(),
            "loss_data": loss_data.item()
        }


