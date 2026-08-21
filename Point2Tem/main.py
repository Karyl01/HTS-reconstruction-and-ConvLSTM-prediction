import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from models.models import *

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 自动选择计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# 适配 [t, x, T] 数据集与 10 点随机采样的 Dataset 类
# ==============================================================================
class HTSDenseDataset(Dataset):
    def __init__(self, master_csv_path, num_sparse=7):
        print(f"正在从 {master_csv_path} 加载密集数据集...")
        self.df = pd.read_csv(master_csv_path)
        self.num_sparse = num_sparse

        # 按时间 t 分组，每个时间点预期包含 256 个空间网格点
        self.time_groups = list(self.df.groupby('t'))
        print(f"加载成功！总共包含 {len(self.time_groups)} 个时间步。")

    def __len__(self):
        return len(self.time_groups)

    def __getitem__(self, idx):
        t_val, group = self.time_groups[idx]

        x_full = group['x'].values.astype(np.float32)
        T_full = group['T'].values.astype(np.float32)

        # 1. 从当前时间点的 256 个点中随机抽取 10 个点作为输入
        total_pts = len(x_full)
        if total_pts >= self.num_sparse:
            sparse_indices = np.random.choice(total_pts, self.num_sparse, replace=False)
        else:
            sparse_indices = np.arange(total_pts)

        sparse_indices.sort()  # 排序使空间位置更直观

        x_sparse = x_full[sparse_indices]
        T_sparse = T_full[sparse_indices]

        # 2. 组装输入特征: 形状 [2, 10]
        sparse_input = np.stack([x_sparse, T_sparse], axis=0)

        # 3. 组装全局真实标签: 形状 [1, 256]
        T_target = np.expand_dims(T_full, axis=0)

        return (
            torch.tensor(sparse_input, dtype=torch.float32),
            torch.tensor(T_target, dtype=torch.float32),
            torch.tensor(x_full, dtype=torch.float32)
        )


# ==============================================================================
# 改进的组合损失类 (MSE + L1，增强对局部极值和细节的拟合)
# ==============================================================================
class HTS_Combined_Loss(nn.Module):
    def __init__(self, alpha=0.8):
        super(HTS_Combined_Loss, self).__init__()
        self.alpha = alpha  # MSE 权重

    def forward(self, T_pred, T_true):
        mse_loss = F.mse_loss(T_pred, T_true)
        l1_loss = F.l1_loss(T_pred, T_true)  # L1 损失对异常值更鲁棒，能帮模型推开平台期
        total_loss = self.alpha * mse_loss + (1.0 - self.alpha) * l1_loss
        return total_loss, {"loss_data": total_loss.item(), "mse": mse_loss.item()}


# ==============================================================================
# 模型预测结果可视化对比函数
# ==============================================================================
def visualize_reconstruction(model, dataset, target_time_idx=None, save_dir="./vis_results"):
    """
    可视化指定时间或随机抽样的温度场重建对比图

    参数:
        model: 训练好的 Point2TempNet1D 模型
        dataset: 数据集对象
        target_time_idx: int 或 list, 你希望指定的样本索引或时间步索引。
                         如果为 None，则默认随机抽取 3 个样本。
        save_dir: 图片保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 确定要画哪些样本/时间索引
    if target_time_idx is not None:
        if isinstance(target_time_idx, int):
            indices = [target_time_idx]
        else:
            indices = list(target_time_idx)
    else:
        indices = np.random.choice(len(dataset), size=min(3, len(dataset)), replace=False)

    plt.figure(figsize=(5 * len(indices), 5))

    for i, idx in enumerate(indices):
        # 确保索引不越界
        idx = int(idx) % len(dataset)
        sparse_input, target_full, x_full = dataset[idx]

        # 增加 Batch 维度并送入设备
        x_in_gpu = sparse_input.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_full = model(x_in_gpu)

        x_coords = x_full.numpy()
        y_true = target_full.squeeze(0).numpy()  # 真实场
        # 兼容不同维度的预测输出 [B, C, N_x] 或 [B, Time, C, N_x]
        if pred_full.ndim == 4:
            y_pred = pred_full.cpu().squeeze(0).squeeze(0).squeeze(0).numpy()
        else:
            y_pred = pred_full.cpu().squeeze(0).squeeze(0).numpy()

        x_sparse = sparse_input[0].numpy()
        T_sparse = sparse_input[1].numpy()

        plt.subplot(1, len(indices), i + 1)
        plt.plot(x_coords, y_true, label='真实温度场 (Ground Truth)', color='black', linewidth=2, linestyle='--')
        plt.plot(x_coords, y_pred, label='模型重建场 (Prediction)', color='red', linewidth=1.5)
        plt.scatter(x_sparse, T_sparse, color='blue', s=50, zorder=5, label='输入稀疏测点 (10点)')

        plt.title(f"样本/时间索引: {idx}", fontsize=12)
        plt.xlabel("引线空间位置 X (m)", fontsize=10)
        plt.ylabel("温度 T (K)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "temperature_reconstruction_custom_time.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"指定时间的对比图已成功保存至: {save_path}")
    plt.show()

# ==============================================================================
# 训练与验证主程序
# ==============================================================================
def train_model():
    master_csv = "./datasets/output_dataset/hts_pinn_dense_dataset.csv"
    num_sparse = 10
    target_len = 256
    batch_size = 16  # 适当减小 Batch Size，增加梯度更新频率，有助于跳出局部最优
    epochs = 1200  # 稍微增加总 Epoch 数
    lr = 1e-3  # 适当调大初始学习率，加快初期收敛

    if not os.path.exists(master_csv):
        print(f"找不到数据集文件: {master_csv}，请先运行你的数据生成脚本。")
        return

    dataset = HTSDenseDataset(master_csv_path=master_csv, num_sparse=num_sparse)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = Point2TempNet1D(in_dim=2, feature_dim=512, target_len=target_len).to(device)

    # 替换为组合损失
    criterion = HTS_Combined_Loss(alpha=0.7)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 改用 AdamW 并加一点权重衰减防止过拟合

    # 改用余弦退火学习率调度器（CosineAnnealingLR），它能平滑地将学习率降到接近 0，强制模型在后期突破平台期
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"\n开始优化后的纯数据驱动模型训练 (Batch Size: {batch_size}, Epochs: {epochs})...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for sparse_inputs, targets, _ in dataloader:
            sparse_inputs = sparse_inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            T_pred = model(sparse_inputs)
            loss, loss_dict = criterion(T_pred, targets)

            loss.backward()

            # 梯度裁剪，防止梯度震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * sparse_inputs.size(0)

        avg_loss = epoch_loss / len(dataset)
        scheduler.step()  # 更新余弦退火学习率

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | MSE: {loss_dict['mse']:.4f} | LR: {current_lr:.6f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/point2temp_net_1d.pth")
    print("\n训练完成！模型权重已保存至 ./checkpoints/point2temp_net_1d.pth")

    print("\n正在生成模型重建效果对比图...")
    visualize_reconstruction(model, dataset, target_time_idx=[0, 50, 100, 300, 500, 600])


if __name__ == "__main__":
    train_model()