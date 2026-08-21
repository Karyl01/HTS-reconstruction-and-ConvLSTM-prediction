import os
import glob
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from lstm_models.lstm_models import *

# -------------------- 配置参数 --------------------
CONFIG = {
    'data_dir': './datasets/output_dataset/single_time_csvs',  # CSV文件夹
    'time_interval': 1.0, # 【新增】重采样时间间隔（秒），若设为 None 则使用原始所有 CSV 步长
    'input_steps': 20,    # 过去步长
    'output_steps': 20,   # 预测未来步长
    'train_ratio': 0.8,
    'val_ratio': 0.1,     # 剩余为测试
    'batch_size': 32,
    'epochs': 200,
    'lr': 1e-3,
    'hidden_dims': [64, 128],
    'dropout': 0.3,
    'use_attention': True,
    'alpha': 0.6,   # MSE权重
    'beta': 0.4,    # MAE权重
    'gamma': 0.05,  # 梯度损失权重
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_dir': './checkpoints',
    'output_dir': './outputs',
    'seed': 42,
}
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# -------------------- 设置随机种子 --------------------
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# -------------------- 辅助工具函数 --------------------
def get_csv_time(filepath):
    """从文件名解析精确的时间浮点数 (例如 '70.1.csv' -> 70.1)"""
    filename = os.path.basename(filepath)
    time_str = os.path.splitext(filename)[0]
    return float(time_str)


# -------------------- 数据集类 --------------------
class TempSequenceDataset(Dataset):
    def __init__(self, data, input_steps, output_steps):
        self.data = data
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.total_samples = data.shape[0] - input_steps - output_steps + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_steps]
        y = self.data[idx + self.input_steps: idx + self.input_steps + self.output_steps]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# -------------------- 数据加载与预处理 --------------------
# def load_data(config):
#     # 1. 修复：按浮点数值正确排序文件名
#     csv_files = sorted(
#         glob.glob(os.path.join(config['data_dir'], '*.csv')),
#         key=get_csv_time
#     )
#     if not csv_files:
#         raise FileNotFoundError(f"No CSV files found in {config['data_dir']}")
#
#     # 提取实际时刻列表 (用于作图精准显示真实时间)
#     time_stamps = np.array([get_csv_time(f) for f in csv_files])
#
#     # 从第一个文件读取 x 空间坐标 (第1列，索引0)
#     sample_df = pd.read_csv(csv_files[0], header=None)
#     x_coords = sample_df.iloc[:, 0].values
#
#     # 读取所有时间步的温度数据 (第2列，索引1)
#     temp_data = []
#     for f in csv_files:
#         df = pd.read_csv(f, header=None)
#         temp = df.iloc[:, 1].values.astype(np.float32)  # 温度 (K)
#         temp_data.append(temp)
#     temp_data = np.array(temp_data)  # (num_time, length)
#
#     print(f"Loaded {temp_data.shape[0]} time steps from {time_stamps[0]}s to {time_stamps[-1]}s.")
#     print(f"Spatial range: x = {x_coords.min():.3f} ~ {x_coords.max():.3f} ({len(x_coords)} points)")
#
#     # 划分训练/验证/测试集
#     total_len = temp_data.shape[0]
#     train_len = int(total_len * config['train_ratio'])
#     val_len = int(total_len * config['val_ratio'])
#
#     train_data = temp_data[:train_len]
#     val_data = temp_data[train_len:train_len + val_len]
#     test_data = temp_data[train_len + val_len:]
#
#     t_min = train_data.min()
#     t_max = train_data.max()
#
#     def normalize(data, min_val, max_val):
#         return (data - min_val) / (max_val - min_val + 1e-8)
#
#     train_norm = normalize(train_data, t_min, t_max)
#     val_norm = normalize(val_data, t_min, t_max)
#     test_norm = normalize(test_data, t_min, t_max)
#
#     dataset_train = TempSequenceDataset(train_norm, config['input_steps'], config['output_steps'])
#     dataset_val = TempSequenceDataset(val_norm, config['input_steps'], config['output_steps'])
#     dataset_test = TempSequenceDataset(test_norm, config['input_steps'], config['output_steps'])
#
#     dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
#     dataloader_val = DataLoader(dataset_val, batch_size=config['batch_size'], shuffle=False, drop_last=False)
#     dataloader_test = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False, drop_last=False)
#
#     return dataloader_train, dataloader_val, dataloader_test, t_min, t_max, x_coords, time_stamps, temp_data
# -------------------- 数据加载与预处理 --------------------
def load_data(config):
    # 1. 获取并按时间排序所有 CSV 文件
    all_csv_files = sorted(
        glob.glob(os.path.join(config['data_dir'], '*.csv')),
        key=get_csv_time
    )
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in {config['data_dir']}")

    all_time_stamps = np.array([get_csv_time(f) for f in all_csv_files])

    # 2. 按指定的 time_interval（如每 20s 或 10s）过滤/重采样文件列表
    target_interval = config.get('time_interval', None)
    if target_interval is not None and target_interval > 0:
        t_start = all_time_stamps[0]
        t_end = all_time_stamps[-1]

        # 生成期望的目标时间点序列: [0.0, 20.0, 40.0, 60.0, ...]
        target_times = np.arange(t_start, t_end + 1e-5, target_interval)

        # 在真实文件中寻找最近距离的时间点索引
        selected_indices = [np.abs(all_time_stamps - t).argmin() for t in target_times]

        # 去重并保持顺序（防止某些间隔过于密集导致匹配到同一个文件）
        selected_indices = sorted(list(set(selected_indices)))

        csv_files = [all_csv_files[i] for i in selected_indices]
        time_stamps = all_time_stamps[selected_indices]
        print(f"[采样成功] 按照间隔 dt ≈ {target_interval}s 筛选出 {len(csv_files)} 个时间点帧。")
    else:
        csv_files = all_csv_files
        time_stamps = all_time_stamps

    # 从第一个文件读取 x 空间坐标
    sample_df = pd.read_csv(csv_files[0], header=None)
    x_coords = sample_df.iloc[:, 0].values

    # 读取重采样后的所有时间步温度数据
    temp_data = []
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        temp = df.iloc[:, 1].values.astype(np.float32)
        temp_data.append(temp)
    temp_data = np.array(temp_data)  # (num_time_steps, spatial_points)

    print(f"Loaded {temp_data.shape[0]} time steps from {time_stamps[0]:.1f}s to {time_stamps[-1]:.1f}s.")
    print(f"Spatial range: x = {x_coords.min():.3f} ~ {x_coords.max():.3f} ({len(x_coords)} points)")

    # 划分训练/验证/测试集
    total_len = temp_data.shape[0]
    train_len = int(total_len * config['train_ratio'])
    val_len = int(total_len * config['val_ratio'])

    train_data = temp_data[:train_len]
    val_data = temp_data[train_len:train_len + val_len]
    test_data = temp_data[train_len + val_len:]

    t_min = train_data.min()
    t_max = train_data.max()

    def normalize(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val + 1e-8)

    train_norm = normalize(train_data, t_min, t_max)
    val_norm = normalize(val_data, t_min, t_max)
    test_norm = normalize(test_data, t_min, t_max)

    dataset_train = TempSequenceDataset(train_norm, config['input_steps'], config['output_steps'])
    dataset_val = TempSequenceDataset(val_norm, config['input_steps'], config['output_steps'])
    dataset_test = TempSequenceDataset(test_norm, config['input_steps'], config['output_steps'])

    dataloader_train = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    return dataloader_train, dataloader_val, dataloader_test, t_min, t_max, x_coords, time_stamps, temp_data

# -------------------- 损失函数 --------------------
def compute_loss(pred, true, alpha=0.6, beta=0.4, gamma=0.05):
    mse = nn.MSELoss()(pred, true)
    mae = nn.L1Loss()(pred, true)
    pred_diff = torch.diff(pred, dim=-1)
    true_diff = torch.diff(true, dim=-1)
    grad_loss = nn.L1Loss()(pred_diff, true_diff)
    loss = alpha * mse + beta * mae + gamma * grad_loss
    return loss, mse, mae, grad_loss


# -------------------- 训练与评估函数 --------------------
def train_one_epoch(model, dataloader, optimizer, config):
    start_time = time.time()
    model.train()
    total_loss, total_mse, total_mae, total_grad = 0, 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(config['device']), y.to(config['device'])
        optimizer.zero_grad()
        pred = model(x)
        loss, mse, mae, grad = compute_loss(pred, y, config['alpha'], config['beta'], config['gamma'])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_mae += mae.item()
        total_grad += grad.item()
    n = len(dataloader)
    time_elapsed = time.time() - start_time
    print("run time =", time_elapsed)
    return total_loss / n, total_mse / n, total_mae / n, total_grad / n


def evaluate(model, dataloader, config):
    model.eval()
    total_loss, total_mse, total_mae, total_grad = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(config['device']), y.to(config['device'])
            pred = model(x)
            loss, mse, mae, grad = compute_loss(pred, y, config['alpha'], config['beta'], config['gamma'])
            total_loss += loss.item()
            total_mse += mse.item()
            total_mae += mae.item()
            total_grad += grad.item()
    n = len(dataloader)
    return total_loss / n, total_mse / n, total_mae / n, total_grad / n





def predict_from_time_step(start_step, input_steps, output_steps, config, model, t_min, t_max, all_temp_data, x_coords,
                           time_stamps):
    """
    智能预测与全步长可视化函数：
    1. 模式 A（数据范围内）：推理未来 output_steps（如20步），生成 20 步折线对比子图 + 2D 时空热力图/误差图。
    2. 模式 B（数据范围外）：开启自回归推演未来状态。
    """
    total_time_steps = all_temp_data.shape[0]
    dt = (time_stamps[-1] - time_stamps[0]) / (total_time_steps - 1) if total_time_steps > 1 else 0.1

    # ------------------ 模式 A：数据范围内（全步长对比验证） ------------------
    if start_step + input_steps + output_steps <= total_time_steps:
        x_raw = all_temp_data[start_step: start_step + input_steps]
        y_true_raw = all_temp_data[
                     start_step + input_steps: start_step + input_steps + output_steps]  # (output_steps, spatial_len)

        x_norm = (x_raw - t_min) / (t_max - t_min + 1e-8)
        x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(config['device'])

        model.eval()
        with torch.no_grad():
            pred_norm = model(x_tensor)

        # 反归一化：还原回真实开尔文温度 (K)
        pred_raw = pred_norm.squeeze(0).cpu().numpy() * (t_max - t_min) + t_min  # (output_steps, spatial_len)

        # ---------------- 1. 计算 20 步整体量化指标 ----------------
        y_true_flat = y_true_raw.flatten()
        y_pred_flat = pred_raw.flatten()

        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100.0

        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        peak_error = np.max(np.abs(y_true_flat - y_pred_flat))

        start_time_val = time_stamps[start_step]
        input_end_time_val = time_stamps[start_step + input_steps - 1]
        pred_start_time_val = time_stamps[start_step + input_steps]
        pred_end_time_val = time_stamps[start_step + input_steps + output_steps - 1]

        print(f"\n[多步验证] 输入历史数据: t = {start_time_val:.2f}s ~ {input_end_time_val:.2f}s (共 {input_steps} 步)")
        print(
            f"[多步验证] 预测未来区间: t = {pred_start_time_val:.2f}s ~ {pred_end_time_val:.2f}s (共 {output_steps} 步)")
        print("-" * 55)
        print(f"  MAE        : {mae:.4f} K")
        print(f"  RMSE       : {rmse:.4f} K")
        print(f"  MAPE       : {mape:.4f} %")
        print(f"  R² Score   : {r2:.6f}")
        print(f"  Peak Error : {peak_error:.4f} K")
        print("-" * 55)

        # ---------------- 2. 画法一：未来 output_steps 个步长的折线对比子图阵列 ----------------
        cols = 5
        rows = int(np.ceil(output_steps / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for step in range(output_steps):
            ax = axes[step]
            t_curr = time_stamps[start_step + input_steps + step]

            ax.plot(x_coords, y_true_raw[step, :], linestyle='--', color='tab:blue', linewidth=1.8, label='True')
            ax.plot(x_coords, pred_raw[step, :], linestyle='-', color='tab:red', linewidth=1.5, alpha=0.85,
                    label='Pred')

            ax.set_title(f'Step +{step + 1} (t={t_curr:.1f}s)', fontsize=10, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5)
            if step == 0:
                ax.legend(loc='best', fontsize=8)

        # 隐藏多余的空子图网格
        for step in range(output_steps, len(axes)):
            fig.delaxes(axes[step])

        fig.text(0.5, 0.01, 'Spatial Coordinate x', ha='center', fontsize=12)
        fig.text(0.01, 0.5, 'Temperature (K)', va='center', rotation='vertical', fontsize=12)
        fig.suptitle(f'1D Temperature Multi-Step Prediction Comparison ({output_steps} Steps Ahead)', fontsize=14,
                     fontweight='bold')
        plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])

        save_subplots_path = os.path.join(config['output_dir'], f'multi_step_subplots_t{start_step}.png')
        plt.savefig(save_subplots_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"20步对比子图网格已保存至: {save_subplots_path}")

        # ---------------- 3. 画法二：2D 空间-时间热力图与误差图对比 ----------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 计算该窗口内的时间轴刻度 (以相对预测时间或绝对时间)
        future_times = time_stamps[start_step + input_steps: start_step + input_steps + output_steps]

        vmin = min(y_true_raw.min(), pred_raw.min())
        vmax = max(y_true_raw.max(), pred_raw.max())

        # 1. True Heatmap
        im0 = axes[0].imshow(y_true_raw, aspect='auto', cmap='jet', origin='lower',
                             extent=[x_coords.min(), x_coords.max(), future_times[0], future_times[-1]],
                             vmin=vmin, vmax=vmax)
        axes[0].set_title('True Temperature Field', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Spatial x')
        axes[0].set_ylabel('Time t (s)')
        fig.colorbar(im0, ax=axes[0], label='Temperature (K)')

        # 2. Pred Heatmap
        im1 = axes[1].imshow(pred_raw, aspect='auto', cmap='jet', origin='lower',
                             extent=[x_coords.min(), x_coords.max(), future_times[0], future_times[-1]],
                             vmin=vmin, vmax=vmax)
        axes[1].set_title('Predicted Temperature Field', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Spatial x')
        fig.colorbar(im1, ax=axes[1], label='Temperature (K)')

        # 3. Absolute Error Heatmap
        abs_err = np.abs(y_true_raw - pred_raw)
        im2 = axes[2].imshow(abs_err, aspect='auto', cmap='inferno', origin='lower',
                             extent=[x_coords.min(), x_coords.max(), future_times[0], future_times[-1]])
        axes[2].set_title('Absolute Error |True - Pred|', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Spatial x')
        fig.colorbar(im2, ax=axes[2], label='Error (K)')

        plt.suptitle(f'Spatio-Temporal Comparison over Future {output_steps} Steps', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_heatmap_path = os.path.join(config['output_dir'], f'multi_step_heatmap_t{start_step}.png')
        plt.savefig(save_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D时空热力图与误差图已保存至: {save_heatmap_path}")

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'PeakError': peak_error
        }

        return pred_raw, y_true_raw, metrics

    # ------------------ 模式 B：数据范围外（自回归外推模式） ------------------
    else:
        print(f"\n[提示] 起始点 ({start_step}) 已超出或接近已知数据总步数 ({total_time_steps})。")
        print(f"[提示] 自动激活【自回归迭代推演 (Autoregressive Rollout)】模式...")

        seed_raw = all_temp_data[-input_steps:]
        seed_norm = (seed_raw - t_min) / (t_max - t_min + 1e-8)
        curr_tensor = torch.FloatTensor(seed_norm).unsqueeze(0).to(config['device'])

        generated_blocks_norm = []
        current_step_count = total_time_steps
        target_end_step = start_step + input_steps + output_steps

        model.eval()
        with torch.no_grad():
            while current_step_count < target_end_step:
                pred_block_norm = model(curr_tensor)
                generated_blocks_norm.append(pred_block_norm.squeeze(0).cpu().numpy())
                curr_tensor = torch.cat([curr_tensor, pred_block_norm], dim=1)[:, -input_steps:, :]
                current_step_count += output_steps

        all_pred_norm = np.concatenate(generated_blocks_norm, axis=0)
        all_pred_raw = all_pred_norm * (t_max - t_min) + t_min

        last_known_time = time_stamps[-1]
        future_predicted_time = last_known_time + (target_end_step - total_time_steps) * dt

        print(f" -> 已通过自回归推演至未来状态: t ≈ {future_predicted_time:.2f}s (第 {target_end_step} 步)")

        plt.figure(figsize=(9, 5))
        plt.plot(x_coords, all_temp_data[-1, :], linestyle='--', color='gray', linewidth=1.8, alpha=0.7,
                 label=f'Last Known True (t={last_known_time:.1f}s)')
        plt.plot(x_coords, all_pred_raw[-1, :], linestyle='-', color='tab:red', linewidth=2.0,
                 label=f'Autoregressive Pred (t ≈ {future_predicted_time:.1f}s)')

        plt.title(f'Autoregressive Future Extrapolation at t ≈ {future_predicted_time:.1f}s', fontsize=13,
                  fontweight='bold')
        plt.xlabel('Spatial Coordinate x', fontsize=11)
        plt.ylabel('Temperature (K)', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='best', fontsize=10)

        save_path = os.path.join(config['output_dir'], f'autoregressive_pred_step{target_end_step}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"自回归外推结果图已保存至: {save_path}")

        return all_pred_raw, None, None



def evaluate_comprehensive_metrics(model, dataloader, config, t_min, t_max):
    """
    在真实物理量纲（反归一化后，单位：K）下计算全测试集的综合评估指标：
    MAE, RMSE, MAPE, R², Peak Error
    """
    model.eval()
    all_preds_norm = []
    all_trues_norm = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(config['device'])
            pred = model(x)
            all_preds_norm.append(pred.cpu().numpy())
            all_trues_norm.append(y.numpy())

    # 拼接所有 Batch -> (Total_Samples, output_steps, spatial_length)
    preds_norm = np.concatenate(all_preds_norm, axis=0)
    trues_norm = np.concatenate(all_trues_norm, axis=0)

    # 1. 核心步骤：反归一化回真实开尔文温度 (K)
    preds_raw = preds_norm * (t_max - t_min) + t_min
    trues_raw = trues_norm * (t_max - t_min) + t_min

    # 展平进行全局指标计算
    y_true = trues_raw.flatten()
    y_pred = preds_raw.flatten()

    # 2. 计算各项指标
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE: 加 1e-8 防止除以 0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0

    # R² 决定系数
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    # 峰值误差 (Peak Error / Maximum Absolute Error)
    peak_error = np.max(np.abs(y_true - y_pred))

    metrics = {
        'MAE (K)': mae,
        'RMSE (K)': rmse,
        'MAPE (%)': mape,
        'R² Score': r2,
        'Peak Error (K)': peak_error
    }

    # 3. 终端可视化格式化输出
    print("\n" + "=" * 25 + " [ Test Set Precision Metrics ] " + "=" * 25)
    print(f"  {'Metric':<20} | {'Value':<15}")
    print("-" * 65)
    for k, v in metrics.items():
        if k == 'R² Score':
            print(f"  {k:<20} | {v:.6f}")
        elif k == 'MAPE (%)':
            print(f"  {k:<20} | {v:.4f}%")
        else:
            print(f"  {k:<20} | {v:.4f}")
    print("=" * 65 + "\n")

    return metrics

# -------------------- 主程序 --------------------
def main():
    config = CONFIG
    print("Using device:", config['device'])

    train_loader, val_loader, test_loader, t_min, t_max, x_coords, time_stamps, all_temp_data = load_data(config)

    sample_x, sample_y = next(iter(train_loader))
    length = sample_x.shape[2]

    # 创建模型
    model = TempPredictor1D(
        input_steps=config['input_steps'],
        output_steps=config['output_steps'],
        length=length,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        use_attention=config['use_attention']
    ).to(config['device'])

    print("\n" + "=" * 30 + " [ Model Structure ] " + "=" * 30)
    print(model)
    print("=" * 81)

    dummy_x = torch.randn(config['batch_size'], config['input_steps'], length).to(config['device'])
    with torch.no_grad():
        _ = model(dummy_x, verbose=True)

    model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')

    if os.path.exists(model_path):
        print(f"\n[提示] 检测到已保存的模型 '{model_path}'，直接加载进行推理。")
        model.load_state_dict(torch.load(model_path, map_location=config['device']))
    else:
        print(f"\n[提示] 开始训练...")
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        best_val_loss = float('inf')
        for epoch in range(1, config['epochs'] + 1):
            train_loss, train_mse, train_mae, train_grad = train_one_epoch(model, train_loader, optimizer, config)
            val_loss, val_mse, val_mae, val_grad = evaluate(model, val_loader, config)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)

            print(f"Epoch {epoch:02d}: Train Loss={train_loss:.5f} | Val Loss={val_loss:.5f}")

        model.load_state_dict(torch.load(model_path, map_location=config['device']))

    # 测试集全量物理指标评估
    test_metrics = evaluate_comprehensive_metrics(model, test_loader, config, t_min, t_max)

    return model, t_min, t_max, x_coords, time_stamps, all_temp_data


if __name__ == '__main__':
    # 1. 运行主程序
    model, t_min, t_max, x_coords, time_stamps, all_temp_data = main()

    # 2. 执行自定义时间起点推理 (要求 START_TIME_STEP 加上输入和输出步长后不超过总数据长度)
    START_TIME_STEP = 1000

    pred_temp, true_temp, metrics = predict_from_time_step(
        start_step=START_TIME_STEP,
        input_steps=CONFIG['input_steps'],
        output_steps=CONFIG['output_steps'],
        config=CONFIG,
        model=model,
        t_min=t_min,
        t_max=t_max,
        all_temp_data=all_temp_data,
        x_coords=x_coords,
        time_stamps=time_stamps
    )

if __name__ == '__main__':
    # 1. 运行主程序
    model, t_min, t_max, x_coords, time_stamps, all_temp_data = main()

    # 2. 执行自定义时间起点推理
    # 注意：数据集重采样后共 601 步，若要测试【数据范围内】的多步对比绘图，
    # START_TIME_STEP + input_steps + output_steps 必须 <= 601 (例如设为 300 或 500)
    START_TIME_STEP = 400

    pred_temp, true_temp, metrics = predict_from_time_step(
        start_step=START_TIME_STEP,
        input_steps=CONFIG['input_steps'],
        output_steps=CONFIG['output_steps'],
        config=CONFIG,
        model=model,
        t_min=t_min,
        t_max=t_max,
        all_temp_data=all_temp_data,
        x_coords=x_coords,
        time_stamps=time_stamps
    )