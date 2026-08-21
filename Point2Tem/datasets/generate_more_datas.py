import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 自动选择计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 定义 1D PINN 网络结构 (x, t) -> T
# ==========================================
class HeatPINN1D(nn.Module):
    def __init__(self):
        super(HeatPINN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


# ==========================================
# 3. 构建 1D 物理损失函数 (PDE Loss)
# ==========================================
def pde_loss_1d(model, x, t, alpha=0.001):
    x.requires_grad_(True)
    t.requires_grad_(True)

    T = model(x, t)

    dT_dt = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]
    dT_dx = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, torch.ones_like(dT_dx), create_graph=True)[0]

    pde_residual = dT_dt - alpha * d2T_dx2
    return torch.mean(torch.clamp(pde_residual ** 2, max=1e3))


# ==========================================
# 4. 1D 数据读取与预处理
# ==========================================
def load_data_1d(file_dict):
    data_list = []
    print("\n🔍 开始加载 1D HTS 数据集...")

    for t_val, file_path in file_dict.items():
        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 文件不存在 {file_path}，已跳过。")
            continue

        try:
            df = pd.read_csv(file_path, header=None, engine='python', sep=r'[\s,\t]+')
        except Exception as e:
            print(f"❌ 读取文件 {file_path} 报错: {e}")
            continue

        if df.shape[1] < 2:
            try:
                df = pd.read_csv(file_path, header=None, engine='python', sep=None)
            except Exception:
                pass

        if df.shape[1] < 2:
            print(f"⚠️ 警告: 文件 {file_path} 解析后不足 2 列，已跳过。")
            continue

        df_clean = df.iloc[:, :2].copy()
        df_clean.columns = ['x', 'T']

        df_clean['x'] = pd.to_numeric(df_clean['x'], errors='coerce')
        df_clean['T'] = pd.to_numeric(df_clean['T'], errors='coerce')

        valid_df = df_clean.dropna().copy()

        if len(valid_df) == 0:
            print(f"⚠️ 警告: 文件 {file_path} 解析后有效行数为 0！")
            continue

        valid_df['t'] = float(t_val)
        data_list.append(valid_df)
        print(f"  ✅ 成功加载文件 {file_path}: 提取到 {len(valid_df)} 行有效数据 (t = {t_val}s)")

    if len(data_list) == 0:
        raise ValueError("❌ 错误：没有任何文件成功装载数据！请检查 CSV 文件格式。")

    full_df = pd.concat(data_list, ignore_index=True)

    x_min, x_max = full_df['x'].min(), full_df['x'].max()
    t_min, t_max = full_df['t'].min(), full_df['t'].max()
    T_min, T_max = full_df['T'].min(), full_df['T'].max()

    print("\n✅ 数据汇总报告:")
    print(f"  ├─ 总有效数据行数: {len(full_df)}")
    print(f"  ├─ 引线位置 X 范围(m): [{x_min:.4f}, {x_max:.4f}]")
    print(f"  ├─ 时间 T 范围(s): [{t_min:.4f}, {t_max:.4f}]")
    print(f"  └─ 温度范围(K): [{T_min:.4f}, {T_max:.4f}]\n")

    norm_params = {
        'x': (x_min, x_max),
        't': (t_min, t_max),
        'T': (T_min, T_max)
    }

    dx = (x_max - x_min) if (x_max - x_min) > 1e-8 else 1.0
    dt = (t_max - t_min) if (t_max - t_min) > 1e-8 else 1.0
    dT = (T_max - T_min) if (T_max - T_min) > 1e-8 else 1.0

    full_df['x_norm'] = (full_df['x'] - x_min) / dx
    full_df['t_norm'] = (full_df['t'] - t_min) / dt
    full_df['T_norm'] = (full_df['T'] - T_min) / dT

    return full_df, norm_params


# ==========================================
# 5. 主训练流程
# ==========================================
def train_pinn_1d(full_df, norm_params, epochs=3000, w_pde=0.01):
    x_data = torch.tensor(full_df['x_norm'].values, dtype=torch.float32).unsqueeze(1).to(device)
    t_data = torch.tensor(full_df['t_norm'].values, dtype=torch.float32).unsqueeze(1).to(device)
    T_data = torch.tensor(full_df['T_norm'].values, dtype=torch.float32).unsqueeze(1).to(device)

    model = HeatPINN1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

    print("🚀 开始训练 1D HTS PINN 模型...")

    pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in pbar:
        num_pde_pts = 5000
        x_pde = torch.rand(num_pde_pts, 1, device=device)
        t_pde = torch.rand(num_pde_pts, 1, device=device)

        model.train()
        optimizer.zero_grad()

        T_pred = model(x_data, t_data)
        loss_data = torch.mean((T_pred - T_data) ** 2)
        loss_physics = pde_loss_1d(model, x_pde, t_pde, alpha=0.001)

        total_loss = loss_data + w_pde * loss_physics

        if torch.isnan(total_loss):
            print(f"\n❌ 第 {epoch + 1} 轮出现 NaN，已提前中断。")
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        if (epoch + 1) % 20 == 0:
            pbar.set_postfix({
                'Data Loss': f"{loss_data.item():.4e}",
                'PDE Loss': f"{loss_physics.item():.4e}",
                'Total': f"{total_loss.item():.4e}"
            })

    return model


# ==========================================
# 6. 生成任意时刻的 1D 引线温度分布
# ==========================================
def generate_samples_1d(model, target_time_sec, norm_params, num_x_pts=200):
    model.eval()

    x_min, x_max = norm_params['x']
    t_min, t_max = norm_params['t']
    T_min, T_max = norm_params['T']

    dx = (x_max - x_min) if (x_max - x_min) > 1e-8 else 1.0
    dt = (t_max - t_min) if (t_max - t_min) > 1e-8 else 1.0

    x_coords = np.linspace(x_min, x_max, num_x_pts)
    x_norm = (x_coords - x_min) / dx
    t_norm = np.full_like(x_norm, (target_time_sec - t_min) / dt)

    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        T_pred_norm = model(x_tensor, t_tensor).cpu().numpy().flatten()

    T_pred_real = T_pred_norm * (T_max - T_min) + T_min
    return x_coords, T_pred_real


# ==========================================
# 7. 新增功能：导出生成的密致数据集到 CSV 文件
# ==========================================
def export_dense_dataset(model, norm_params, t_start=0, t_end=600, t_step=5, num_x_pts=200,
                         save_dir="./output_dataset"):
    """
    按指定时间步长（默认每 5s）生成全时空网格数据并导出为 CSV 文件
    """
    os.makedirs(save_dir, exist_ok=True)
    single_csv_dir = os.path.join(save_dir, "single_time_csvs")
    os.makedirs(single_csv_dir, exist_ok=True)

    time_steps = np.arange(t_start, t_end + t_step, t_step)
    all_rows = []

    print(f"\n📦 开始批量生成密集数据集 (从 {t_start}s 到 {t_end}s，步长 {t_step}s)...")

    for t_sec in tqdm(time_steps, desc="Exporting CSVs"):
        x_coords, T_preds = generate_samples_1d(model, t_sec, norm_params, num_x_pts=num_x_pts)

        # 1. 独立保存单时间点 CSV [x, T]（与原始 CSV 格式保持完全相同）
        df_single = pd.DataFrame({'x': x_coords, 'T': T_preds})
        # 独立保存单时间点 CSV [x, T]（文件名保留小数）
        df_single.to_csv(os.path.join(single_csv_dir, f"{t_sec:.1f}.csv"), index=False, header=False)

        # 2. 收集数据准备汇总保存 [t, x, T]
        for x_val, T_val in zip(x_coords, T_preds):
            all_rows.append({'t': t_sec, 'x': x_val, 'T': T_val})

    # 3. 导出包含全时间全空间的大数据集 CSV
    full_export_df = pd.DataFrame(all_rows)
    master_csv_path = os.path.join(save_dir, "hts_pinn_dense_dataset.csv")
    full_export_df.to_csv(master_csv_path, index=False)

    print(f"\n🎉 数据生成完毕！已成功保存保存至:")
    print(f"  ├─ 1. 全量数据汇总文件: {os.path.abspath(master_csv_path)}")
    print(f"  └─ 2. 按时间拆分的 CSV 目录: {os.path.abspath(single_csv_dir)}\n")


# ==========================================
# 8. 可视化
# ==========================================
def plot_results_1d(model, norm_params, vis_times, save_dir="./vis_results"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for t_sec in vis_times:
        x_coords, T_line = generate_samples_1d(model, t_sec, norm_params)
        plt.plot(x_coords, T_line, label=f't = {t_sec}s', linewidth=2)

    plt.title("HTS 引线沿轴向温度分布演化曲线 (1D PINN)", fontsize=14, fontweight='bold')
    plt.xlabel("引线位置 X (m)", fontsize=12)
    plt.ylabel("温度 T (K)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    save_path = os.path.join(save_dir, "hts_1d_temperature_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 1D 温度曲线图已成功保存至: {save_path}")
    plt.show()


# ==========================================
# 9. 执行入口
# ==========================================
if __name__ == "__main__":
    files = {
        0: '0.csv',
        200: '200.csv',
        400: '400.csv',
        512: '512.csv',
        532: '532.csv',
        536: '536.csv',
        540: '540.csv'
    }

    # 1. 加载 1D 数据
    full_df, norm_params = load_data_1d(files)

    # 2. 训练 1D PINN 模型
    model = train_pinn_1d(full_df, norm_params, epochs=3000, w_pde=0.01)

    # ==========================================================
    # 【修改点 1】：将生成密集数据集的网格点数修改为 256
    # ==========================================================
    num_pts = 256

    # 3. 批量生成并保存高密度数据集 (num_x_pts 改为 256)
    export_dense_dataset(model, norm_params, t_start=0, t_end=600, t_step=0.1, num_x_pts=num_pts)

    # 4. 选定关键时间节点绘制 1D 引线轴向温度曲线
    vis_times = [0, 100, 200, 300, 400, 500, 532, 540, 600]
    plot_results_1d(model, norm_params, vis_times)