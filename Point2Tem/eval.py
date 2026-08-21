import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入你的模型定义（假设保存在 model.py 中）
from models.models import Point2TempNet1D


def select_sparse_sensors(full_temp_curve, num_points=7):
    """
    从真实的密集温度曲线中均匀或随机抽取几个离散点作为稀疏测点输入
    full_temp_curve: 形状为 [N_x] 的一维真实温度数组
    """
    N_x = len(full_temp_curve)
    # 均匀挑选索引（你也可以根据实际物理测点的位置坐标来指定固定索引）
    indices = np.linspace(0, N_x - 1, num_points, dtype=int)

    # 构造坐标 x (假设引线长度为 0.5m，归一化或按实际比例算)
    x_coords = indices / (N_x - 1) * 0.5

    # 获取对应点的真实温度
    sensor_temps = full_temp_curve[indices]

    # 组合成模型需要的输入形状 [2, num_points] -> row 0: x坐标, row 1: 温度T
    sparse_input = np.stack([x_coords, sensor_temps], axis=0)
    return sparse_input, indices


def evaluate_real_data(model_path, real_data_path, sample_idx=0, num_points=7):
    """
    主推理与绘图函数 (适配 t, x, T 表头格式)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 1. 加载模型
    N_x = 128  # 保持与你训练时的一致
    model = Point2TempNet1D(in_dim=2, feature_dim=512, target_len=N_x).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("模型权重加载成功！")

    # 2. 加载真实的密集数据集 (表头为 t, x, T)
    import pandas as pd
    df = pd.read_csv(real_data_path)
    print(f"成功读取 CSV，数据总行数: {len(df)}")

    unique_times = df['t'].unique()
    target_t = unique_times[sample_idx]  # 取第 sample_idx 个时刻

    sub_df = df[df['t'] == target_t].sort_values(by='x')
    true_full_curve = sub_df['T'].values  # 原始真实温度数组 (长度 200)
    original_x = sub_df['x'].values

    # 3. 对齐网格：如果 CSV 里的网格点数不是 128，插值对齐到模型要求的 128
    if len(true_full_curve) != N_x:
        current_len = len(true_full_curve)
        print(f"提示: 数据集中该时刻网格点数为 {current_len}, 正在插值对齐到模型要求的 {N_x}...")
        target_x = np.linspace(original_x.min(), original_x.max(), N_x)
        true_full_curve_resampled = np.interp(target_x, original_x, true_full_curve)
    else:
        true_full_curve_resampled = true_full_curve

    # 4. 从对齐后的曲线上抽取离散测点输入
    sparse_np, sensor_indices = select_sparse_sensors(true_full_curve_resampled, num_points=num_points)

    # 转换为模型输入的 Tensor 格式 [Batch=1, in_dim=2, Num_points]
    sparse_tensor = torch.tensor(sparse_np, dtype=torch.float32).unsqueeze(0).to(device)

    # 5. 模型推理预测 (输出形状 [1, 1, 128])
    with torch.no_grad():
        pred_tensor = model(sparse_tensor)
        pred_curve = pred_tensor.squeeze().cpu().numpy()

    # 6. 计算误差指标 (此时两者长度均为 128)
    mae = np.mean(np.abs(pred_curve - true_full_curve_resampled))
    rmse = np.sqrt(np.mean((pred_curve - true_full_curve_resampled) ** 2))
    print(f"对比评估 (时刻 t={target_t}) -> MAE: {mae:.4f} K, RMSE: {rmse:.4f} K")

    # 7. 绘图展示
    x_axis = np.linspace(0, 0.5, N_x)  # 引线总长度 0.5m

    plt.figure(figsize=(10, 5), dpi=300)

    # 7.1 画真实密集温度曲线（拟合线，使用对齐后的数组）
    plt.plot(x_axis, true_full_curve_resampled, label="True Temperature (Ground Truth)",
             color="black", linewidth=2.0, linestyle="-")

    # 7.2 画模型预测温度曲线
    plt.plot(x_axis, pred_curve, label="Predicted Temperature (Point2TempNet)",
             color="red", linewidth=2.0, linestyle="--")

    # 7.3 在图上标出被选出来的真实离散测点
    sensor_x = sparse_np[0, :]
    sensor_T = sparse_np[1, :]
    plt.scatter(sensor_x, sensor_T, color="blue", s=70, zorder=5,
                label=f"Selected Sparse Sensors ({num_points} pts)")

    plt.title(f"HTS Temperature Reconstruction (t={target_t}, Sample #{sample_idx})", fontsize=14)
    plt.xlabel("Position along HTS tape (m)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    # 保存并展示
    output_filename = f"real_data_comparison_sample_{sample_idx}.png"
    plt.savefig(output_filename)
    print(f"对比图已成功保存为 '{output_filename}'")
    plt.show()


if __name__ == "__main__":
    # --- 请根据你的实际路径修改参数 ---
    MODEL_PATH = "results/point2temp_hts_model.pth"  # 训练好的模型权重路径
    REAL_DATA_PATH = "datasets/output_dataset/hts_pinn_dense_dataset.csv"  # 你的真实数据集路径

    # 运行第 0 个样本，抽取 7 个离散测点进行预测和对比
    evaluate_real_data(MODEL_PATH, REAL_DATA_PATH, sample_idx=0, num_points=7)