import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== 1. 配置参数 ====================
# 请将此处的路径修改为你 70.0.csv 文件的实际存放路径
CSV_PATH = './output_dataset/single_time_csvs/500.0.csv'

# 如果你的文件名其实是 70.csv，请换成下面这行：
# CSV_PATH = './datasets/output_dataset/single_time_csvs/70.csv'

OUTPUT_DIR = './outputs'  # 图像保存文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 2. 可视化主函数 ====================
def visualize_single_csv(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：未找到文件 '{file_path}'，请检查文件路径和文件名！")
        return

    filename = os.path.basename(file_path)
    print(f"正在读取文件: {filename} ...")

    # 读取 CSV 文件（无表头模式）
    df = pd.read_csv(file_path, header=None)

    # 提取坐标和温度数据
    if df.shape[1] >= 2:
        # 如果有两列及以上：第1列为空间坐标/节点，第2列为温度
        x_coords = df.iloc[:, 0].values
        temp_values = df.iloc[:, 1].values
        x_label = 'Spatial Coordinate / Grid Index'
    else:
        # 如果只有一列：这列作为温度，自动生成节点索引
        temp_values = df.iloc[:, 0].values
        x_coords = np.arange(len(temp_values))
        x_label = 'Grid Index'

    # 计算关键数据统计
    max_val = np.max(temp_values)
    min_val = np.min(temp_values)
    mean_val = np.mean(temp_values)
    max_idx = np.argmax(temp_values)
    min_idx = np.argmin(temp_values)

    print(f"----- 数据统计汇总 -----")
    print(f"空间网格点数 : {len(temp_values)}")
    print(f"最高温度 (Max): {max_val:.2f} K (位于位置 x = {x_coords[max_idx]})")
    print(f"最低温度 (Min): {min_val:.2f} K (位于位置 x = {x_coords[min_idx]})")
    print(f"平均温度 (Mean): {mean_val:.2f} K")

    # 创建画布
    plt.figure(figsize=(10, 5.5), dpi=150)

    # 绘制连续温度场曲线
    plt.plot(x_coords, temp_values, color='#1f77b4', linewidth=2.2, linestyle='-', label='True Temperature')

    # 高亮极值点
    plt.scatter(x_coords[max_idx], max_val, color='tab:red', s=70, zorder=5,
                label=f'Max Temp: {max_val:.2f} K')
    plt.scatter(x_coords[min_idx], min_val, color='tab:green', s=70, zorder=5,
                label=f'Min Temp: {min_val:.2f} K')

    # 图表样式与细节修饰
    plt.title(f'1D Temperature Distribution Profile ({filename})', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel('Temperature (K)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)

    # 左上角添加数据信息文本框
    info_text = f"Points: {len(temp_values)}\nMax: {max_val:.2f} K\nMin: {min_val:.2f} K\nMean: {mean_val:.2f} K"
    plt.gca().text(0.03, 0.94, info_text, transform=plt.gca().transAxes, fontsize=9.5,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.tight_layout()

    # 保存图片
    save_name = f"plot_{filename.replace('.csv', '')}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像保存成功：{save_path}")

    # 展出会话窗口
    plt.show()

# ==================== 3. 执行入口 ====================
if __name__ == '__main__':
    visualize_single_csv(CSV_PATH)