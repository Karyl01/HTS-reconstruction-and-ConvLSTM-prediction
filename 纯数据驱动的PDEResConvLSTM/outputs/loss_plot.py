import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(file_path):
    """
    从日志文件中提取所有 Epoch 的 Train Loss 和 Val Loss。
    返回两个列表：epochs, train_losses, val_losses
    """
    epochs = []
    train_losses = []
    val_losses = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配形如 "Epoch 01: Train Loss=0.21699 | Val Loss=0.03979"
            match = re.search(
                r'Epoch\s+(\d+):\s+Train Loss=([\d.]+)\s+\|\s+Val Loss=([\d.]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    return epochs, train_losses, val_losses

def plot_loss_curve(epochs, train_losses, val_losses, save_path='loss_curve.png'):
    """
    绘制 Train Loss 和 Val Loss 随 Epoch 变化的曲线，并标注最佳 Val Loss。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o', markersize=3, linewidth=1.5)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='s', markersize=3, linewidth=1.5)

    # 标注最佳验证损失
    best_val_idx = np.argmin(val_losses)
    best_epoch = epochs[best_val_idx]
    best_val = val_losses[best_val_idx]
    plt.scatter(best_epoch, best_val, color='green', s=100, zorder=5, label=f'Best Val Loss = {best_val:.5f} (Epoch {best_epoch})')
    plt.annotate(f'Best: {best_val:.5f}',
                 xy=(best_epoch, best_val),
                 xytext=(best_epoch+10, best_val+0.001),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=10, color='green')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.title('Training and Validation Loss vs. Epoch', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Loss curve saved to {save_path}")

if __name__ == "__main__":
    # 请将下面的路径替换为您实际日志文件的路径
    log_file = "output.txt"  # 修改为您保存日志的位置
    epochs, train_losses, val_losses = parse_log(log_file)
    if epochs:
        print(f"成功解析 {len(epochs)} 个 Epoch 的数据")
        plot_loss_curve(epochs, train_losses, val_losses)
    else:
        print("未在日志中找到任何 Epoch 记录，请检查文件格式。")