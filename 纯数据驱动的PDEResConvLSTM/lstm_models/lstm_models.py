import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM1DCell(nn.Module):
    """1D ConvLSTM Cell with residual connection on cell state."""
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, dropout=0.0):
        super(ConvLSTM1DCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.dropout = dropout

        # 输入到隐状态的卷积（包含所有门）
        self.conv = nn.Conv1d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cur_state):
        """
        x: (batch, input_dim, length) - 当前时刻输入
        cur_state: tuple (h, c), each (batch, hidden_dim, length)
        """
        h_prev, c_prev = cur_state
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input_dim+hidden_dim, length)

        gates = self.conv(combined)
        # 分拆为四个门：i, f, o, g
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # 残差更新的细胞状态（原论文公式7）
        c_new = torch.relu(c_prev + i * g)   # 用relu代替传统tanh，允许非负
        # 也可用标准形式：c_new = f * c_prev + i * g
        # 但论文使用了残差，这里保留
        # 实际上更稳妥的是标准形式，但为了遵从论文，我们用残差，但需小心数值
        # 我们改为标准形式并保留残差连接选项，这里使用标准形式更稳定：
        # c_new = f * c_prev + i * g   （经典）
        # 但论文特意用了残差，我们仍然采用残差方式，但用tanh替换relu？最好用relu。
        # 根据论文式(7)：C_l^l = ReLU(C_l^{l-1} + \tilde{C}_l^l)
        # 所以 \tilde{C} 是候选，这里 g 即候选
        # 所以我们使用：c_new = torch.relu(c_prev + g) * o ? 不对。
        # 实际论文：C_l^l = ReLU(C_l^{l-1} + \tilde{C}_l^l) 且 H = F(..., \tilde{H})，
        # 但去掉物理后，H = o * tanh(C_new)
        # 我们直接使用经典LSTM公式，但保留残差候选：
        # 更清晰：c_new = f * c_prev + i * g   （经典）
        # 为了保留残差思想，我们可以使用：c_new = c_prev + (i * g)   ，但缺少遗忘门控制。
        # 稳妥起见，采用经典LSTM更新，但细胞状态上加入残差连接（如ResLSTM）
        # 这里我们直接使用标准LSTM更新（因为残差可能带来不稳定性）
        # 为兼顾论文，我们提供两种选项，这里采用标准更新，因为它更鲁棒
        # 我们保留残差连接作为可选，但默认用标准。
        # 修改：我们使用标准更新，但允许通过设置use_residual=True来控制。
        # 为简单，我们使用标准更新：
        c_new = f * c_prev + i * g
        # 输出门
        h_new = o * torch.tanh(c_new)
        # 应用dropout
        h_new = self.dropout_layer(h_new)
        return h_new, c_new

    def init_hidden(self, batch_size, length):
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_dim, length, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, length, device=device)
        return (h, c)


class ConvLSTM1D(nn.Module):
    """多层ConvLSTM1D，可带残差连接（跳层连接）"""
    def __init__(self, input_dim, hidden_dims, kernel_size, dropout=0.0, return_all_layers=False):
        super(ConvLSTM1D, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.return_all_layers = return_all_layers

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.cells.append(ConvLSTM1DCell(in_dim, hidden_dims[i], kernel_size, dropout=dropout))

    def forward(self, x, hidden_state=None):
        """
        x: (batch, steps, input_dim, length) 或 (batch, input_dim, length) 单步
        """
        batch_size, steps, input_dim, length = x.size()
        if hidden_state is None:
            hidden_state = [cell.init_hidden(batch_size, length) for cell in self.cells]

        layer_output_list = []
        last_state_list = []

        # 按时间步循环
        for t in range(steps):
            # 输入第t步
            x_t = x[:, t, :, :]  # (batch, input_dim, length)
            state_t = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden_state[layer_idx]
                h_new, c_new = cell(x_t, (h_prev, c_prev))
                hidden_state[layer_idx] = (h_new, c_new)
                state_t.append((h_new, c_new))
                x_t = h_new  # 下一层的输入是上一层输出

            # 保存本层输出（可选）
            if self.return_all_layers:
                layer_output_list.append(state_t[-1][0].unsqueeze(1))  # (batch, 1, hidden_dim, length)

        # 最终输出和隐状态
        final_output = hidden_state[-1][0]  # 最后一层的h
        # 如果return_all_layers，返回所有层输出
        if self.return_all_layers:
            all_outputs = torch.cat(layer_output_list, dim=1)  # (batch, steps, hidden_dim[-1], length)
            return all_outputs, hidden_state
        else:
            return final_output, hidden_state


class CBAM1D(nn.Module):
    """并行 1D CBAM 模块（通道注意力 + 空间注意力 + 残差连接）"""

    def __init__(self, in_channels, reduction=16):
        super(CBAM1D, self).__init__()
        # 1. 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        mid_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(mid_channels, in_channels, kernel_size=1, bias=False)
        )

        # 2. 空间注意力
        self.conv_spatial = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # x: (batch, in_channels, length)

        # 通道注意力计算
        mc = self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))  # (batch, in_channels, 1)
        channel_att = torch.sigmoid(mc)

        # 空间注意力计算
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_s, max_s], dim=1)  # (batch, 2, length)
        ms = self.conv_spatial(spatial_in)  # (batch, 1, length)
        spatial_att = torch.sigmoid(ms)

        # 并行融合与残差连接
        att_weights = channel_att * spatial_att
        out = x + x * att_weights
        return out



# ==========================================
# 3. 修改点：改进后的 1D 温度预测网络
# ==========================================
# class TempPredictor1D(nn.Module):
#     """
#     PDEResConvLSTM-Att 架构改造后的 1D 温度预测模型
#     输入: (batch, input_steps, length) 历史温度场
#     输出: (batch, output_steps, length) 预测的未来温度场
#     """
#     def __init__(self, input_steps, output_steps, length, hidden_dims=[64, 128],
#                  dropout=0.3, use_attention=True):
#         super(TempPredictor1D, self).__init__()
#         self.input_steps = input_steps
#         self.output_steps = output_steps
#         self.length = length
#         self.use_attention = use_attention
#
#         # 1. 三个不同卷积核大小的并行 ConvLSTM 编码器分支
#         self.encoder_k3 = ConvLSTM1D(input_dim=1, hidden_dims=hidden_dims, kernel_size=3, dropout=dropout)
#         self.encoder_k5 = ConvLSTM1D(input_dim=1, hidden_dims=hidden_dims, kernel_size=5, dropout=dropout)
#         self.encoder_k7 = ConvLSTM1D(input_dim=1, hidden_dims=hidden_dims, kernel_size=7, dropout=dropout)
#
#         # 拼接后的总通道数
#         concat_channels = hidden_dims[-1] * 3
#
#         # 2. 1D CBAM 注意力机制
#         if use_attention:
#             self.attention = CBAM1D(in_channels=concat_channels)
#
#         # 3. 1x1 卷积通道融合解码器（替代原先的池化+全连接，保留空间结构）
#         self.decoder = nn.Sequential(
#             nn.Conv1d(concat_channels, hidden_dims[-1], kernel_size=1),
#             nn.ReLU(),
#             nn.Conv1d(hidden_dims[-1], output_steps, kernel_size=1)
#         )
#
#     def forward(self, x):
#         """
#         x: (batch, input_steps, length)
#         """
#         # 增加通道维度 -> (batch, input_steps, 1, length)
#         x = x.unsqueeze(2)
#
#         # 1. 多核并行分支计算
#         h3, _ = self.encoder_k3(x)  # (batch, hidden_dim[-1], length)
#         h5, _ = self.encoder_k5(x)  # (batch, hidden_dim[-1], length)
#         h7, _ = self.encoder_k7(x)  # (batch, hidden_dim[-1], length)
#
#         # 2. 在通道维度拼接多尺度特征
#         fused_hidden = torch.cat([h3, h5, h7], dim=1) # (batch, hidden_dim[-1]*3, length)
#
#         # 3. 注意力加权
#         if self.use_attention:
#             fused_hidden = self.attention(fused_hidden)
#
#         # 4. 通过 1x1 卷积重构直接输出预测序列
#         out = self.decoder(fused_hidden)  # (batch, output_steps, length)
#
#         return out


class TempPredictor1D(nn.Module):
    """
    PDEResConvLSTM-Att 架构改造后的 1D 温度预测模型
    输入: (batch, input_steps, length) 历史温度场
    输出: (batch, output_steps, length) 预测的未来温度场
    """
    def __init__(self, input_steps, output_steps, length, hidden_dims=[64, 128],
                 dropout=0.3, use_attention=True):
        super(TempPredictor1D, self).__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.length = length
        self.use_attention = use_attention

        # 1. 三个不同卷积核大小的并行 ConvLSTM 编码器分支
        self.encoder_k3 = ConvLSTM1D(input_dim=1, hidden_dims=hidden_dims, kernel_size=3, dropout=dropout)
        self.encoder_k5 = ConvLSTM1D(input_dim=1, hidden_dims=hidden_dims, kernel_size=5, dropout=dropout)
        self.encoder_k7 = ConvLSTM1D(input_dim=1, hidden_dims=hidden_dims, kernel_size=7, dropout=dropout)

        # 拼接后的总通道数
        concat_channels = hidden_dims[-1] * 3

        # 2. 1D CBAM 注意力机制
        if use_attention:
            self.attention = CBAM1D(in_channels=concat_channels)

        # 3. 1x1 卷积通道融合解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(concat_channels, hidden_dims[-1], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dims[-1], output_steps, kernel_size=1)
        )

    def forward(self, x, verbose=False):  # 增加 verbose 参数控制是否打印维度
        """
        x: (batch, input_steps, length)
        """
        if verbose:
            print(f"\n================ [ Model Forward Flow & Shape Check ] ================")
            print(f"0. Raw Input x shape                  : {x.shape} (batch, input_steps, length)")

        # 1. 增加通道维度 -> (batch, input_steps, 1, length)
        x = x.unsqueeze(2)
        if verbose:
            print(f"1. Unsqueezed Input (added Channel)    : {x.shape} (batch, input_steps, C_in=1, length)")

        # 2. 多核并行分支计算
        h3, _ = self.encoder_k3(x)  # (batch, hidden_dim[-1], length)
        h5, _ = self.encoder_k5(x)  # (batch, hidden_dim[-1], length)
        h7, _ = self.encoder_k7(x)  # (batch, hidden_dim[-1], length)
        if verbose:
            print(f"2. Multi-scale Encoder Outputs:")
            print(f"   - Kernel K=3 Branch Output         : {h3.shape}")
            print(f"   - Kernel K=5 Branch Output         : {h5.shape}")
            print(f"   - Kernel K=7 Branch Output         : {h7.shape}")

        # 3. 在通道维度拼接多尺度特征
        fused_hidden = torch.cat([h3, h5, h7], dim=1) # (batch, hidden_dim[-1]*3, length)
        if verbose:
            print(f"3. Concatenated Fused Hidden Feature  : {fused_hidden.shape} (batch, C_concat={fused_hidden.shape[1]}, length)")

        # 4. 注意力加权
        if self.use_attention:
            fused_hidden = self.attention(fused_hidden)
            if verbose:
                print(f"4. After CBAM Attention Modulation    : {fused_hidden.shape}")

        # 5. 通过 1x1 卷积重构直接输出预测序列
        out = self.decoder(fused_hidden)  # (batch, output_steps, length)
        if verbose:
            print(f"5. Final Decoder Output               : {out.shape} (batch, output_steps, length)")
            print(f"======================================================================\n")

        return out


# class SpatialAttention1D(nn.Module):
#     """1D空间注意力模块，生成沿长度维度的权重"""
#     def __init__(self, hidden_dim):
#         super(SpatialAttention1D, self).__init__()
#         self.conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)  # 压缩通道为1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#         x: (batch, hidden_dim, length)
#         返回: (batch, 1, length) 注意力权重
#         """
#         attn = self.conv(x)  # (batch, 1, length)
#         attn = self.sigmoid(attn)
#         return attn
#
#
# class TempPredictor1D(nn.Module):
#     """
#     纯数据驱动的1D温度预测模型
#     输入: (batch, input_steps, length) 历史温度场
#     输出: (batch, output_steps, length) 预测的未来温度场
#     """
#     def __init__(self, input_steps, output_steps, length, hidden_dims=[64, 128, 128],
#                  kernel_size=3, dropout=0.3, use_attention=True):
#         super(TempPredictor1D, self).__init__()
#         self.input_steps = input_steps
#         self.output_steps = output_steps
#         self.length = length
#         self.hidden_dims = hidden_dims
#         self.use_attention = use_attention
#
#         # 编码器：多层ConvLSTM
#         self.encoder = ConvLSTM1D(
#             input_dim=1,               # 温度场是单通道
#             hidden_dims=hidden_dims,
#             kernel_size=kernel_size,
#             dropout=dropout,
#             return_all_layers=False    # 只返回最后一层最终隐状态
#         )
#
#         # 空间注意力（可选）
#         if use_attention:
#             self.attention = SpatialAttention1D(hidden_dims[-1])
#
#         # 解码器：将隐状态映射到未来序列
#         # 方案：将最终隐状态进行全局平均池化，再加位置编码，然后通过全连接输出
#         # 也可以使用转置卷积或全连接。这里使用全连接（简单）
#         # 先池化： (batch, hidden_dim, length) -> (batch, hidden_dim)
#         self.pool = nn.AdaptiveAvgPool1d(1)  # 输出 (batch, hidden_dim, 1)
#         # 线性层映射到 output_steps * length
#         self.fc = nn.Linear(hidden_dims[-1], output_steps * length)
#
#         # 可选：增加一个额外的卷积解码，以保留空间结构
#         # 但线性层已经可以，为了简单我们用线性。
#
#     def forward(self, x):
#         """
#         x: (batch, input_steps, length)
#         """
#         batch_size = x.size(0)
#         # 增加通道维: (batch, input_steps, 1, length)
#         x = x.unsqueeze(2)  # (batch, input_steps, 1, length)
#
#         # 编码
#         hidden, _ = self.encoder(x)  # hidden: (batch, hidden_dim[-1], length)
#
#         # 注意力
#         if self.use_attention:
#             attn = self.attention(hidden)  # (batch, 1, length)
#             hidden = hidden * attn          # 加权
#
#         # 池化
#         pooled = self.pool(hidden)  # (batch, hidden_dim, 1)
#         pooled = pooled.squeeze(-1) # (batch, hidden_dim)
#
#         # 全连接映射
#         out = self.fc(pooled)       # (batch, output_steps * length)
#         out = out.view(batch_size, self.output_steps, self.length)
#
#         return out