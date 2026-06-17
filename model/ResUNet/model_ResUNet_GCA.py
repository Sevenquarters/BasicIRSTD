"""
改进的 ResUNet + GCA (Gaussian Curvature Attention)
用于红外微小目标检测

架构：
ResUNet 主干 + 三条 Skip Connection 的 GCA 预融合增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Res_block(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class GCA(nn.Module):
    """
    Gaussian Curvature Attention

    流程：
    1) 先进行卷积融合保持语义
    2) Sobel 获取一阶导数
    3) 二阶差分获取二阶导数
    4) 计算高斯曲率并做通道聚合
    5) 动态门控进行软选择输出（可消融）
    6) 额外返回单通道曲率权重图作为物理先验
    """

    def __init__(self, channels, eps=1e-6, use_gate=True):
        super(GCA, self).__init__()
        self.eps = eps
        self.use_gate = use_gate

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.curvature_scale = nn.Conv2d(1, 1, kernel_size=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1) if use_gate else None
        self.sigmoid = nn.Sigmoid()

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # Discrete second derivatives
        dxx = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        dyy = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        dxy = torch.tensor(
            [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 4.0

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("dxx", dxx)
        self.register_buffer("dyy", dyy)
        self.register_buffer("dxy", dxy)

    def _depthwise_conv3x3(self, x, kernel):
        c = x.shape[1]
        weight = kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)

    def forward(self, x):
        fused = self.fusion(x)

        fx = self._depthwise_conv3x3(fused, self.sobel_x)
        fy = self._depthwise_conv3x3(fused, self.sobel_y)
        fxx = self._depthwise_conv3x3(fused, self.dxx)
        fyy = self._depthwise_conv3x3(fused, self.dyy)
        fxy = self._depthwise_conv3x3(fused, self.dxy)

        # K = (fxx*fyy - fxy^2) / (1 + fx^2 + fy^2)^2
        numerator = fxx * fyy - fxy * fxy
        denominator = (1.0 + fx * fx + fy * fy) ** 2 + self.eps
        curvature = numerator / denominator

        curvature_agg = torch.mean(torch.abs(curvature), dim=1, keepdim=True)
        curvature_weight = self.sigmoid(self.curvature_scale(curvature_agg))
        curvature_enhanced = fused * curvature_weight

        if self.use_gate:
            gate = self.sigmoid(self.gate(fused))
            out = gate * curvature_enhanced + (1.0 - gate) * fused
        else:
            out = curvature_enhanced
        return out, curvature_weight


class ResUNet_GCA(nn.Module):
    """
    ResUNet with GCA Attention

    改进点：
    1. 在三条 Skip Connection 上引入 GCA（融合前）
    2. 用高斯曲率响应增强几何显著性
    3. 用动态门控抑制平坦背景高频噪声
    """

    def __init__(self, num_classes=1, input_channels=1, block=Res_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(ResUNet_GCA, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # ====== 编码路径 ======
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])

        # ====== 解码路径 ======
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_2 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        # ====== GCA 模块（Skip 融合前）======
        self.gca_2_0 = GCA(nb_filter[2])  # 对应 x2_0
        self.gca_1_0 = GCA(nb_filter[1])  # 对应 x1_0
        self.gca_0_0 = GCA(nb_filter[0])  # 对应 x0_0

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        # ====== 编码 ======
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        # ====== Skip 融合前 GCA 增强 ======
        x2_0_g, prior_2_0 = self.gca_2_0(x2_0)
        x1_0_g, prior_1_0 = self.gca_1_0(x1_0)
        x0_0_g, prior_0_0 = self.gca_0_0(x0_0)

        # ====== 解码 ======
        x2_1 = self.conv2_1(torch.cat([x2_0_g, self.up(x3_0) * prior_2_0], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0_g, self.up(x2_1) * prior_1_0], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0_g, self.up(x1_2) * prior_0_0], 1))

        output = self.final(x0_3)
        return output.sigmoid()


def ResUNet_GCA_model(**kwargs):
    """方便的模型初始化函数"""
    return ResUNet_GCA(**kwargs)


class ResUNet_GCA_DSPG(ResUNet_GCA):
    """
    ResUNet with Dual-Stream Prior Guidance (DSPG)

    逻辑与 ResUNet_GCA 一致，用于明确区分新架构训练名称。
    """

    def __init__(self, num_classes=1, input_channels=1, block=Res_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(ResUNet_GCA_DSPG, self).__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            block=block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
        )


def ResUNet_GCA_DSPG_model(**kwargs):
    """方便的模型初始化函数（Dual-Stream Prior Guidance）"""
    return ResUNet_GCA_DSPG(**kwargs)


class ResUNet_GCA_NoGate(ResUNet_GCA):
    """
    ResUNet with GCA (Ablation: no dynamic gate)

    仅保留曲率注意力权重，不使用动态门控分支。
    """

    def __init__(self, num_classes=1, input_channels=1, block=Res_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256]):
        super(ResUNet_GCA_NoGate, self).__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            block=block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
        )
        self.gca_2_0 = GCA(nb_filter[2], use_gate=False)
        self.gca_1_0 = GCA(nb_filter[1], use_gate=False)
        self.gca_0_0 = GCA(nb_filter[0], use_gate=False)


def ResUNet_GCA_NoGate_model(**kwargs):
    """方便的模型初始化函数（无门控消融）"""
    return ResUNet_GCA_NoGate(**kwargs)

