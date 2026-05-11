"""
改进的 ResUNet + CBAM 注意力机制
用于红外微小目标检测

架构：
ResUNet 主干 + CBAM 注意力 (通道+空间) + 多尺度特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
                nn.BatchNorm2d(out_channels))
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


class ChannelAttention(nn.Module):
    """通道注意力 (CBAM 的通道部分)"""
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力 (CBAM 的空间部分)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """完整的 CBAM 模块 (通道 + 空间注意力)"""
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class ResUNet_CBAM(nn.Module):
    """
    ResUNet with CBAM Attention
    
    改进点：
    1. 在每个解码层后添加 CBAM 注意力
    2. 在跳跃连接处融合多尺度信息
    3. 更好的特征学习能力
    """
    def __init__(self, num_classes=1, input_channels=1, block=Res_block, 
                 num_blocks=[2,2,2,2], nb_filter=[16, 32, 64, 128, 256]):
        super(ResUNet_CBAM, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ====== 编码路径 ======
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])

        # ====== 解码路径 ======
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_2 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_3 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        # ====== CBAM 注意力模块 ======
        # 在解码层添加注意力，提升特征质量
        self.cbam_2_1 = CBAM(nb_filter[2])
        self.cbam_1_2 = CBAM(nb_filter[1])
        self.cbam_0_3 = CBAM(nb_filter[0])

        # 最终输出层
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        # ====== 编码 ======
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        # ====== 解码 + 注意力 ======
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_1 = self.cbam_2_1(x2_1)  # 注意力增强
        
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))
        x1_2 = self.cbam_1_2(x1_2)  # 注意力增强
        
        x0_3 = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))
        x0_3 = self.cbam_0_3(x0_3)  # 注意力增强

        # ====== 输出 ======
        output = self.final(x0_3)
        return output


def ResUNet_CBAM_model(**kwargs):
    """方便的模型初始化函数"""
    return ResUNet_CBAM(**kwargs)
