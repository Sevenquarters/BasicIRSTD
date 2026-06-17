# hotphoto.py
# 合并 ResUNet-GCA 网络定义与 Hook 特征热力图 + 预测掩膜生成
# 依赖: torch, torchvision, numpy, Pillow, opencv-python, matplotlib (可选)
# 运行: python hotphoto.py

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 网络模块：Res_block, GCA, ResUNet_GCA, ResUNet_GCA_model
# -------------------------

class Res_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class GCA(nn.Module):
    """
    Gaussian Curvature Attention (GCA) module.
    Uses Sobel filters to compute image derivatives and computes curvature:
    K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2
    Produces an attention map from curvature and applies it to features.
    """
    def __init__(self, in_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None: mid_ch = in_ch // 2 if in_ch//2>0 else in_ch
        self.reduce = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
        # small conv to produce modulation from K
        self.k_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )

        # Predefined Sobel kernels (as conv) for gradient calc (applied per-channel averaged intensity)
        sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        sobel_y = sobel_x.T
        lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

        self.register_buffer('sobel_x', torch.from_numpy(sobel_x).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.from_numpy(sobel_y).unsqueeze(0).unsqueeze(0))
        self.register_buffer('lap', torch.from_numpy(lap).unsqueeze(0).unsqueeze(0))

    def compute_curvature(self, intensity):
        """
        intensity: (B,1,H,W) float tensor
        returns K: (B,1,H,W)
        """
        # first derivatives
        f_x = F.conv2d(intensity, self.sobel_x, padding=1)
        f_y = F.conv2d(intensity, self.sobel_y, padding=1)
        # second derivatives
        f_xx = F.conv2d(f_x, self.sobel_x, padding=1)
        f_yy = F.conv2d(f_y, self.sobel_y, padding=1)
        f_xy = F.conv2d(f_x, self.sobel_y, padding=1)
        # curvature numerator and denominator
        numerator = f_xx * f_yy - f_xy * f_xy
        denom = (1.0 + f_x * f_x + f_y * f_y)
        denom = denom * denom  # squared
        # avoid div by zero
        K = numerator / (denom + 1e-8)
        return K

    def forward(self, x):
        """
        x: (B,C,H,W)
        returns: modulated features of same shape
        """
        # compute intensity as channel mean
        intensity = x.mean(dim=1, keepdim=True)
        K = self.compute_curvature(intensity)  # (B,1,H,W)
        # map curvature to [0,1] via small conv and sigmoid
        k_map = self.k_conv(K)
        k_att = self.sig(k_map)  # attention map from curvature
        # reduce/expand pathway
        y = self.reduce(x)
        y = F.relu(y, inplace=True)
        y = self.expand(y)
        y = self.sig(y)  # content gating
        # combine: attention * content
        out = x * (1.0 + k_att) * y
        return out

class ResUNet_GCA(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, num_classes=1):
        super().__init__()
        self.enc1 = Res_block(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = Res_block(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = Res_block(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = Res_block(base_ch*4, base_ch*8)

        # GCA modules at different scales - name them so we can register hooks
        self.gca_0_0 = GCA(base_ch)
        self.gca_1_0 = GCA(base_ch*2)
        self.gca_2_0 = GCA(base_ch*4)

        # decoder
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = Res_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = Res_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = Res_block(base_ch*2, base_ch)

        self.final = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)            # base_ch
        g1 = self.gca_0_0(e1)        # apply gca at scale0
        p1 = self.pool1(g1)

        e2 = self.enc2(p1)           # base_ch*2
        g2 = self.gca_1_0(e2)
        p2 = self.pool2(g2)

        e3 = self.enc3(p2)           # base_ch*4
        g3 = self.gca_2_0(e3)
        p3 = self.pool3(g3)

        e4 = self.enc4(p3)           # base_ch*8

        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, g3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, g2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, g1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

def ResUNet_GCA_model(**kwargs):
    """
    初始化函数，返回模型实例
    可通过 kwargs 调整 in_ch, base_ch, num_classes
    """
    in_ch = kwargs.get('in_ch', 1)
    base_ch = kwargs.get('base_ch', 32)
    num_classes = kwargs.get('num_classes', 1)
    model = ResUNet_GCA(in_ch=in_ch, base_ch=base_ch, num_classes=num_classes)
    return model

# -------------------------
# Hook 与热力图生成逻辑
# -------------------------

if __name__ == '__main__':
    # 设备适应
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型并移至设备
    model = ResUNet_GCA_model(in_ch=1, base_ch=32, num_classes=1).to(device)
    model.eval()

    # 在此处加载权重（如果有）
    # 请在下方中文注释处修改为你本地的 .pth 权重路径并取消注释
    # model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))

    # 全局变量用于保存 hook 捕获的特征
    captured_features = {'feat': None}

    def hook_fn(module, input, output):
        """
        希望捕获 module 的真实输出特征流。用户要求取 output[0]。
        但 output 可能是 Tensor 或 tuple/list，做兼容处理。
        """
        feat = None
        if isinstance(output, (tuple, list)):
            if len(output) > 0:
                feat = output[0]
            else:
                feat = None
        else:
            feat = output
        if feat is not None:
            # 将特征复制到 CPU（不阻塞主计算流）
            captured_features['feat'] = feat.detach().cpu().clone()

    # 注册 hook 到第二层 GCA 模块：model.gca_2_0
    model.gca_2_0.register_forward_hook(hook_fn)

    # -------------------------
    # 读取输入图像并做前向推理
    # -------------------------
    # 下面的路径请根据本地环境修改（中文注释高亮）
    # ==== 请根据本机修改以下路径（已默认指向仓库内示例文件，权重仍需手动替换） ====
    # 模型权重路径（请替换为你的 .pth 文件）
    MODEL_WEIGHTS_PATH = r'model/ResUNet/ResUNet_GCA.pth'  # <<--- 修改此处为你的模型权重路径
    # 输入红外原图（默认使用仓库中的 NUDT-SIRST 示例）
    INPUT_IMAGE_PATH = r'datasets/NUDT-SIRST/images/000116.png'  # <<--- 如需替换请修改此处
    # 输出保存路径（特征热力图/预测掩膜）
    OUT_HEATMAP_PATH = r'plots/diff_examples/NUDT-SIRST_feature_heatmap_000116.png'  # <<--- 保存的特征热力图路径
    OUT_MASK_PATH = r'plots/diff_examples/NUDT-SIRST_pred_mask_000116.png'           # <<--- 保存的预测掩膜路径

    # 如果用户确实提供了权重路径，则尝试加载（否则跳过）
    if os.path.isfile(MODEL_WEIGHTS_PATH):
        try:
            sd = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
            model.load_state_dict(sd)
            print(f'Loaded weights from {MODEL_WEIGHTS_PATH}')
        except Exception as e:
            print('加载权重失败，继续使用随机初始化模型。错误：', e)
    else:
        print('未找到权重，使用随机初始化模型，请修改 MODEL_WEIGHTS_PATH。')

    # 读取图像
    if not os.path.isfile(INPUT_IMAGE_PATH):
        raise FileNotFoundError(f'未找到输入图像，请检查 INPUT_IMAGE_PATH: {INPUT_IMAGE_PATH}')
    pil_img = Image.open(INPUT_IMAGE_PATH).convert('L')
    img_np = np.array(pil_img).astype(np.float32) / 255.0  # H x W, float [0,1]
    h, w = img_np.shape

    # 转为 Tensor: (B,1,H,W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    # 前向（触发 hook）
    with torch.no_grad():
        out_logits = model(img_tensor)  # (B,1,H,W) or (B,C,H,W)

    # 捕获到的特征位于 captured_features['feat']
    feat = captured_features.get('feat', None)
    if feat is None:
        print('未在 hook 中捕获到特征，请确认 model.gca_2_0 的前向确实被触发。')
        # 仍然继续生成预测掩膜
    else:
        # feat: CPU tensor (B,C,Hf,Wf)
        # 取 batch 0；如果空间尺寸不同，需要上采样到原图大小
        feat = feat[0]  # (C,Hf,Wf)
        # 计算跨通道绝对值平均：torch.mean(torch.abs(features), dim=0)
        heat = torch.mean(torch.abs(feat), dim=0, keepdim=False)  # (Hf,Wf)
        # 归一化映射到 [0,255]
        heat_np = heat.numpy()
        # 数值稳定归一化
        mn = float(np.min(heat_np))
        mx = float(np.max(heat_np))
        if mx - mn < 1e-8:
            norm = np.zeros_like(heat_np, dtype=np.uint8)
        else:
            norm = (heat_np - mn) / (mx - mn)
            norm = (norm * 255.0).astype(np.uint8)

        # 如果特征分辨率 != 输入图分辨率，使用 cv2.resize 上采样
        if norm.shape != (h, w):
            norm_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            norm_resized = norm

        # 应用 JET 伪彩色（OpenCV 的 COLORMAP_JET）
        heatmap_color = cv2.applyColorMap(norm_resized, cv2.COLORMAP_JET)  # BGR uint8

        # 将原始灰度图拓展为 BGR
        gray_u8 = (img_np * 255.0).astype(np.uint8)
        gray_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

        # 叠合：0.4*原图 + 0.6*热力图
        overlay = cv2.addWeighted(gray_bgr, 0.4, heatmap_color, 0.6, 0)

        # 保存热力图与叠合结果
        out_heatmap_parent = Path(OUT_HEATMAP_PATH).parent
        out_heatmap_parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(OUT_HEATMAP_PATH, overlay)
        print(f'已保存特征热力图叠合图: {OUT_HEATMAP_PATH}')

    # 生成预测掩膜：使用输出 logits（未经过 sigmoid），阈值 0.5
    # 确保 out_logits 与输入大小一致（若不同则 resize）
    out_np = out_logits.detach().cpu().squeeze(0).squeeze(0).numpy()  # Hout x Wout
    # 如果尺寸不一致，上采样到原图
    if out_np.shape != (h, w):
        out_np_resized = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        out_np_resized = out_np
    prob = 1.0 / (1.0 + np.exp(-out_np_resized))  # sigmoid
    mask = (prob >= 0.5).astype(np.uint8) * 255

    out_mask_parent = Path(OUT_MASK_PATH).parent
    out_mask_parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(OUT_MASK_PATH, mask)
    print(f'已保存预测掩膜图: {OUT_MASK_PATH}')

    print('完成：特征热力图与预测掩膜已生成。')