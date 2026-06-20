import torch
import torch.nn as nn
import torch.nn.functional as F


class LDEM(nn.Module):
    """Low-level Detail Enhancement Module.

    Designed for the shallow feature map right after the first encoder block.
    It preserves weak local contrast, point-like edge response, and small target
    structures with a lightweight residual enhancement path.
    """
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 4)
        self.detail_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.ReLU(inplace=True)

    def forward(self, x):
        detail = self.detail_conv(x)
        weight = self.attention(detail)
        detail = detail * weight
        return self.refine(x + detail)


class DetailSelectiveGate(nn.Module):
    """Selective gate for shallow enhanced features.

    It uses three lightweight cues to decide which shallow responses should be
    preserved before they are reused by later dense nested paths:
    1. enhanced shallow detail response
    2. local contrast response
    3. semantic guidance from the next encoder stage
    """
    def __init__(self, channels, semantic_channels):
        super().__init__()
        hidden = max(channels // 2, 4)
        self.detail_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.contrast_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.ReLU(inplace=True)

    def forward(self, detail_feat, semantic_feat):
        semantic_feat = F.interpolate(
            semantic_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=True,
        )
        local_mean = F.avg_pool2d(detail_feat, kernel_size=3, stride=1, padding=1)
        contrast_feat = torch.abs(detail_feat - local_mean)

        gate_input = torch.cat([
            self.detail_proj(detail_feat),
            self.semantic_proj(semantic_feat),
            self.contrast_proj(contrast_feat),
        ], dim=1)
        gate = self.gate(gate_input)
        gated_detail = detail_feat * gate
        return self.refine(detail_feat + gated_detail)


class StabilityRefinementUnit(nn.Module):
    """Stabilize gated shallow responses with semantic counter-correction.

    This unit is designed for the V3 model. It keeps the V2 detail-enhanced and
    gated shallow feature, then adds three lightweight stabilizing operations:
    1. residual magnitude limiting
    2. semantic counter-correction from the next encoder stage
    3. cross-level consistency filtering
    """
    def __init__(self, channels, semantic_channels):
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(semantic_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.support_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.inconsistency_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.10))
        self.activate = nn.ReLU(inplace=True)

    def forward(self, detail_feat, semantic_feat):
        semantic_feat = F.interpolate(
            semantic_feat,
            size=detail_feat.shape[2:],
            mode='bilinear',
            align_corners=True,
        )
        semantic_feat = self.semantic_proj(semantic_feat)

        support = self.support_gate(torch.cat([detail_feat, semantic_feat], dim=1))
        inconsistency = torch.abs(detail_feat - semantic_feat)
        inconsistency = self.inconsistency_gate(inconsistency)
        stable_mask = torch.clamp(support * (1.0 - inconsistency), min=0.0, max=1.0)

        refined = self.refine(
            torch.cat([detail_feat * stable_mask, semantic_feat * stable_mask], dim=1)
        )

        # Limit residual magnitude so the gate will not overreact in later epochs.
        refined = torch.tanh(refined) * self.residual_scale
        return self.activate(detail_feat + refined)


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

class DNANet(nn.Module):
    def __init__(self, num_classes=1,input_channels=1, block=Res_CBAM_block, num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256], deep_supervision=True, mode='test'):
        super(DNANet, self).__init__()
        self.mode = mode
        self.relu = nn.ReLU(inplace = True)
        self.deep_supervision = deep_supervision
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4  = nn.Upsample(scale_factor=4,   mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,   mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16,  mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1],  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        self.conv0_4_final = self._make_layer(block, nb_filter[0]*5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final  = nn.Conv2d (nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels,  output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0),self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1),self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0),self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1),self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2),self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(x4_0)),self.up_8(self.conv0_3_1x1(x3_1)),
                       self.up_4 (self.conv0_2_1x1(x2_2)),self.up  (self.conv0_1_1x1(x1_3)), x0_4], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(Final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(Final_x0_4).sigmoid()
            return output


class DNANet_LDEM(DNANet):
    """DNANet variant with shallow low-level detail enhancement.

    The enhancement is injected after x0_0 so the strengthened shallow features
    are reused by all later nested dense interaction paths.
    """
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block,
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256],
                 deep_supervision=True, mode='test'):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            block=block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=deep_supervision,
            mode=mode,
        )
        self.ldem0 = LDEM(nb_filter[0])

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0 = self.ldem0(x0_0)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        final_x0_4 = self.conv0_4_final(
            torch.cat([
                self.up_16(self.conv0_4_1x1(x4_0)),
                self.up_8(self.conv0_3_1x1(x3_1)),
                self.up_4(self.conv0_2_1x1(x2_2)),
                self.up(self.conv0_1_1x1(x1_3)),
                x0_4
            ], 1)
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(final_x0_4).sigmoid()
            return output


class DNANet_LDEM_Gate(DNANet):
    """DNANet variant with shallow detail enhancement and selective gating."""
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block,
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256],
                 deep_supervision=True, mode='test'):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            block=block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=deep_supervision,
            mode=mode,
        )
        self.ldem0 = LDEM(nb_filter[0])
        self.detail_gate0 = DetailSelectiveGate(nb_filter[0], nb_filter[1])

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0 = self.ldem0(x0_0)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_0 = self.detail_gate0(x0_0, x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        final_x0_4 = self.conv0_4_final(
            torch.cat([
                self.up_16(self.conv0_4_1x1(x4_0)),
                self.up_8(self.conv0_3_1x1(x3_1)),
                self.up_4(self.conv0_2_1x1(x2_2)),
                self.up(self.conv0_1_1x1(x1_3)),
                x0_4
            ], 1)
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(final_x0_4).sigmoid()
            return output


class DNANet_LDEM_Gate_Stable(DNANet):
    """DNANet V3 with detail enhancement, selective gate, and stable refinement."""
    def __init__(self, num_classes=1, input_channels=1, block=Res_CBAM_block,
                 num_blocks=[2, 2, 2, 2], nb_filter=[16, 32, 64, 128, 256],
                 deep_supervision=True, mode='test'):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            block=block,
            num_blocks=num_blocks,
            nb_filter=nb_filter,
            deep_supervision=deep_supervision,
            mode=mode,
        )
        self.ldem0 = LDEM(nb_filter[0])
        self.detail_gate0 = DetailSelectiveGate(nb_filter[0], nb_filter[1])
        self.stable_refine0 = StabilityRefinementUnit(nb_filter[0], nb_filter[1])

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x0_0 = self.ldem0(x0_0)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_0 = self.detail_gate0(x0_0, x1_0)
        x0_0 = self.stable_refine0(x0_0, x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        final_x0_4 = self.conv0_4_final(
            torch.cat([
                self.up_16(self.conv0_4_1x1(x4_0)),
                self.up_8(self.conv0_3_1x1(x3_1)),
                self.up_4(self.conv0_2_1x1(x2_2)),
                self.up(self.conv0_1_1x1(x1_3)),
                x0_4
            ], 1)
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1).sigmoid()
            output2 = self.final2(x0_2).sigmoid()
            output3 = self.final3(x0_3).sigmoid()
            output4 = self.final4(final_x0_4).sigmoid()
            if self.mode == 'train':
                return [output1, output2, output3, output4]
            else:
                return output4
        else:
            output = self.final(final_x0_4).sigmoid()
            return output


