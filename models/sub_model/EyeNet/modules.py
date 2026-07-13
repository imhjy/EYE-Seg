import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention 在 “Attention Is All You Need” 中提出
    计算所有 key 的查询的点积，将每个 key 除以 sqrt（dim），
    并应用 softmax 函数来获取值的权重
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx):
        B, C, H, W = xx.shape
        x = xx.view(B, C, -1).permute(2, 0, 1)  # (H*W, cav_num, C), perform self attention on each pixel.
        h = self.att(x, x, x)
        out = h.permute(1, 2, 0).view(B, C, H, W)  # C, W, H before

        return out


class GTFD(nn.Module):
    """
    en: 32 => 64  64 => 128 128 => 256 256 => 256
    de: 256 => 256  256 => 128 128 => 64 64 => 32

    """

    def __init__(self, layer_nums=[2, 3, 5, 8], layer_dims=[32, 64, 128, 256], strides=[2, 2, 2, 2]):
        super().__init__()
        self.layer_dims = layer_dims
        self.layer_nums = layer_nums

        self.enblocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.att_fusion_net = nn.ModuleList()
        self.num_levels = len(layer_nums)

        for idx in range(self.num_levels):
            current_dim = idx if idx == self.num_levels - 1 else idx + 1  # 1 2 3 3
            cur_layers = [
                nn.ZeroPad2d(1),  # 做padding的效果
                nn.Conv2d(
                    layer_dims[idx], layer_dims[current_dim], kernel_size=3,
                    stride=strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(layer_dims[current_dim], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(layer_dims[current_dim], layer_dims[current_dim],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(layer_dims[current_dim], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                ])
            self.enblocks.append(nn.Sequential(*cur_layers))
            current_dim2 = idx + 1 if idx == 0 else idx  # 1 2 3 3
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(  # 3 3 2 1
                    layer_dims[self.num_levels - current_dim2], layer_dims[self.num_levels - idx - 1],
                    kernel_size=strides[idx],
                    stride=strides[idx], bias=False
                ),
                nn.BatchNorm2d(layer_dims[self.num_levels - idx - 1],
                               eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)))
            self.att_fusion_net.append(AttFusion(layer_dims[current_dim]))

    def forward(self, inputs):

        encode_feature_list = []
        output_feature_list = []

        # 编码
        input_tensor = inputs[0]
        for i in range(len(inputs)):
            x = self.enblocks[i](input_tensor)
            if i != (len(inputs) - 1):
                sig_x = torch.sigmoid(x)
                input_tensor = x + sig_x * inputs[i + 1]  # 用更宏观的特征引导微观特征
            encode_feature_list.append(x)

        # AttFusion
        att_feature_list = []
        for i in range(len(inputs)):
            att_feature_list.append(self.att_fusion_net[i](encode_feature_list[i]))

        input_tensor = att_feature_list[len(att_feature_list) - 1]
        global_feature = input_tensor
        for i in range(len(inputs)):
            if i == 0:
                input_tensor = self.deblocks[i](input_tensor)
                output_feature_list.append(input_tensor)
            else:
                currentIdx = len(inputs) - i - 1
                x = encode_feature_list[currentIdx]
                sig_x = torch.sigmoid(x)
                input_tensor = sig_x * input_tensor + att_feature_list[currentIdx]
                input_tensor = self.deblocks[i](input_tensor)
                output_feature_list.append(input_tensor)
        output_feature_list.reverse()  # 翻转一下
        return output_feature_list, global_feature


# ==================================================================

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # groups = in_channels
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MFEblock(nn.Module):
    def __init__(self, in_channels, atrous_rates=[2, 4, 8]):
        super(MFEblock, self).__init__()
        out_channels = in_channels
        # modules = []
        # modules.append(nn.Sequential(
        # nn.Conv2d(in_channels, out_channels, 1, bias=False),
        # nn.BatchNorm2d(out_channels),
        # nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            # groups = in_channels , bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), )
        # nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()

        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE4 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x):
        # 多特征提取
        y0 = self.layer1(x)
        y1 = self.layer2(y0 + x)
        y2 = self.layer3(y1 + x)
        y3 = self.layer4(y2 + x)

        # 多尺度融合
        # res = torch.cat([y0,y1,y2,y3], dim=1)
        y0_weight = self.SE1(self.gap(y0))  # gap (1,64,32,32) => (1,64,1,1)
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight, y1_weight, y2_weight, y3_weight], 2)
        weight = self.softmax(self.softmax_1(weight))

        # 调整维度权重
        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)
        y3_weight = torch.unsqueeze(weight[:, :, 3], 2)
        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2 + y3_weight * y3
        return self.project(x_att + x)


class DFEM(nn.Module):
    def __init__(self, inc):
        super(DFEM, self).__init__()

        self.Conv_1 = nn.Sequential(nn.Conv2d(inc * 2, inc, kernel_size=1),
                                    nn.BatchNorm2d(inc),
                                    nn.ReLU(inplace=True)
                                    )

        self.Conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=1, stride=1),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc, (inc // 2) * 3, kernel_size=(3, 1), stride=1, padding=(1, 0), dilation=1),
            nn.BatchNorm2d((inc // 2) * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d((inc // 2) * 3, inc * 2, kernel_size=(1, 3), stride=1, padding=(0, 1), dilation=1),
            nn.BatchNorm2d(inc * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc * 2, inc, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, global_f, local_f):
        cat = torch.cat([global_f, local_f], dim=1)
        cat = self.Conv_1(cat) + global_f + local_f
        c = self.Conv(cat) + cat
        c = self.relu(c) + local_f
        return c


class EFEA(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.mfeblock = MFEblock(in_channels=in_channels)
        self.dfem = DFEM(inc=in_channels)

    def forward(self, global_f, local_f):
        local_f = self.mfeblock(local_f) + local_f

        f1 = self.dfem(global_f, local_f)

        return f1

# ==================================================================

class LDCA(nn.Module):
    def __init__(self, in_channel_global=256, in_channel_high=512, gamma=2, b=1, flag=True):
        super().__init__()
        self.in_channel_global = in_channel_global
        self.in_channel_high = in_channel_high

        if flag:
            self.dense_conv1 = nn.Sequential(
                nn.Conv2d(in_channel_global, in_channel_high, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channel_high),
                nn.ReLU(inplace=True)
            )
        else:
            self.dense_conv1 = nn.Sequential(
                nn.Conv2d(in_channel_global, in_channel_high, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channel_high),
                nn.ReLU(inplace=True)
            )



        self.dense_conv2 = nn.Sequential(
            nn.Conv2d(in_channel_high, in_channel_high, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel_high),
            nn.ReLU(inplace=True)
        )

        self.dense_conv3 = nn.Sequential(
            nn.Conv2d(in_channel_high * 3, in_channel_high * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel_high * 2),
            nn.Conv2d(in_channel_high * 2, in_channel_high, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.learn = nn.Parameter(torch.zeros(in_channel_high, 1, 1))  # (32,1,1)

        kernel_size = int(abs((math.log(in_channel_high, 2) + b) / gamma))  # math.log(channels, 2) 以2为底数, 求对数
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_feature, high_dimensional_features):
        global_feature = self.dense_conv1(global_feature)
        global_feature_process = self.dense_conv2(global_feature)

        cat = torch.cat([global_feature, global_feature_process, high_dimensional_features], dim=1)

        cat = self.dense_conv3(cat)

        v = self.avg_pool(cat)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return cat * v + high_dimensional_features * self.learn
