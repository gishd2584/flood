
# 用于STDC模型的ConvNeXt V2工具模块

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm层，支持两种数据格式：channels_last (默认) 或 channels_first.
    
    channels_last对应形状为(batch_size, height, width, channels)的输入
    channels_first对应形状为(batch_size, channels, height, width)的输入
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """全局响应归一化 (Global Response Normalization) 层
    
    这是ConvNeXt V2的核心创新模块，用于增强通道间的特征竞争。
    通过计算空间维度的L2范数并进行归一化来实现。
    
    Args:
        dim (int): 输入通道数
    """
    def __init__(self, dim):
        super().__init__()
        # gamma和beta是可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x的形状: (N, H, W, C)
        # 计算每个通道在空间维度(H, W)上的L2范数
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # 对每个样本的所有通道范数进行归一化
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        # 应用可学习的仿射变换
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2基础模块
    
    结构：DWConv -> LayerNorm -> 1x1 Conv -> GELU -> GRN -> 1x1 Conv -> DropPath
    
    Args:
        dim (int): 输入通道数
        drop_path (float): DropPath率，默认为0.0
        layer_scale_init_value (float): LayerScale初始化值，默认为1e-6
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 深度可分离卷积（7x7）
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm层
        self.norm = LayerNorm(dim, eps=1e-6)
        # 第一个1x1卷积（升维到4倍）
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # GELU激活函数
        self.act = nn.GELU()
        # GRN层（ConvNeXt V2的关键创新）
        self.grn = GRN(4 * dim)
        # 第二个1x1卷积（降维回原始维度）
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # LayerScale（可选的缩放参数）
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # DropPath用于随机深度
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # 深度卷积
        x = self.dwconv(x)
        # 维度转换: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # LayerNorm
        x = self.norm(x)
        # 升维
        x = self.pwconv1(x)
        # 激活
        x = self.act(x)
        # GRN层
        x = self.grn(x)
        # 降维
        x = self.pwconv2(x)
        # 如果使用LayerScale
        if self.gamma is not None:
            x = self.gamma * x
        # 维度转换回来: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # 残差连接和DropPath
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """随机深度（Stochastic Depth），按样本随机丢弃路径（当应用于残差块的主路径时）
    
    Args:
        drop_prob (float): 丢弃概率，默认为0.0
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 生成随机张量，形状为(batch_size, 1, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        output = x.div(keep_prob) * random_tensor
        return output