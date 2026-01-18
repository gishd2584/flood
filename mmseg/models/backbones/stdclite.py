# Copyright (c) OpenMMLab. All rights reserved.
"""
STDC-MidNeXt-Lite + PeakSimAM
专为 Jetson NX 洪水分割优化的轻量化骨干网络
包含:
1. MidNeXtBlock: 5x5 DWConv + LayerNorm + GRN
2. PeakSimAM: 引入峰值感知的无参数注意力
3. STDC-Lite: 通道瘦身策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential
from mmseg.registry import MODELS
from ..utils import resize

# ==================================================================
# 1. 基础组件 (LayerNorm, GRN, DropPath)
# ==================================================================

class LayerNorm(nn.Module):
    """支持 NCHW 和 NHWC 的 LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
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
    """全局响应归一化 (Global Response Normalization)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ==================================================================
# 2. 核心创新一: PeakSimAM (您的专属注意力)
# ==================================================================

class PeakSimAM(nn.Module):
    """
    [Ours] Peak-Aware SimAM
    改进点: 在 SimAM 能量函数基础上，引入 Peak Factor (峰值因子)。
    原理: 洪水/水体通常具有高反射率或特定高响应，峰值因子强化了对显著区域的关注。
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        
        # 1. 计算 SimAM 基础能量项 (基于方差的背景抑制)
        mean = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mean).pow(2)
        
        # 简化的能量计算 (衡量离群程度)
        global_var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y_simam = x_minus_mu_square / (4 * global_var) + 0.5
        
        # 2. 计算 Peak Factor (基于最大值的峰值增强)
        # 获取每个通道的空间最大值
        x_max = x.amax(dim=[2, 3], keepdim=True) + self.e_lambda
        # 计算相对强度 (0~1)
        peak_factor = x / x_max
        
        # 3. 融合: 既要统计离群(SimAM)，又要响应显著(Peak)
        final_y = y_simam * peak_factor
        
        return x * self.activaton(final_y)

# ==================================================================
# 3. 核心创新二: MidNeXtBlock (您的专属卷积块)
# ==================================================================

class MidNeXtBlock(BaseModule):
    """
    [Ours] MidNeXtBlock
    特点: 5x5 DWConv + LayerNorm + GELU + GRN (模仿 ConvNeXt V2)
    优势: 比 3x3 视野大，比 7x7 速度快，适合边缘端
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 1. 5x5 Depthwise Convolution (平衡速度与感受野)
        # padding=2 保证尺寸不变
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        
        # 2. 模仿 ConvNeXt 的归一化 (LayerNorm)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 3. 倒瓶颈结构 (Inverted Bottleneck)
        # 内部通道放大 4 倍 (提供强语义提取能力)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # 维度转换 NCHW -> NHWC 以适应 LayerNorm 和 Linear
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        # 维度转回 NHWC -> NCHW
        x = x.permute(0, 3, 1, 2) 

        x = input + self.drop_path(x)
        return x

# ==================================================================
# 4. 集成模块 STDCModule_Custom
# ==================================================================

class STDCModule_Custom(BaseModule):
    """
    集成了 MidNeXtBlock 和 PeakSimAM 的 STDC 模块
    """
    def __init__(self, in_channels, out_channels, stride, norm_cfg, act_cfg, 
                 drop_path_rate=0.0, use_attn=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.stride = stride
        self.with_downsample = (stride == 2)
        
        # --- 1. 下采样 / 通道调整 (保持 STDC 原逻辑) ---
        if self.with_downsample:
            self.downsample = ConvModule(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)
        elif in_channels != out_channels:
            self.downsample = ConvModule(
                in_channels, out_channels, kernel_size=1, stride=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.downsample = nn.Identity()
            
        # --- 2. 核心特征提取：使用 MidNeXtBlock ---
        self.midnext_block = MidNeXtBlock(out_channels, drop_path=drop_path_rate)
        
        # --- 3. PeakSimAM 注意力 ---
        self.use_attn = use_attn
        if use_attn:
            self.attn = PeakSimAM()

    def forward(self, x):
        # 1. 调整尺寸/通道
        x = self.downsample(x)
        
        # 2. MidNeXtBlock 提取特征
        feat = self.midnext_block(x)
        
        # 3. PeakSimAM
        if self.use_attn:
            feat = self.attn(feat)
            
        # 残差连接 (MidNeXtBlock 内部已有残差，这里主要用于连接下采样后的 x)
        if x.shape == feat.shape:
            return x + feat 
        return feat

# ==================================================================
# 5. 骨干网络 (Backbone)
# ==================================================================

@MODELS.register_module()
class STDCNet_MidNeXt_Lite(BaseModule):
    """
    STDC-MidNeXt-Lite
    Stage 1&2: 普通卷积 (保细节)
    Stage 3/4/5: MidNeXtBlock (强语义 + 大核)
    """
    
    # 极简层数配置：每个深层 Stage 只用 1 个 Block
    # 因为 MidNeXtBlock 计算量较大 (MLP放大4倍)，必须减少层数
    arch_settings = {
        'STDCNet_Custom': [(1, 1), (1, 1), (1, 1)], 
    }

    def __init__(self, stdc_type='STDCNet_Custom', in_channels=3, 
                 # 默认使用瘦身版通道 (Lite)
                 channels=(32, 64, 128, 256, 512), 
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.channels = channels
        self.strides = self.arch_settings[stdc_type]
        
        # --- Stage 1 & 2: 保持 STDC 原生结构 (快速下采样) ---
        # 浅层不使用 MidNeXt，避免显存爆炸，且不需要太强的语义
        self.stages = ModuleList()
        self.stages.append(ConvModule(
            in_channels, channels[0], kernel_size=3, stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.stages.append(ConvModule(
            channels[0], channels[1], kernel_size=3, stride=2, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.num_shallow_features = 2

        # --- Stage 3, 4, 5: 使用 MidNeXtBlock + PeakSimAM ---
        total_depth = sum([n for n, s in self.strides])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0

        for idx, (num_blocks, stride_mode) in enumerate(self.strides):
            stage_idx = idx + 2 
            in_ch = channels[stage_idx - 1] # 上一个 Stage 的输出作为输入
            out_ch = channels[stage_idx]    # 当前 Stage 的目标输出
            
            layers = []
            
            # 第一个 Block (负责下采样)
            layers.append(STDCModule_Custom(
                in_ch, out_ch, stride=2, 
                norm_cfg=norm_cfg, act_cfg=act_cfg, 
                drop_path_rate=dpr[cur], use_attn=True
            ))
            cur += 1
            
            # 后续 Block (如果配置了多层)
            for _ in range(num_blocks - 1):
                layers.append(STDCModule_Custom(
                    out_ch, out_ch, stride=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg,
                    drop_path_rate=dpr[cur], use_attn=True
                ))
                cur += 1
            
            self.stages.append(Sequential(*layers))

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return tuple(outs[self.num_shallow_features:])

# ==================================================================
# 6. 配套组件 (ARM, FFM, ContextPath)
# ==================================================================

class AttentionRefinementModule(BaseModule):
    """ARM: 使用 PeakSimAM 替代原来的 Sigmoid Attention"""
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        # 这里的 Attention 也可以替换为 PeakSimAM，效果更好
        self.attention = PeakSimAM() 
        
    def forward(self, x):
        feat = self.conv_layer(x)
        return self.attention(feat)

class FeatureFusionModuleWithDualAttention(BaseModule):
    """FFM: 融合模块，使用 PeakSimAM"""
    def __init__(self, in_channels, out_channels, scale_factor=4, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), use_attn=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.use_attn = use_attn
        
        self.conv0 = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # 使用 PeakSimAM
        if self.use_attn: 
            self.attn_block = PeakSimAM()
            
        # 原有的通道注意力保留，形成双重注意
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(out_channels, channels, 1, bias=False, act_cfg=act_cfg),
            ConvModule(channels, out_channels, 1, bias=False, act_cfg=None),
            nn.Sigmoid())

    def forward(self, spatial, context):
        x = self.conv0(torch.cat([spatial, context], dim=1))
        
        # 1. PeakSimAM (空间+峰值关注)
        if self.use_attn: 
            x = self.attn_block(x)
            
        # 2. Channel Attention (通道关注)
        return x * self.channel_attention(x) + x

@MODELS.register_module()
class STDCContextPathNet_MidNeXt(BaseModule):
    """最终的 Context Path 类 (Backbone + ARM + FFM)"""
    def __init__(self, backbone_cfg, last_in_channels=(512, 256), out_channels=64,
                 ffm_cfg=None, upsample_mode='nearest', align_corners=None,
                 norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 构建 MidNeXt 骨干
        self.backbone = STDCNet_MidNeXt_Lite(**backbone_cfg)
        
        # 构建 ARM (针对 Stage 4, 5 的输出)
        self.arm_list = nn.ModuleList()
        for in_ch in last_in_channels:
            self.arm_list.append(AttentionRefinementModule(in_ch, out_channels, norm_cfg))
            
        # 构建 FFM
        if ffm_cfg is None:
            # 自动推断通道: Spatial(Stage3) + Context
            spatial_ch = 128 # Lite版 Stage3 输出
            ffm_cfg = dict(in_channels=spatial_ch + out_channels, out_channels=128, scale_factor=4, use_attn=True)
            
        self.ffm = FeatureFusionModuleWithDualAttention(**ffm_cfg, norm_cfg=norm_cfg)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = self.backbone(x) # 输出: [Stage3, Stage4, Stage5]
        
        # outs[-1] 是 Stage 5 (512通道)
        avg_feat = self.arm_list[0](outs[-1])
        
        # 多级 ARM 融合 (Stage 5 + Stage 4)
        if len(self.arm_list) > 1:
            avg_feat = resize(avg_feat, size=outs[-2].shape[2:], mode=self.upsample_mode, align_corners=self.align_corners)
            avg_feat = avg_feat + self.arm_list[1](outs[-2])
        
        # 上采样到 Stage 3 尺寸
        avg_feat = resize(avg_feat, size=outs[0].shape[2:], mode=self.upsample_mode, align_corners=self.align_corners)
        
        # FFM 融合 (Stage 3 + 上下文特征)
        feat_fuse = self.ffm(outs[0], avg_feat)
        
        return tuple([feat_fuse] + list(outs))