# stdc_midnext_ablation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential
from mmseg.registry import MODELS
from ..utils import resize

# 复用原文件中的基础组件以保持一致性
# 假设原文件名为 stdc_midnext_peak.py，如果在同一目录可以直接 import
# 这里为了文件独立运行，我重新列出必要的 LayerNorm 和 GRN
# (如果能 import，请保留 import 语句并删除下面的 LayerNorm/GRN/PeakSimAM/DropPath 定义)

# ================= 基础组件 (复制自原文件) =================
class LayerNorm(nn.Module):
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
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class DropPath(nn.Module):
    def __init__(self, drop_path=0.): # 修正变量名 drop_prob -> drop_path 以匹配调用
        super().__init__()
        self.drop_prob = drop_path
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class PeakSimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        mean = x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_square = (x - mean).pow(2)
        global_var = x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda
        y_simam = x_minus_mu_square / (4 * global_var) + 0.5
        x_max = x.amax(dim=[2, 3], keepdim=True) + self.e_lambda
        peak_factor = x / x_max
        final_y = y_simam * peak_factor
        return x * self.activaton(final_y)

# ================= 可变核大小的 MidNeXtBlock =================

class MidNeXtBlock_Ablation(BaseModule):
    """支持动态 Kernel Size 的 Block"""
    def __init__(self, dim, kernel_size, drop_path=0.):
        super().__init__()
        
        # 自动计算 padding 以保持尺寸不变
        # k=3 -> p=1, k=5 -> p=2, k=7 -> p=3
        padding = kernel_size // 2 
        
        # 动态卷积核
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) 
        x = input + self.drop_path(x)
        return x

class STDCModule_Ablation(BaseModule):
    def __init__(self, in_channels, out_channels, stride, kernel_size, norm_cfg, act_cfg, 
                 drop_path_rate=0.0, use_attn=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.stride = stride
        self.with_downsample = (stride == 2)
        
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
            
        # 传入 kernel_size
        self.midnext_block = MidNeXtBlock_Ablation(out_channels, kernel_size=kernel_size, drop_path=drop_path_rate)
        
        self.use_attn = use_attn
        if use_attn:
            self.attn = PeakSimAM()

    def forward(self, x):
        x = self.downsample(x)
        feat = self.midnext_block(x)
        if self.use_attn:
            feat = self.attn(feat)
        if x.shape == feat.shape:
            return x + feat 
        return feat

# ================= 骨干网络基类 (支持传参) =================

class STDCNet_MidNeXt_Base(BaseModule):
    arch_settings = {'STDCNet_Custom': [(1, 1), (1, 1), (1, 1)]}

    def __init__(self, kernel_size, stdc_type='STDCNet_Custom', in_channels=3, 
                 channels=(32, 64, 128, 256, 512), 
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.strides = self.arch_settings[stdc_type]
        self.stages = ModuleList()
        
        # Stage 1 & 2 (保持不变)
        self.stages.append(ConvModule(in_channels, channels[0], 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.stages.append(ConvModule(channels[0], channels[1], 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.num_shallow_features = 2

        # Stage 3, 4, 5
        total_depth = sum([n for n, s in self.strides])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0

        for idx, (num_blocks, stride_mode) in enumerate(self.strides):
            stage_idx = idx + 2 
            in_ch = channels[stage_idx - 1]
            out_ch = channels[stage_idx]
            layers = []
            
            # 传入 kernel_size
            layers.append(STDCModule_Ablation(
                in_ch, out_ch, stride=2, kernel_size=kernel_size, # <--- 关键修改
                norm_cfg=norm_cfg, act_cfg=act_cfg, 
                drop_path_rate=dpr[cur], use_attn=True
            ))
            cur += 1
            for _ in range(num_blocks - 1):
                layers.append(STDCModule_Ablation(
                    out_ch, out_ch, stride=1, kernel_size=kernel_size, # <--- 关键修改
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

# ================= 辅助组件 (ARM/FFM) =================
# 简单复制或引用，确保类名正确

class AttentionRefinementModule(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        self.attention = PeakSimAM() 
    def forward(self, x):
        return self.attention(self.conv_layer(x))

class FeatureFusionModuleWithDualAttention(BaseModule):
    def __init__(self, in_channels, out_channels, scale_factor=4, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), use_attn=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.use_attn = use_attn
        self.conv0 = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if self.use_attn: self.attn_block = PeakSimAM()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(out_channels, channels, 1, bias=False, act_cfg=act_cfg),
            ConvModule(channels, out_channels, 1, bias=False, act_cfg=None),
            nn.Sigmoid())
    def forward(self, spatial, context):
        x = self.conv0(torch.cat([spatial, context], dim=1))
        if self.use_attn: x = self.attn_block(x)
        return x * self.channel_attention(x) + x

# ==========================================================
# 注册 3x3 和 7x7 两个版本的 Context Path 网络
# ==========================================================

@MODELS.register_module()
class STDCContextPathNet_MidNeXt_3x3(BaseModule):
    """3x3 Kernel 版本"""
    def __init__(self, backbone_cfg, last_in_channels=(512, 256), out_channels=64,
                 ffm_cfg=None, upsample_mode='nearest', align_corners=None,
                 norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 强制指定 kernel_size=3
        self.backbone = STDCNet_MidNeXt_Base(kernel_size=3, **backbone_cfg)
        
        self.arm_list = nn.ModuleList()
        for in_ch in last_in_channels:
            self.arm_list.append(AttentionRefinementModule(in_ch, out_channels, norm_cfg))
        
        if ffm_cfg is None:
            spatial_ch = 128 
            ffm_cfg = dict(in_channels=spatial_ch + out_channels, out_channels=128, scale_factor=4, use_attn=True)
        
        self.ffm = FeatureFusionModuleWithDualAttention(**ffm_cfg, norm_cfg=norm_cfg)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = self.backbone(x)
        avg_feat = self.arm_list[0](outs[-1])
        if len(self.arm_list) > 1:
            avg_feat = resize(avg_feat, size=outs[-2].shape[2:], mode=self.upsample_mode, align_corners=self.align_corners)
            avg_feat = avg_feat + self.arm_list[1](outs[-2])
        avg_feat = resize(avg_feat, size=outs[0].shape[2:], mode=self.upsample_mode, align_corners=self.align_corners)
        feat_fuse = self.ffm(outs[0], avg_feat)
        return tuple([feat_fuse] + list(outs))

@MODELS.register_module()
class STDCContextPathNet_MidNeXt_7x7(BaseModule):
    """7x7 Kernel 版本"""
    def __init__(self, backbone_cfg, last_in_channels=(512, 256), out_channels=64,
                 ffm_cfg=None, upsample_mode='nearest', align_corners=None,
                 norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        # 强制指定 kernel_size=7
        self.backbone = STDCNet_MidNeXt_Base(kernel_size=7, **backbone_cfg)
        
        self.arm_list = nn.ModuleList()
        for in_ch in last_in_channels:
            self.arm_list.append(AttentionRefinementModule(in_ch, out_channels, norm_cfg))
        
        if ffm_cfg is None:
            spatial_ch = 128 
            ffm_cfg = dict(in_channels=spatial_ch + out_channels, out_channels=128, scale_factor=4, use_attn=True)
        
        self.ffm = FeatureFusionModuleWithDualAttention(**ffm_cfg, norm_cfg=norm_cfg)
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = self.backbone(x)
        avg_feat = self.arm_list[0](outs[-1])
        if len(self.arm_list) > 1:
            avg_feat = resize(avg_feat, size=outs[-2].shape[2:], mode=self.upsample_mode, align_corners=self.align_corners)
            avg_feat = avg_feat + self.arm_list[1](outs[-2])
        avg_feat = resize(avg_feat, size=outs[0].shape[2:], mode=self.upsample_mode, align_corners=self.align_corners)
        feat_fuse = self.ffm(outs[0], avg_feat)
        return tuple([feat_fuse] + list(outs))