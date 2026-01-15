# Copyright (c) OpenMMLab. All rights reserved.
"""融合ConvNeXt V2模块的STDC网络 - 完全修复版
这个文件在原始STDC的基础上集成了ConvNeXt V2的GRN模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
from ..utils import resize

# 导入或定义ConvNeXt V2的工具模块
try:
    from .convnext_utils import GRN, LayerNorm, ConvNeXtV2Block, DropPath
except ImportError:
    # 如果找不到convnext_utils，使用本地定义
    class LayerNorm(nn.Module):
        """LayerNorm层，支持两种数据格式"""
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
        """全局响应归一化层"""
        def __init__(self, dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

        def forward(self, x):
            Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            return self.gamma * (x * Nx) + self.beta + x

    class DropPath(nn.Module):
        """随机深度"""
        def __init__(self, drop_prob=0.):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output

    class ConvNeXtV2Block(nn.Module):
        """ConvNeXt V2基础模块"""
        def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.norm = LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, 4 * dim)
            self.act = nn.GELU()
            self.grn = GRN(4 * dim)
            self.pwconv2 = nn.Linear(4 * dim, dim)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                        requires_grad=True) if layer_scale_init_value > 0 else None
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
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)
            x = input + self.drop_path(x)
            return x


class AttentionRefinementModule(BaseModule):
    """注意力精炼模块"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='Sigmoid')))

    def forward(self, x):
        x = self.conv_layer(x)
        attn = self.attention(x)
        return x * attn


class STDCModuleWithGRN(BaseModule):
    """增强版STDC模块 - 完全修复版
    
    按照原始STDC的逻辑实现，确保通道数计算正确
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_cfg=None,
                 act_cfg=None,
                 num_convs=4,
                 fusion_type='add',
                 use_grn=True,
                 drop_path_rate=0.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1
        assert fusion_type in ['add', 'cat']
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_convs = num_convs
        self.fusion_type = fusion_type
        self.use_grn = use_grn
        
        # 判断是否需要下采样
        self.with_downsample = (stride == 2)
        
        # 构建layers
        self.layers = ModuleList()
        
        # 第一个1x1卷积
        self.layers.append(
            ConvModule(
                in_channels, 
                out_channels // 2, 
                kernel_size=1, 
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        
        # 如果需要下采样，添加下采样层
        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)
            
            # add模式需要skip连接
            if fusion_type == 'add':
                self.skip = Sequential(
                    ConvModule(
                        in_channels,
                        out_channels // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=in_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=None),
                    ConvModule(
                        out_channels // 2,
                        out_channels // 2,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=None))
        
        # 添加num_convs-1个3x3卷积层
        for i in range(1, num_convs):
            in_ch = out_channels // 2 if i == 1 else out_channels // 4
            out_ch = out_channels // 4
            
            conv_layer = ConvModule(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            
            # 如果启用GRN，添加GRN层
            if self.use_grn:
                self.layers.append(Sequential(conv_layer, GRNWrapper(out_ch)))
            else:
                self.layers.append(conv_layer)
        
        # 最后的1x1卷积
        if fusion_type == 'cat':
            # 计算拼接后的通道数
            # 有下采样: 拼接out[1:]，共(num_convs-1)个，每个out_channels//4
            # 无下采样: 拼接out[0:]，第一个out_channels//2 + 其余(num_convs-1)个out_channels//4
            if self.with_downsample:
                concat_channels = out_channels // 4 * (num_convs - 1)
            else:
                concat_channels = out_channels // 2 + out_channels // 4 * (num_convs - 1)
            
            self.conv_last = ConvModule(
                concat_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            # add模式
            self.conv_last = ConvModule(
                out_channels // 2,
                out_channels // 2,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, x):
        # 完全按照原始STDC的逻辑
        out = [self.layers[0](x)]
        
        # 处理后续层
        for i in range(1, len(self.layers)):
            if self.with_downsample and i == 1:
                # 第二层需要先下采样第一层的输出
                if self.fusion_type == 'cat':
                    x_down = self.downsample(out[0])
                    out.append(self.layers[i](x_down))
                else:  # add模式
                    out.append(self.layers[i](out[0]))
            else:
                # 其他层处理前一层的输出
                out.append(self.layers[i](out[i - 1]))
        
        # 融合
        if self.fusion_type == 'add':
            # add模式：逐元素相加
            out_sum = out[1]
            for i in range(2, len(out)):
                out_sum = out_sum + out[i]
            
            if self.with_downsample:
                out_sum = out_sum + self.skip(x)
            
            return self.conv_last(out_sum)
        else:
            # cat模式：拼接
            if self.with_downsample:
                # 有下采样：拼接out[1:]（跳过第一个未下采样的）
                out_cat = torch.cat(out[1:], dim=1)
            else:
                # 无下采样：拼接所有
                out_cat = torch.cat(out, dim=1)
            
            return self.conv_last(out_cat)


class GRNWrapper(nn.Module):
    """GRN层的包装器"""
    def __init__(self, dim):
        super().__init__()
        self.grn = GRN(dim)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.grn(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtV2EnhancedBlock(BaseModule):
    """使用完整ConvNeXt V2 Block增强的特征提取模块"""
    def __init__(self, dim, drop_path=0., num_blocks=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(dim=dim, drop_path=drop_path) 
            for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@MODELS.register_module()
class STDCNetWithConvNeXtV2(BaseModule):
    """融合ConvNeXt V2模块的STDC网络"""

    arch_settings = {
        'STDCNet1': [(2, 1), (2, 1), (2, 1)],
        'STDCNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self,
                 stdc_type,
                 in_channels,
                 channels,
                 bottleneck_type,
                 norm_cfg,
                 act_cfg,
                 num_convs=4,
                 with_final_conv=False,
                 use_grn=True,
                 use_convnext_blocks=None,
                 drop_path_rate=0.0,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert stdc_type in self.arch_settings
        assert bottleneck_type in ['add', 'cat']
        assert len(channels) == 5

        self.in_channels = in_channels
        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.pretrained = pretrained
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv
        self.use_grn = use_grn
        
        if use_convnext_blocks is None:
            use_convnext_blocks = [False] * len(self.stage_strides)
        self.use_convnext_blocks = use_convnext_blocks

        # 前两个浅层stage
        self.stages = ModuleList([
            ConvModule(
                self.in_channels,
                self.channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                self.channels[0],
                self.channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        ])
        
        self.num_shallow_features = len(self.stages)

        # 后续的深层stage
        for idx, strides in enumerate(self.stage_strides):
            stage_idx = len(self.stages) - 1
            
            if self.use_convnext_blocks[idx]:
                stage = self._make_convnext_stage(
                    self.channels[stage_idx], 
                    self.channels[stage_idx + 1],
                    strides, 
                    norm_cfg, 
                    act_cfg, 
                    drop_path_rate)
            else:
                stage = self._make_stdc_stage(
                    self.channels[stage_idx], 
                    self.channels[stage_idx + 1],
                    strides, 
                    norm_cfg, 
                    act_cfg, 
                    bottleneck_type,
                    drop_path_rate)
            
            self.stages.append(stage)

        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1],
                max(1024, self.channels[-1]),
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def _make_stdc_stage(self, in_channels, out_channels, strides, 
                         norm_cfg, act_cfg, bottleneck_type, drop_path_rate):
        """构建STDC stage"""
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModuleWithGRN(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride,
                    norm_cfg,
                    act_cfg,
                    num_convs=self.num_convs,
                    fusion_type=bottleneck_type,
                    use_grn=self.use_grn,
                    drop_path_rate=drop_path_rate))
        return Sequential(*layers)

    def _make_convnext_stage(self, in_channels, out_channels, strides,
                            norm_cfg, act_cfg, drop_path_rate):
        """构建ConvNeXt V2增强的stage"""
        layers = []
        
        for i, stride in enumerate(strides):
            if i == 0:
                layers.append(
                    ConvModule(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                layers.append(
                    ConvNeXtV2EnhancedBlock(
                        dim=out_channels,
                        drop_path=drop_path_rate,
                        num_blocks=1))
        
        return Sequential(*layers)

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        
        # 只返回深层特征
        outs = outs[self.num_shallow_features:]
        return tuple(outs)


@MODELS.register_module()
class STDCContextPathNetWithConvNeXtV2(BaseModule):
    """融合ConvNeXt V2的STDC Context Path网络"""

    def __init__(self,
                 backbone_cfg,
                 last_in_channels=(1024, 512),
                 out_channels=128,
                 ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4),
                 upsample_mode='nearest',
                 align_corners=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone_cfg)
        
        self.arm_list = nn.ModuleList()
        for in_ch in last_in_channels:
            self.arm_list.append(
                AttentionRefinementModule(in_ch, out_channels, norm_cfg))
        
        self.ffm = FeatureFusionModuleWithGRN(**ffm_cfg, norm_cfg=norm_cfg)
        
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        # 骨干网络
        outs = self.backbone(x)
        
        # ARM处理 - 从最后一层开始
        # ARM[0]处理outs[-1] (1024ch) -> 128ch
        avg_feat = self.arm_list[0](outs[-1])
        
        # 如果有多个ARM，处理倒数第二层
        if len(self.arm_list) > 1:
            # 上采样到倒数第二层的尺寸
            avg_feat = resize(
                avg_feat,
                size=outs[-2].shape[2:],
                mode=self.upsample_mode,
                align_corners=self.align_corners)
            
            # ARM[1]处理倒数第二层并融合
            feat_2 = self.arm_list[1](outs[-2])
            avg_feat = avg_feat + feat_2
        
        # 关键修复：将avg_feat上采样到outs[0]的尺寸
        avg_feat = resize(
            avg_feat,
            size=outs[0].shape[2:],
            mode=self.upsample_mode,
            align_corners=self.align_corners)
        
        # 特征融合
        feat_fuse = self.ffm(outs[0], avg_feat)
        
        return tuple([feat_fuse] + list(outs))


class FeatureFusionModuleWithGRN(BaseModule):
    """增强版特征融合模块"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 use_grn=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.use_grn = use_grn
        
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                out_channels,
                channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None), 
            nn.Sigmoid())
        
        if self.use_grn:
            self.grn = GRNWrapper(out_channels)

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        
        if self.use_grn:
            x = self.grn(x)
        
        attn = self.attention(x)
        x_attn = x * attn
        return x_attn + x