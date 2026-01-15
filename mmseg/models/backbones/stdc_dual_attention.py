# Copyright (c) OpenMMLab. All rights reserved.
"""融合ConvNeXt V2和SimAM的STDC网络
同时使用GRN（全局响应归一化）和SimAM（无参数3D注意力）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
from ..utils import resize

# ConvNeXt V2 和 SimAM 组件
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
    """全局响应归一化层（ConvNeXt V2）"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class SimAM(nn.Module):
    """SimAM注意力模块（无参数3D注意力）"""
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
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


class GRNWrapper(nn.Module):
    """GRN包装器，处理NCHW格式"""
    def __init__(self, dim):
        super().__init__()
        self.grn = GRN(dim)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.grn(x)
        x = x.permute(0, 3, 1, 2)
        return x


class AttentionRefinementModule(BaseModule):
    """注意力精炼模块"""
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels, out_channels, kernel_size=3, padding=1,
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(out_channels, out_channels, kernel_size=1,
                      norm_cfg=norm_cfg, act_cfg=dict(type='Sigmoid')))

    def forward(self, x):
        x = self.conv_layer(x)
        attn = self.attention(x)
        return x * attn


class STDCModuleWithDualAttention(BaseModule):
    """STDC模块 - 融合GRN和SimAM双重注意力
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长
        norm_cfg (dict): 归一化配置
        act_cfg (dict): 激活函数配置
        num_convs (int): 卷积层数量
        fusion_type (str): 融合类型 ('add' or 'cat')
        use_grn (bool): 是否使用GRN
        use_simam (bool): 是否使用SimAM
        simam_lambda (float): SimAM的λ参数
        drop_path_rate (float): DropPath率
        init_cfg (dict): 初始化配置
    """

    def __init__(self, in_channels, out_channels, stride, norm_cfg=None, act_cfg=None,
                 num_convs=4, fusion_type='add', use_grn=True, use_simam=True,
                 simam_lambda=1e-4, drop_path_rate=0.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1 and fusion_type in ['add', 'cat']
        
        self.stride = stride
        self.fusion_type = fusion_type
        self.use_grn = use_grn
        self.use_simam = use_simam
        self.with_downsample = (stride == 2)
        
        self.layers = ModuleList()
        
        # 第一层：1x1卷积
        conv_0 = ConvModule(in_channels, out_channels // 2, kernel_size=1, norm_cfg=norm_cfg)
        
        if self.with_downsample:
            self.downsample = ConvModule(
                out_channels // 2, out_channels // 2, kernel_size=3, stride=2,
                padding=1, groups=out_channels // 2, norm_cfg=norm_cfg, act_cfg=None)
            
            if fusion_type == 'add':
                self.layers.append(Sequential(conv_0, self.downsample))
                self.skip = Sequential(
                    ConvModule(in_channels, out_channels // 2, kernel_size=3, stride=2,
                              padding=1, groups=in_channels, norm_cfg=norm_cfg, act_cfg=None),
                    ConvModule(out_channels // 2, out_channels // 2, kernel_size=1,
                              norm_cfg=norm_cfg, act_cfg=None))
            else:
                self.layers.append(conv_0)
        else:
            self.layers.append(conv_0)
        
        # 3x3卷积层 + 双重注意力
        for i in range(1, num_convs):
            in_ch = out_channels // 2 if i == 1 else out_channels // 4
            out_ch = out_channels // 4
            
            conv = ConvModule(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                            norm_cfg=norm_cfg, act_cfg=act_cfg)
            
            # 构建注意力序列：Conv -> GRN -> SimAM
            modules = [conv]
            if self.use_grn:
                modules.append(GRNWrapper(out_ch))
            if self.use_simam:
                modules.append(SimAM(e_lambda=simam_lambda))
            
            self.layers.append(Sequential(*modules))
        
        # 输出卷积
        if fusion_type == 'cat':
            if self.with_downsample:
                concat_channels = out_channels // 4 * (num_convs - 1)
            else:
                concat_channels = out_channels // 2 + out_channels // 4 * (num_convs - 1)
            self.conv_last = ConvModule(concat_channels, out_channels, kernel_size=1,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.conv_last = ConvModule(out_channels // 2, out_channels // 2, kernel_size=1,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        out = [self.layers[0](x)]
        
        for i in range(1, len(self.layers)):
            if self.with_downsample and i == 1:
                if self.fusion_type == 'cat':
                    x_down = self.downsample(out[0])
                    out.append(self.layers[i](x_down))
                else:
                    out.append(self.layers[i](out[0]))
            else:
                out.append(self.layers[i](out[i - 1]))
        
        if self.fusion_type == 'add':
            out_sum = out[1]
            for i in range(2, len(out)):
                out_sum = out_sum + out[i]
            if self.with_downsample:
                out_sum = out_sum + self.skip(x)
            return self.conv_last(out_sum)
        else:
            if self.with_downsample:
                out_cat = torch.cat(out[1:], dim=1)
            else:
                out_cat = torch.cat(out, dim=1)
            return self.conv_last(out_cat)


class ConvNeXtV2EnhancedBlock(BaseModule):
    """ConvNeXt V2增强模块"""
    def __init__(self, dim, drop_path=0., num_blocks=1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(dim=dim, drop_path=drop_path) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@MODELS.register_module()
class STDCNetWithDualAttention(BaseModule):
    """融合ConvNeXt V2和SimAM的STDC网络
    
    同时使用GRN和SimAM两种注意力机制
    
    Args:
        stdc_type (str): 'STDCNet1' or 'STDCNet2'
        in_channels (int): 输入通道数
        channels (tuple[int]): 每个stage的通道数
        bottleneck_type (str): 'add' or 'cat'
        norm_cfg (dict): 归一化配置
        act_cfg (dict): 激活函数配置
        num_convs (int): STDC模块中的卷积数量
        with_final_conv (bool): 是否添加最终卷积
        use_grn (bool): 是否使用GRN
        use_simam (bool): 是否使用SimAM
        simam_lambda (float): SimAM的λ参数
        use_convnext_blocks (list[bool]): 每个stage是否使用ConvNeXt块
        drop_path_rate (float): DropPath率
        pretrained (str): 预训练权重路径
        init_cfg (dict): 初始化配置
    """

    arch_settings = {
        'STDCNet1': [(2, 1), (2, 1), (2, 1)],
        'STDCNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self, stdc_type, in_channels, channels, bottleneck_type, norm_cfg, act_cfg,
                 num_convs=4, with_final_conv=False, use_grn=True, use_simam=True,
                 simam_lambda=1e-4, use_convnext_blocks=None, drop_path_rate=0.0,
                 pretrained=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert stdc_type in self.arch_settings
        assert bottleneck_type in ['add', 'cat']
        assert len(channels) == 5

        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv
        self.use_grn = use_grn
        self.use_simam = use_simam
        self.simam_lambda = simam_lambda
        
        if use_convnext_blocks is None:
            use_convnext_blocks = [False] * len(self.stage_strides)
        self.use_convnext_blocks = use_convnext_blocks

        # 前两个浅层stage
        self.stages = ModuleList([
            ConvModule(in_channels, self.channels[0], kernel_size=3, stride=2,
                      padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(self.channels[0], self.channels[1], kernel_size=3, stride=2,
                      padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        ])
        
        self.num_shallow_features = len(self.stages)

        # 深层stage
        for idx, strides in enumerate(self.stage_strides):
            stage_idx = len(self.stages) - 1
            
            if self.use_convnext_blocks[idx]:
                stage = self._make_convnext_stage(
                    self.channels[stage_idx], self.channels[stage_idx + 1],
                    strides, norm_cfg, act_cfg, drop_path_rate)
            else:
                stage = self._make_stdc_stage(
                    self.channels[stage_idx], self.channels[stage_idx + 1],
                    strides, norm_cfg, act_cfg, bottleneck_type, drop_path_rate)
            
            self.stages.append(stage)

        if self.with_final_conv:
            self.final_conv = ConvModule(
                self.channels[-1], max(1024, self.channels[-1]), 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def _make_stdc_stage(self, in_channels, out_channels, strides, 
                         norm_cfg, act_cfg, bottleneck_type, drop_path_rate):
        layers = []
        for i, stride in enumerate(strides):
            layers.append(
                STDCModuleWithDualAttention(
                    in_channels if i == 0 else out_channels, out_channels, stride,
                    norm_cfg, act_cfg, num_convs=self.num_convs,
                    fusion_type=bottleneck_type, use_grn=self.use_grn,
                    use_simam=self.use_simam, simam_lambda=self.simam_lambda,
                    drop_path_rate=drop_path_rate))
        return Sequential(*layers)

    def _make_convnext_stage(self, in_channels, out_channels, strides,
                            norm_cfg, act_cfg, drop_path_rate):
        layers = []
        for i, stride in enumerate(strides):
            if i == 0:
                layers.append(ConvModule(in_channels, out_channels, kernel_size=3,
                                        stride=stride, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg))
            else:
                layers.append(ConvNeXtV2EnhancedBlock(dim=out_channels, drop_path=drop_path_rate, num_blocks=1))
        return Sequential(*layers)

    def forward(self, x):
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        
        if self.with_final_conv:
            outs[-1] = self.final_conv(outs[-1])
        
        outs = outs[self.num_shallow_features:]
        return tuple(outs)


@MODELS.register_module()
class STDCContextPathNetWithDualAttention(BaseModule):
    """融合双重注意力的STDC Context Path网络"""

    def __init__(self, backbone_cfg, last_in_channels=(1024, 512), out_channels=128,
                 ffm_cfg=None, upsample_mode='nearest', align_corners=None,
                 norm_cfg=dict(type='BN'), init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone_cfg)
        self.out_channels = out_channels
        self.last_in_channels = last_in_channels
        
        # ARM
        self.arm_list = nn.ModuleList()
        for in_ch in last_in_channels:
            self.arm_list.append(
                AttentionRefinementModule(in_ch, out_channels, norm_cfg))
        
        # FFM
        if ffm_cfg is None:
            spatial_ch = backbone_cfg['channels'][2]
            ffm_cfg = dict(in_channels=spatial_ch + out_channels, out_channels=256,
                          scale_factor=4, use_grn=True, use_simam=True)
        
        self.ffm = FeatureFusionModuleWithDualAttention(**ffm_cfg, norm_cfg=norm_cfg)
        
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = self.backbone(x)
        
        # ARM处理
        avg_feat = self.arm_list[0](outs[-1])
        
        if len(self.arm_list) > 1:
            avg_feat = resize(avg_feat, size=outs[-2].shape[2:],
                            mode=self.upsample_mode, align_corners=self.align_corners)
            feat_2 = self.arm_list[1](outs[-2])
            avg_feat = avg_feat + feat_2
        
        # 上采样到outs[0]的尺寸
        avg_feat = resize(avg_feat, size=outs[0].shape[2:],
                        mode=self.upsample_mode, align_corners=self.align_corners)
        
        # FFM
        feat_fuse = self.ffm(outs[0], avg_feat)
        
        return tuple([feat_fuse] + list(outs))


class FeatureFusionModuleWithDualAttention(BaseModule):
    """特征融合模块 - 融合GRN和SimAM"""

    def __init__(self, in_channels, out_channels, scale_factor=4,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),
                 use_grn=True, use_simam=True, simam_lambda=1e-4, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.use_grn = use_grn
        self.use_simam = use_simam
        
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # 双重注意力
        if self.use_grn:
            self.grn = GRNWrapper(out_channels)
        if self.use_simam:
            self.simam = SimAM(e_lambda=simam_lambda)
        
        # 通道注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(out_channels, channels, 1, norm_cfg=None, bias=False, act_cfg=act_cfg),
            ConvModule(channels, out_channels, 1, norm_cfg=None, bias=False, act_cfg=None),
            nn.Sigmoid())

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        
        # 应用双重注意力
        if self.use_grn:
            x = self.grn(x)
        if self.use_simam:
            x = self.simam(x)
        
        # 通道注意力
        attn = self.attention(x)
        x_attn = x * attn
        return x_attn + x