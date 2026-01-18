"""
STDC Dual Attention 消融实验版本
去掉 ConvNeXt V2 模块，仅保留 SimAM 注意力机制
用于对比实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, Sequential
from mmseg.registry import MODELS
from torch.nn import ModuleList


class LayerNorm(nn.Module):
    """支持两种数据格式的LayerNorm: channels_last (NHWC) 或 channels_first (NCHW)"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SimAM(nn.Module):
    """SimAM: 无参数的3D注意力模块"""
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class DropPath(nn.Module):
    """DropPath正则化"""
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


class STDCModuleAblation(BaseModule):
    """STDC模块消融版本 - 仅使用SimAM注意力
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步长
        norm_cfg (dict): 归一化配置
        act_cfg (dict): 激活函数配置
        num_convs (int): 卷积层数量
        fusion_type (str): 融合类型 ('add' or 'cat')
        use_simam (bool): 是否使用SimAM
        simam_lambda (float): SimAM的λ参数
        drop_path_rate (float): DropPath率
        init_cfg (dict): 初始化配置
    """

    def __init__(self, in_channels, out_channels, stride, norm_cfg=None, act_cfg=None,
                 num_convs=4, fusion_type='add', use_simam=True,
                 simam_lambda=1e-4, drop_path_rate=0.0, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert num_convs > 1 and fusion_type in ['add', 'cat']
        
        self.stride = stride
        self.fusion_type = fusion_type
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
        
        # 3x3卷积层 + SimAM注意力
        for i in range(1, num_convs):
            in_ch = out_channels // 2 if i == 1 else out_channels // 4
            out_ch = out_channels // 4
            
            conv = ConvModule(in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                            norm_cfg=norm_cfg, act_cfg=act_cfg)
            
            # 仅使用SimAM注意力
            if self.use_simam:
                modules = [conv, SimAM(e_lambda=simam_lambda)]
            else:
                modules = [conv]
            
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


@MODELS.register_module()
class STDCNetAblation(BaseModule):
    """STDC网络消融版本 - 去掉ConvNeXt V2模块
    
    仅保留SimAM注意力机制，用于对比实验
    
    Args:
        stdc_type (str): 'STDCNet1' or 'STDCNet2'
        in_channels (int): 输入通道数
        channels (tuple[int]): 每个stage的通道数
        bottleneck_type (str): 'add' or 'cat'
        norm_cfg (dict): 归一化配置
        act_cfg (dict): 激活函数配置
        num_convs (int): STDC模块中的卷积数量
        with_final_conv (bool): 是否添加最终卷积
        use_simam (bool): 是否使用SimAM
        simam_lambda (float): SimAM的λ参数
        drop_path_rate (float): DropPath率
        pretrained (str): 预训练权重路径
        init_cfg (dict): 初始化配置
    """

    arch_settings = {
        'STDCNet1': [(2, 1), (2, 1), (2, 1)],
        'STDCNet2': [(2, 1, 1, 1), (2, 1, 1, 1, 1), (2, 1, 1)]
    }

    def __init__(self, stdc_type, in_channels, channels, bottleneck_type, norm_cfg, act_cfg,
                 num_convs=4, with_final_conv=False, use_simam=True,
                 simam_lambda=1e-4, drop_path_rate=0.0,
                 pretrained=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert stdc_type in self.arch_settings
        assert bottleneck_type in ['add', 'cat']
        assert len(channels) == 5

        self.channels = channels
        self.stage_strides = self.arch_settings[stdc_type]
        self.num_convs = num_convs
        self.with_final_conv = with_final_conv
        self.use_simam = use_simam
        self.simam_lambda = simam_lambda

        # 前两个浅层stage
        self.stages = ModuleList([
            ConvModule(in_channels, self.channels[0], kernel_size=3, stride=2,
                      padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(self.channels[0], self.channels[1], kernel_size=3, stride=2,
                      padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        ])
        
        self.num_shallow_features = len(self.stages)

        # 深层stage - 全部使用STDC模块
        for idx, strides in enumerate(self.stage_strides):
            stage_idx = len(self.stages) - 1
            
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
                STDCModuleAblation(
                    in_channels if i == 0 else out_channels, out_channels, stride,
                    norm_cfg, act_cfg, num_convs=self.num_convs,
                    fusion_type=bottleneck_type, use_simam=self.use_simam,
                    simam_lambda=self.simam_lambda,
                    drop_path_rate=drop_path_rate))
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
class STDCContextPathNetAblation(BaseModule):
    """STDC Context Path网络消融版本"""

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
                          scale_factor=4, use_simam=True)
        
        self.ffm = FeatureFusionModuleAblation(**ffm_cfg, norm_cfg=norm_cfg)
        
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

    def forward(self, x):
        outs = self.backbone(x)
        
        # ARM处理
        avg_feat = self.arm_list[0](outs[-1])
        
        if len(self.arm_list) > 1:
            # 使用 F.interpolate 代替 resize
            avg_feat = F.interpolate(
                avg_feat, 
                size=outs[-2].shape[2:],
                mode=self.upsample_mode, 
                align_corners=self.align_corners
            )
            feat_2 = self.arm_list[1](outs[-2])
            avg_feat = avg_feat + feat_2
        
        # 上采样到outs[0]的尺寸
        avg_feat = F.interpolate(
            avg_feat, 
            size=outs[0].shape[2:],
            mode=self.upsample_mode, 
            align_corners=self.align_corners
        )
        
        # FFM
        feat_fuse = self.ffm(outs[0], avg_feat)
        
        return tuple([feat_fuse] + list(outs))


class FeatureFusionModuleAblation(BaseModule):
    """特征融合模块消融版本 - 仅使用SimAM"""

    def __init__(self, in_channels, out_channels, scale_factor=4,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),
                 use_simam=True, simam_lambda=1e-4, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.use_simam = use_simam
        
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # SimAM注意力
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
        
        # 应用SimAM注意力
        if self.use_simam:
            x = self.simam(x)
        
        # 通道注意力
        attn = self.attention(x)
        x_attn = x * attn
        return x_attn + x