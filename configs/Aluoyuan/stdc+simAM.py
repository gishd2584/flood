# STDC消融实验配置文件
# 去掉ConvNeXt V2模块，仅保留SimAM注意力
# 用于洛源洪水分割任务

_base_ = [
    '../_base_/datasets/luoyuanflood.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/epoch_100.py'
]

# 归一化配置
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

# 数据预处理配置
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)

# 模型配置
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    
    # 使用消融版本的STDC骨干网络（去掉ConvNeXt V2）
    backbone=dict(
        type='STDCContextPathNetAblation',
        backbone_cfg=dict(
            type='STDCNetAblation',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False,
            # 注意力配置（仅SimAM）
            use_simam=True,         # 启用SimAM（无参数3D注意力）
            simam_lambda=1e-4,      # SimAM的λ参数
            drop_path_rate=0.1,
        ),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(
            in_channels=384,        # 256(spatial) + 128(context)
            out_channels=256,
            scale_factor=4,
            use_simam=True,         # FFM中也使用SimAM
            simam_lambda=1e-4,
        )
    ),
    
    # 解码头配置
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        num_classes=3,
        in_index=0,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    
    # 辅助头（可选）
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=256,
        num_convs=1,
        num_classes=3,
        in_index=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    
    # 训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# 学习率调度器配置（基于epoch）
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100,
        by_epoch=True)
]

