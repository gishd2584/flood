# STDC消融实验配置文件 - 瘦身版 (Slim Version)
# 通过配置减少通道数，无需修改源码
# 预计计算量减少 50% 以上，适合 Jetson NX

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
    
    # --- [修改区域 1] 骨干网络配置 ---
    backbone=dict(
        type='STDCContextPathNetAblation',
        backbone_cfg=dict(
            type='STDCNetAblation',
            stdc_type='STDCNet1',
            in_channels=3,
            # [核心修改]：通道数减半策略
            # 原版: (32, 64, 256, 512, 1024)
            # 修改版: (32, 64, 128, 256, 512) 
            # 解释：前两层保持不变（太小了没必要减），后三层减半（计算量大头）
            channels=(32, 64, 128, 256, 512), 
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False,
            use_simam=True,
            simam_lambda=1e-4,
            drop_path_rate=0.1,
        ),
        # [核心修改]：必须与上面的 channels 对应
        # 原版是最后两层 (1024, 512) -> ARM
        # 现在对应修改后的最后两层 (512, 256)
        last_in_channels=(512, 256), 
        
        # [核心修改]：Context Path 的输出通道也减半
        # 原版: 128 -> 修改版: 64
        out_channels=64, 
        
        # [核心修改]：FFM 融合模块参数
        ffm_cfg=dict(
            # in_channels 计算公式 = Spatial Path通道 + Context Path通道
            # Spatial Path 是 backbone channels 的第3个元素 (idx 2)，现在是 128
            # Context Path 是上面的 out_channels，现在是 64
            # 所以: 128 + 64 = 192
            in_channels=192,        
            
            # FFM 输出通道也减半，原版 256 -> 修改版 128
            out_channels=128,
            scale_factor=4,
            use_simam=True,
            simam_lambda=1e-4,
        )
    ),
    
    # --- [修改区域 2] 解码头配置 ---
    decode_head=dict(
        type='FCNHead',
        # [必须修改]：对应 FFM 的 out_channels
        in_channels=128, 
        
        # [建议修改]：中间层通道数也减半，进一步省算力
        channels=128, 
        
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
    
    # --- [修改区域 3] 辅助头配置 ---
    auxiliary_head=dict(
        type='FCNHead',
        # [必须修改]：对应 Backbone 的 Stage 4 (channels[3])
        # 原版 512 -> 修改版 256
        in_channels=256,
        
        # [建议修改]：辅助头的中间层可以更小，比如 64
        channels=64,
        
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
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# 学习率调度器配置
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100,
        by_epoch=True)
]