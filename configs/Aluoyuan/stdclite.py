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
# --- 修正后的模型配置 (STDC-MidNeXt-Lite 瘦身版) ---
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    
    backbone=dict(
        # 1. 修改类名为最新的 MidNeXt 上下文路径类
        type='STDCContextPathNet_MidNeXt', 
        
        backbone_cfg=dict(
            # 2. 修改为 MidNeXt 对应的 arch_setting
            # 必须和 stdc_midnext_peak.py 里的 arch_settings key 一致
            # 我们代码里写的是 'STDCNet_Custom'
            stdc_type='STDCNet_Custom', 
            
            # 3. 保持瘦身版通道配置 (核心点)
            channels=(32, 64, 128, 256, 512), 
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            drop_path_rate=0.1,
        ),
        
        # 4. 对应瘦身后的最后两层输入通道 (512, 256)
        last_in_channels=(512, 256), 
        
        # 5. Context Path 输出通道 (瘦身版 64)
        out_channels=64, 
        
        # 6. FFM 配置
        ffm_cfg=dict(
            # in = Stage3(128) + Context(64) = 192
            in_channels=192,        
            out_channels=128,
            # 启用 PeakSimAM
            use_attn=True,         
        )
    ),
    
    # 解码头 (保持瘦身配置)
    decode_head=dict(
        type='FCNHead',
        in_channels=128, # 对应 FFM 输出
        channels=128,    # 中间层也瘦身
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
    
    # 辅助头 (保持瘦身配置)
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256, # 对应 Stage 4
        channels=64,     # 进一步瘦身
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