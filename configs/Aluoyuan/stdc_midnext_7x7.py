_base_ = [
    '../_base_/datasets/luoyuanflood.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/epoch_100.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=crop_size,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    
    backbone=dict(
        # === 核心修改：调用 7x7 版本的类 ===
        type='STDCContextPathNet_MidNeXt_7x7', 
        
        backbone_cfg=dict(
            stdc_type='STDCNet_Custom', 
            channels=(32, 64, 128, 256, 512), 
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            drop_path_rate=0.1,
        ),
        last_in_channels=(512, 256), 
        out_channels=64, 
        ffm_cfg=dict(
            in_channels=192,        
            out_channels=128,
            use_attn=True,         
        )
    ),
    
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
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
    
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
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

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=100,
        by_epoch=True)
]