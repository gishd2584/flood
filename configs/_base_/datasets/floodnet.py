# dataset settings
dataset_type = 'FloodNetDataset'  # 你注册的类名
data_root = 'data/guilinflood_split' # 请修改为你实际的数据集根目录

# -------------------------------------------------------------------------
# 关键设置：Crop Size
# 4K 图片太大，显存吃不消，必须切块训练。
# 推荐: (512, 512) 或 (768, 768)。
# 如果显存大 (24G+)，可以尝试 (1024, 1024)。
# -------------------------------------------------------------------------
crop_size = (768, 768)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # 1. Resize: 
    # 对于遥感/4K图，通常不建议直接 Resize 到很小，会丢失细节。
    # 这里设置 ratio_range 只是为了做多尺度增强 (0.5倍到2.0倍之间浮动)，
    # 保持 keep_ratio=True 以免变形。
    dict(type='RandomResize', scale=(3840, 2160), ratio_range=(0.5, 2.0), keep_ratio=True),
    
    # 2. RandomCrop (核心):
    # 在大图上随机切出 crop_size 大小的块进行训练
    # cat_max_ratio=0.75 表示如果切出来的块里背景太多，可能会重新切（可选）
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'), # 光照颜色增强，增加鲁棒性
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # 测试时，我们希望保留原分辨率，或者只做轻微缩放
    dict(type='Resize', scale=(3840, 2160), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# -------------------------------------------------------------------------
# Dataloader 配置
# -------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=4,  # 根据你的显存调整，如果 OOM 就改成 2
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', 
            seg_map_path='ann_dir/train'
        ),
        # 假设你的图片后缀是 .png，如果是 .jpg 请修改
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, # 验证集通常 batch_size=1
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', 
            seg_map_path='ann_dir/val'
        ),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# -------------------------------------------------------------------------
# 评估器配置
# -------------------------------------------------------------------------
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator