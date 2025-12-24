# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.1)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=320000,
        by_epoch=False)
]
# training schedule for 320k
# 修改为按 Epoch 训练
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 指定类型为 EpochBased
    max_epochs=100,              # 设置总共训练多少个 Epoch
    val_interval=5               # 每隔多少个 Epoch 验证一次
)
val_cfg = dict(type='ValLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
