_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/luoyuanflood.py',
    '../_base_/default_runtime.py', '../_base_/schedules/epoch_100.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3))

# optimizer
# Segformer commonly uses AdamW
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(type='LinearLR', by_epoch=True, start_factor=0.1, begin=0, end=5),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=5,
        end=100,
        by_epoch=True,
    )
]

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
