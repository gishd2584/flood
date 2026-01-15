_base_ = [
    '../_base_/models/stdc.py','../_base_/datasets/luoyuanflood.py',
    '../_base_/default_runtime.py','../_base_/schedules/epoch_100.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
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

