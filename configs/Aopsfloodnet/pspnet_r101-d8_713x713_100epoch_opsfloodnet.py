_base_ = [
    '../_base_/models/pspnet_r50-d8.py','../_base_/datasets/opsfloodnet.py',
    '../_base_/default_runtime.py','../_base_/schedules/epoch_100.py'
]
crop_size = (713, 713)
data_preprocessor = dict(size=crop_size)
model = dict(backbone=dict(depth=101),
             decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)