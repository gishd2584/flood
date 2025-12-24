_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/floodnet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/epoch_100.py'
]
crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
test_dataloader = None
test_cfg = None
test_evaluator = None   