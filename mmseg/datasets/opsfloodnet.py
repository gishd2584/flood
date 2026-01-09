from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FloodNetDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'water', 'flood'),
        palette=[[0, 0, 0], [0, 0, 128], [128, 0, 0]]) # 给背景一个颜色，比如黑色

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)


