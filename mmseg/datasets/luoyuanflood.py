from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LuoYuanFlood(BaseSegDataset):

    METAINFO = dict(
        classes=('Background', 'water', 'flood'),
        palette = [
            [0, 0, 0],        # 0 Background 
            [0, 0, 128],      # 1 water
            [128, 0, 0],      # 2 flood

        ])
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)



