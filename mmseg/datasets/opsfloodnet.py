from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

custom_imports = dict(
    imports=['mmseg.datasets.opsfloodnet'],
    allow_failed_imports=False
)
@DATASETS.register_module()
class OPSFloodNet(BaseSegDataset):

    METAINFO = dict(
        classes=('Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', 'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass'),
        palette = [
            [0, 0, 0],        # 0 Background - black
            [0, 0, 128],      # 1 Building-flooded - dark blue
            [128, 0, 0],      # 2 Building-non-flooded - dark red
            [0, 0, 255],      # 3 Road-flooded - blue
            [255, 0, 0],      # 4 Road-non-flooded - red
            [0, 128, 255],    # 5 Water - light blue
            [0, 128, 0],      # 6 Tree - green
            [255, 255, 0],    # 7 Vehicle - yellow
            [0, 255, 255],    # 8 Pool - cyan
            [128, 255, 128],  # 9 Grass - light green
        ])
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='_lab.png', reduce_zero_label=False, **kwargs)

