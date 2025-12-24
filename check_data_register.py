from mmseg.registry import DATASETS

# 强制导入你的数据集文件，确保装饰器被执行
# 假设你的数据集文件在 mmseg/datasets/my_dataset.py
# 必须确保这一行能运行，否则注册表里肯定没有
# try:
#     import mmseg.datasets.my_dataset 
# except ImportError:
#     # 如果你直接放在 mmseg/datasets/__init__.py 里导入了，这里可以忽略
#     pass

# 检查
print('FloodNetDataset' in DATASETS)
# 或者获取类对象
print(DATASETS.get('FloodNetDataset'))