import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter

def check_label_values(dataset_root, split='train'):
    """
    检查分割标签图像中的像素值分布。
    
    Args:
        dataset_root (str): 数据集根目录路径 (例如 'opsfloodnet')
        split (str): 要检查的子集，默认为 'train' (通常检查训练集就足够了)
    """
    
    # 构建标签文件夹路径
    # 根据你的截图，结构是 opsfloodnet/labels/train
    label_dir = os.path.join(dataset_root, 'labels', split)
    
    if not os.path.exists(label_dir):
        print(f"错误: 找不到路径 {label_dir}")
        return

    print(f"正在扫描目录: {label_dir} ...")
    
    # 获取所有图片文件
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = [f for f in os.listdir(label_dir) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print("未找到任何标签图像文件。")
        return

    print(f"找到 {len(files)} 个标签文件。开始分析...")

    # 用于存储所有出现过的像素值
    unique_values_total = set()
    # 用于统计每个值在多少张图片中出现过（辅助判断）
    value_counts = Counter()

    # 使用 tqdm 显示进度条
    for file_name in tqdm(files):
        file_path = os.path.join(label_dir, file_name)
        
        # 读取图像 (以灰度模式读取，flag=0 或 cv2.IMREAD_UNCHANGED)
        # 注意：对于分割标签，必须确保不进行任何颜色转换
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"警告: 无法读取文件 {file_name}")
            continue
            
        # 如果是多通道图像（例如 RGB 标注），需要特殊处理
        # 这里假设是标准的单通道灰度 Mask (index mask)
        if len(img.shape) > 2:
            # 如果是 3 通道，通常意味着是调色板模式或者 RGB 标注
            # 这里简单取第一个通道，或者你需要根据具体情况修改
            # 很多时候 mmseg 要求的是单通道的 index map
            print(f"警告: 文件 {file_name} 是多通道图像 {img.shape}，将仅使用第一个通道分析。")
            img = img[:, :, 0]

        # 获取当前图片中的唯一值
        unique_in_img = np.unique(img)
        
        # 更新总集合
        unique_values_total.update(unique_in_img)
        
        # 更新计数
        for val in unique_in_img:
            value_counts[val] += 1

    print("\n" + "="*40)
    print("分析完成！结果如下：")
    print("="*40)
    
    sorted_values = sorted(list(unique_values_total))
    print(f"所有标签中检测到的唯一像素值 (Class IDs): {sorted_values}")
    
    print("\n详细统计 (值: 出现该值的文件数):")
    for val in sorted_values:
        print(f"  Class ID {val}: {value_counts[val]} 张图片")

    print("="*40)
    
    # 给出建议
    if len(sorted_values) > 0:
        print("\n[注册建议]")
        print(f"你的 `classes` 列表应该包含 {len(sorted_values)} 个元素（如果不包含忽略索引）。")
        print(f"对应的顺序应该是 ID 为 {sorted_values[0]} 的类名, ID 为 {sorted_values[1]} 的类名, ...")
        
        if 255 in sorted_values:
            print("注意: 检测到值 255。在 MMSegmentation 中，255 通常被用作 'ignore_index'。")
            print("      定义 classes 时通常不需要包含 255 对应的类名。")

# ==========================================
# 配置区域
# ==========================================
if __name__ == '__main__':
    # 请修改这里的路径为你实际的 opsfloodnet 路径
    # 如果脚本就在 opsfloodnet 旁边，可以直接写 'opsfloodnet'
    dataset_path = '/Users/cenzx/Desktop/graduate/code/flood/data/opsfloodnet' 
    
    check_label_values(dataset_path, split='train')