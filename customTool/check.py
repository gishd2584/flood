import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# ================= 配置区域 (请根据实际情况修改) =================

# 数据集根目录
DATA_ROOT = Path("/Users/cenzx/Desktop/graduate/code/flood/data/guilinflood_split")

# 目录名称配置
IMG_DIR_NAME = 'img_dir'
ANN_DIR_NAME = 'ann_dir'

# 文件后缀配置
# 图片后缀 (例如 .jpg, .png, .tif)
IMG_SUFFIX = '.jpg' 
# 标签后缀 (例如 .png, .tif)
SEG_SUFFIX = '.png' 

# 标签文件名是否有额外后缀？
# 如果图片是 "abc.jpg"，标签是 "abc.png"，则留空字符串 ""
# 如果图片是 "abc.jpg"，标签是 "abc_label.png"，则填 "_label"
SEG_FILENAME_SUFFIX = "" 

# =============================================================

def check_split(split_name):
    print(f"\n{'='*20} 正在检查数据集划分: {split_name} {'='*20}")
    
    img_dir = DATA_ROOT / IMG_DIR_NAME / split_name
    ann_dir = DATA_ROOT / ANN_DIR_NAME / split_name
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"[错误] 目录不存在: {img_dir} 或 {ann_dir}")
        return

    # 获取所有图片文件
    img_files = sorted(list(img_dir.glob(f'*{IMG_SUFFIX}')))
    print(f"找到图片数量: {len(img_files)}")
    
    if len(img_files) == 0:
        print("[警告] 没有找到图片，请检查路径或后缀设置。")
        return

    # 统计变量
    resolutions = set()
    label_values = set()
    mask_channels = set()
    missing_masks = []
    
    # 进度条遍历
    pbar = tqdm(img_files, desc=f"分析 {split_name}")
    for img_path in pbar:
        # 1. 构造对应的 Mask 路径
        img_stem = img_path.stem # 文件名不带后缀
        mask_name = f"{img_stem}{SEG_FILENAME_SUFFIX}{SEG_SUFFIX}"
        mask_path = ann_dir / mask_name
        
        if not mask_path.exists():
            missing_masks.append(mask_name)
            continue
            
        # 2. 读取图片 (仅为了检查分辨率，为了速度可跳过，但建议检查)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n[错误] 无法读取图片: {img_path}")
            continue
        h, w = img.shape[:2]
        resolutions.add((w, h))
        
        # 3. 读取 Mask (关键步骤)
        # cv2.IMREAD_UNCHANGED 确保读取原始值，不进行颜色转换
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"\n[错误] 无法读取标签: {mask_path}")
            continue
            
        # 检查 Mask 维度
        if len(mask.shape) == 2:
            mask_channels.add(1) # 单通道
        else:
            mask_channels.add(mask.shape[2]) # 多通道 (通常是错误的)
            
        # 统计 Mask 中的唯一像素值
        unique_vals = np.unique(mask)
        label_values.update(unique_vals)

    # ================= 报告输出 =================
    print(f"\n--- {split_name} 分析报告 ---")
    
    # 1. 完整性
    if missing_masks:
        print(f"[严重警告] 缺少 {len(missing_masks)} 个对应的标签文件！")
        print(f"示例缺失: {missing_masks[:3]} ...")
    else:
        print("[通过] 所有图片都有对应的标签文件。")
        
    # 2. 分辨率
    print(f"分辨率 (W, H) 分布: {list(resolutions)}")
    if len(resolutions) > 1:
        print("[提示] 数据集中包含多种分辨率的图片。")
        
    # 3. Mask 通道数
    print(f"Mask 通道数: {list(mask_channels)}")
    if 3 in mask_channels:
        print("[严重警告] 检测到 3 通道 (RGB) 的 Mask！")
        print("MMSegmentation 通常需要单通道 (灰度) Mask (shape=[H, W])。")
        print("如果你的 Mask 是黑白的但是存成了 3 通道，请在 Pipeline 中使用 LoadAnnotations(reduce_zero_label=False) 或预处理转为单通道。")

    # 4. 标签值 (最重要)
    sorted_labels = sorted(list(label_values))
    print(f"检测到的标签值 (Pixel Values): {sorted_labels}")
    print(f"总共检测到的类别数 (包含背景/忽略值): {len(sorted_labels)}")
    
    # 智能建议
    if 255 in sorted_labels:
        print("[建议] 检测到值 255。通常 255 在 MMSeg 中被用作 'ignore_index' (不参与 Loss 计算)。")
        real_classes = [x for x in sorted_labels if x != 255]
        print(f"      实际有效类别数可能是: {len(real_classes)} (排除 255)")
    
    if 0 in sorted_labels and 1 in sorted_labels:
        print("[建议] 标签包含 0 和 1。")
        print("      如果 0 是背景，1 是目标，num_classes 应设为 2。")
        print("      如果 0 是背景且你想忽略它，可以使用 reduce_zero_label=True (但这会将 0 变成 255，1 变成 0)。")

if __name__ == "__main__":
    if not DATA_ROOT.exists():
        print(f"根目录不存在: {DATA_ROOT}")
    else:
        check_split('train')
        check_split('val')