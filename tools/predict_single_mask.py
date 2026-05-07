import os
import os.path as osp
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
import mmcv
import argparse
import time

# ==================== 调色板定义 ====================
# FloodNet数据集调色板
PALETTE = [
    [0, 0, 0],        # Background - 黑色
    [0, 0, 255],      # Water - 红色
    [255, 255, 255],  # flood - 白色
]

CLASS_NAMES = [
    'Background', 'Water', 'flood'
]

# 支持的图像格式
IMG_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

def parse_args():
    parser = argparse.ArgumentParser(description='Single model inference for segmentation')
    parser.add_argument('--config', type=str, required=True,
                        help='指定模型的配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='指定模型的权重文件(.pth)路径')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入图像目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出可视化掩膜的保存目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='推理设备')
    return parser.parse_args()

def get_image_list(input_dir):
    """获取所有图像文件列表"""
    image_list = []
    for ext in IMG_EXTENSIONS:
        image_list.extend(glob(osp.join(input_dir, ext)))
        image_list.extend(glob(osp.join(input_dir, ext.upper())))
    return sorted(image_list)

def mask_to_color(mask, palette):
    """将单通道掩码转换为彩色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    return color_mask

def main():
    args = parse_args()
    
    print("="*60)
    print("Single Model Inference (Visualization Mask Only)")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    print(f"\nLoading model...")
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 获取图像列表
    image_list = get_image_list(args.input_dir)
    print(f"Found {len(image_list)} images to process.\n")
    
    if len(image_list) == 0:
        return
    
    total_time = 0
    
    # 批量推理
    for img_path in tqdm(image_list, desc="Inferencing"):
        img = mmcv.imread(img_path)
        img_name = osp.splitext(osp.basename(img_path))[0]
        
        # 推理并计时
        start_time = time.time()
        result = inference_model(model, img)
        total_time += time.time() - start_time
        
        # 获取分割掩码
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        
        # 只保存彩色分割图（可视化掩膜）
        color_mask = mask_to_color(pred_mask, PALETTE)
        color_path = osp.join(args.output_dir, f"{img_name}.png")
        cv2.imwrite(color_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    
    # 输出统计信息
    avg_time = total_time / len(image_list)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\nDone!")
    print(f"Average inference time: {avg_time*1000:.2f} ms ({fps:.2f} FPS)")
    print(f"Visual masks saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
