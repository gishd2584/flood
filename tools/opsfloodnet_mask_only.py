import os
import os.path as osp
import sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import time

# ──────────────────────────────────────────────────────────────
# OPSFloodNet 类别定义
# ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Background',           # 0
    'Building-flooded',     # 1
    'Building-non-flooded', # 2
    'Road-flooded',         # 3
    'Road-non-flooded',     # 4
    'Water',                # 5
    'Tree',                 # 6
    'Vehicle',              # 7
    'Pool',                 # 8 
    'Grass',                # 9
]

# BGR palette（与 opsfloodnet.py 一致，注意 OpenCV 是 BGR）
PALETTE_BGR = [
    [0,   0,   0  ],  # 0 Background
    [128, 0,   0  ],  # 1 Building-flooded   (BGR of dark-blue)
    [0,   0,   128],  # 2 Building-non-flooded
    [255, 0,   0  ],  # 3 Road-flooded       (BGR of blue)
    [0,   0,   255],  # 4 Road-non-flooded
    [255, 128, 0  ],  # 5 Water              (BGR of light-blue)
    [0,   128, 0  ],  # 6 Tree
    [0,   255, 255],  # 7 Vehicle
    [255, 255, 0  ],  # 8 Pool
    [128, 255, 128],  # 9 Grass
]

# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='OPSFloodNet 多模型彩色可视化掩膜生成脚本'
    )
    parser.add_argument('--work-dir',  type=str, required=True,
                        help='包含各模型文件夹的根目录')
    parser.add_argument('--img-dir',   type=str, required=True,
                        help='待推理的图片目录')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='可视化的掩膜输出根目录')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='指定模型名称（可选，默认处理全部）')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# 模型发现 & 权重 / 配置查找
# ──────────────────────────────────────────────────────────────
def find_models(work_dir, specified=None):
    """扫描工作目录，返回 [(model_name, config_path, checkpoint_path), ...]"""
    found = []

    if not osp.exists(work_dir):
        print(f"[ERROR] Work dir not found: {work_dir}")
        return found

    candidates = specified if specified else os.listdir(work_dir)

    for name in sorted(candidates):
        model_dir = osp.join(work_dir, name)
        if not osp.isdir(model_dir):
            continue

        # 查找配置文件 (.py)
        configs = glob(osp.join(model_dir, '*.py'))
        if not configs:
            print(f"[WARN] {name}: no .py config found, skipped")
            continue
        config_path = configs[0]  # 取第一个

        # 查找权重文件：优先 best_mIoU，其次最新 epoch
        ckpts = glob(osp.join(model_dir, '*.pth'))
        if not ckpts:
            print(f"[WARN] {name}: no .pth checkpoint found, skipped")
            continue

        best_ckpts = [c for c in ckpts if 'best' in osp.basename(c).lower()]
        if best_ckpts:
            def miou_val(p):
                base = osp.basename(p)
                try:
                    parts = base.replace('.pth', '').split('_')
                    return float(parts[-1])
                except Exception:
                    return 0.0
            checkpoint_path = max(best_ckpts, key=miou_val)
        else:
            def epoch_val(p):
                base = osp.basename(p)
                try:
                    return int(''.join(filter(str.isdigit, base.split('_')[-1].replace('.pth', ''))))
                except Exception:
                    return 0
            checkpoint_path = max(ckpts, key=epoch_val)

        found.append((name, config_path, checkpoint_path))
        print(f"[INFO] Found model: {name}")
        print(f"       config:     {osp.basename(config_path)}")
        print(f"       checkpoint: {osp.basename(checkpoint_path)}")

    return found


# ──────────────────────────────────────────────────────────────
# 辅助：掩码转彩色
# ──────────────────────────────────────────────────────────────
def mask_to_color(mask, palette=PALETTE_BGR):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        if i >= len(palette):
            break
        color[mask == i] = c
    return color


# ──────────────────────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────────────────────
def run_inference(model_name, config_path, checkpoint_path,
                  img_dir, output_dir, device='cuda:0'):
    """使用 mmseg InferenceAPI 对 img_dir 下全部图像推理，只保存彩色预测图"""
    
    # 每个模型生成自己独立的输出文件夹
    color_dir = osp.join(output_dir, model_name)
    os.makedirs(color_dir, exist_ok=True)

    from mmseg.apis import init_model, inference_model
    import mmcv

    print(f"\n[{model_name}] Loading model...")
    try:
        model = init_model(config_path, checkpoint_path, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {e}")
        return

    # 收集图像列表
    img_extensions = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob(osp.join(img_dir, ext)))
        img_files.extend(glob(osp.join(img_dir, ext.upper())))
    img_files = sorted(img_files)

    if not img_files:
        print(f"[ERROR] No images found in {img_dir}")
        return

    print(f"[{model_name}] Running inference on {len(img_files)} images...")

    total_time = 0
    success_count = 0
    
    for img_path in tqdm(img_files, desc=f"  {model_name}"):
        img_name = osp.splitext(osp.basename(img_path))[0]

        try:
            start_time = time.time()
            result = inference_model(model, img_path)
            total_time += time.time() - start_time
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        except Exception as e:
            print(f"[WARN] Inference failed for {img_name}: {e}")
            continue

        # 核心逻辑：只保留按原始 PALETTE 映射好的彩色伪彩色掩码
        color_img = mask_to_color(pred_mask, PALETTE_BGR)
        # 因为 PALETTE_BGR 本身就是 BGR 格式，所以无需 cvtColor，直接写入即可
        cv2.imwrite(osp.join(color_dir, f"{img_name}.png"), color_img)
        success_count += 1

    avg_time = total_time / success_count if success_count > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{model_name}] Done. {success_count}/{len(img_files)} images saved to {color_dir}")
    print(f"[{model_name}] Average time: {avg_time*1000:.2f} ms ({fps:.2f} FPS)")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print("OPSFloodNet 仅生成彩色掩膜脚本")
    print("=" * 60)
    print(f"Work dir:    {args.work_dir}")
    print(f"Image dir:   {args.img_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Device:      {args.device}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    models_meta = find_models(args.work_dir, specified=args.models)
    if not models_meta:
        print("[ERROR] No valid models found. Exiting.")
        sys.exit(1)

    print(f"\nTotal models to process: {len(models_meta)}")

    for name, config_path, checkpoint_path in models_meta:
        run_inference(
            model_name=name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            img_dir=args.img_dir,
            output_dir=args.output_dir,
            device=args.device,
        )

if __name__ == '__main__':
    main()
