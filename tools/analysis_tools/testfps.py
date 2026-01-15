import os
import time
import logging
import argparse
from glob import glob
from datetime import datetime
import torch
from mmseg.apis import init_model, inference_model

def setup_logger(log_file=None):
    """配置日志器"""
    logger = logging.getLogger('fps_benchmark')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file is None:
        log_file = f'fps_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file

def get_image_list(img_path):
    """获取图片列表，支持单张图片或目录"""
    if os.path.isfile(img_path):
        return [img_path]
    elif os.path.isdir(img_path):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_list = []
        for ext in extensions:
            img_list.extend(glob(os.path.join(img_path, ext)))
            img_list.extend(glob(os.path.join(img_path, ext.upper())))
        return sorted(img_list)
    else:
        raise ValueError(f'无效的图片路径: {img_path}')

def benchmark_fps(config_file, checkpoint_file, img_path, 
                  warmup_iters=10, test_iters=100, device='cuda:0', 
                  log_file=None, multi_img=False):
    """测试模型FPS"""
    logger, log_file = setup_logger(log_file)
    
    # 获取图片列表
    img_list = get_image_list(img_path)
    num_images = len(img_list)
    
    logger.info('=' * 60)
    logger.info('FPS Benchmark 开始')
    logger.info('=' * 60)
    logger.info(f'配置文件: {config_file}')
    logger.info(f'模型权重: {checkpoint_file}')
    logger.info(f'图片数量: {num_images}')
    logger.info(f'测试模式: {"多图循环" if multi_img and num_images > 1 else "单图重复"}')
    logger.info(f'设备: {device}')
    logger.info(f'预热次数: {warmup_iters}')
    logger.info(f'测试次数: {test_iters}')
    
    # 加载模型
    logger.info('正在加载模型...')
    model = init_model(config_file, checkpoint_file, device=device)
    logger.info('模型加载完成')
    
    # 预热
    logger.info(f'开始预热 ({warmup_iters} 次)...')
    for i in range(warmup_iters):
        img = img_list[i % num_images] if multi_img else img_list[0]
        inference_model(model, img)
    logger.info('预热完成')
    
    # 测试FPS
    logger.info(f'开始测试 ({test_iters} 次)...')
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(test_iters):
        img = img_list[i % num_images] if multi_img else img_list[0]
        inference_model(model, img)
        if (i + 1) % 20 == 0:
            logger.info(f'进度: {i + 1}/{test_iters}')
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 计算结果
    total_time = end_time - start_time
    fps = test_iters / total_time
    avg_time = total_time / test_iters * 1000
    
    logger.info('=' * 60)
    logger.info('测试结果')
    logger.info('=' * 60)
    logger.info(f'总耗时: {total_time:.4f} 秒')
    logger.info(f'平均每帧: {avg_time:.2f} 毫秒')
    logger.info(f'FPS: {fps:.2f}')
    logger.info('=' * 60)
    logger.info(f'日志已保存至: {log_file}')
    
    return fps, avg_time

def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg FPS Benchmark')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument('--img', default='demo/demo.png', 
                        help='测试图片路径或图片目录')
    parser.add_argument('--warmup', type=int, default=10, help='预热次数')
    parser.add_argument('--iters', type=int, default=100, help='测试次数')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--log-file', default=None, help='日志文件路径')
    parser.add_argument('--multi-img', action='store_true', 
                        help='使用多图循环测试（模拟真实I/O场景）')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    benchmark_fps(
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        img_path=args.img,
        warmup_iters=args.warmup,
        test_iters=args.iters,
        device=args.device,
        log_file=args.log_file,
        multi_img=args.multi_img
    )