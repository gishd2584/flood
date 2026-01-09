import os
import cv2
import numpy as np
from tqdm import tqdm

def clean_label_values(dataset_root):
    """
    遍历 labels 下的 train, val, test 文件夹，
    将所有大于 9 的像素值强制修改为 0 (Background)。
    """
    
    # 定义需要处理的子文件夹
    sub_dirs = ['train', 'val', 'test']
    
    # 定义有效类别的最大索引 (0-9是有效的，所以最大是9)
    MAX_VALID_CLASS_ID = 9
    TARGET_BACKGROUND_ID = 0
    
    print(f"准备开始清洗数据: {dataset_root}")
    print(f"规则: 所有 > {MAX_VALID_CLASS_ID} 的像素将被重置为 {TARGET_BACKGROUND_ID}")
    print("="*50)

    total_files_fixed = 0
    
    for split in sub_dirs:
        label_dir = os.path.join(dataset_root, 'labels', split)
        
        if not os.path.exists(label_dir):
            print(f"跳过: 目录不存在 {label_dir}")
            continue
            
        print(f"正在处理文件夹: {split} ...")
        
        # 获取图片文件
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        files = [f for f in os.listdir(label_dir) if f.lower().endswith(valid_extensions)]
        
        # 使用 tqdm 显示进度
        for file_name in tqdm(files):
            file_path = os.path.join(label_dir, file_name)
            
            # 1. 读取图像 (保持原样读取，不做颜色转换)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"错误: 无法读取 {file_name}")
                continue
            
            # 2. 检查是否存在异常值
            # 创建一个掩码，找出所有大于 9 的像素位置
            invalid_mask = img > MAX_VALID_CLASS_ID
            
            # 3. 如果存在异常值，进行修改并保存
            if np.any(invalid_mask):
                # 统计有多少个像素是坏的（可选，仅用于日志）
                # invalid_count = np.count_nonzero(invalid_mask)
                
                # 将异常位置的像素值设为 0
                img[invalid_mask] = TARGET_BACKGROUND_ID
                
                # 覆盖保存原文件
                cv2.imwrite(file_path, img)
                
                total_files_fixed += 1
                # 如果你想看具体修了哪些文件，可以取消下面这行的注释
                # tqdm.write(f"已修复: {file_name} (包含异常值)")

    print("="*50)
    print("清洗完成！")
    print(f"共修复了 {total_files_fixed} 个包含错误标签的文件。")
    print("现在你可以重新运行之前的 check 脚本来验证结果了。")

# ==========================================
# 配置区域
# ==========================================
if __name__ == '__main__':
    # 请修改这里的路径为你实际的 opsfloodnet 路径
    dataset_path = '/Users/cenzx/Desktop/graduate/code/flood/data/opsfloodnet'
    
    # 二次确认，防止手滑
    confirm = input(f"警告: 此操作将修改 {dataset_path} 下的文件且不可撤销。\n请确认你已经备份了数据? (输入 y 继续): ")
    if confirm.lower() == 'y':
        clean_label_values(dataset_path)
    else:
        print("操作已取消。")