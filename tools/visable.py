import numpy as np
import torch
import mmcv
from mmseg.apis import init_model,inference_model 
from mmengine.visualization import Visualizer
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import os   

def preprocess_image(img, mean, std):#图片预处理
    preprocessing = Compose([        #这里可以根据自己pipeline自行修改
        ToTensor(),
        Normalize(mean=mean, std=std)
     ])
    return preprocessing(img.copy()).unsqueeze(0)

config = '/root/autodl-fs/luoyuan/gcnet/gcnet-s_4xb8-100epoch_opsfloodnet-512x512.py'     #模型配置文件
checkpoint = '/root/autodl-fs/luoyuan/gcnet/best_mIoU_epoch_100.pth'  #模型checkpoint.pth文件
# modelList = [ ('/root/autodl-fs/luoyuan/bisenetv1/bisenetv1_fcn_4xb8-100epoch_luoyuanflood-512x512.py',
#           '/root/autodl-fs/luoyuan/bisenetv1/best_mIoU_epoch_100.pth'),
#           ('/root/autodl-fs/luoyuan/bisenetv2/bisenetv2_fcn_4xb8-100epoch_luoyuanflood-512x512.py',
#            '/root/autodl-fs/luoyuan/bisenetv2/best_mIoU_epoch_100.pth'),
#            ('/root/autodl-fs/luoyuan/ddrnet/ddrnet_23_in1k-pre_4xb8-100epoch_luoyuanflood-512x512.py',
#             '/root/autodl-fs/luoyuan/ddrnet/best_mIoU_epoch_85.pth'),
#             ('/root/autodl-fs/luoyuan/gcnet/gcnet-s_4xb8-100epoch_opsfloodnet-512x512.py',
#              '/root/autodl-fs/luoyuan/gcnet/best_mIoU_epoch_100.pth'),  
#              ('/root/autodl-fs/luoyuan/pidnet/pidnet-s_2xb8-100epoch_1024x1024-opsfloodnet-per.py',
#                 '/root/autodl-fs/luoyuan/pidnet/best_mIoU_epoch_100.pth'),
#               ('/root/autodl-fs/luoyuan/stdc/stdc1_4xb8-100epoch_luyuanflood-512x512.py',
#                '/root/autodl-fs/luoyuan/stdc/best_mIoU_epoch_95.pth'),
#                ('/root/autodl-fs/luoyuan/stdc+convnextv2/Stdc+convnextv2.py',
#                 '/root/autodl-fs/luoyuan/stdc+convnextv2/best_mIoU_epoch_100.pth'),
#                 ('/root/autodl-fs/luoyuan/stdc+convnextv2+simAM/stdc+convnextv2+simAM.py',
#                  '/root/autodl-fs/luoyuan/stdc+convnextv2+simAM/best_mIoU_epoch_100.pth'),
#                  ('/root/autodl-fs/luoyuan/stdc+simAM/stdc+simAM.py',
#                   '/root/autodl-fs/luoyuan/stdc+simAM/best_mIoU_epoch_100.pth'),
#                   ('/root/autodl-fs/luoyuan/stdc+simAM_lite/stdc+simAM_lite.py',
#                    '/root/autodl-fs/luoyuan/stdc+simAM_lite/best_mIoU_epoch_100.pth'),
#                     ('/root/autodl-fs/luoyuan/mystdclite/stdclite.py',
#                    '/root/autodl-fs/luoyuan/mystdclite/best_mIoU_epoch_100.pth')]  #模型列表
modelList = [ 
              ('/root/autodl-fs/luoyuan/stdc/stdc1_4xb8-100epoch_luyuanflood-512x512.py',
               '/root/autodl-fs/luoyuan/stdc/best_mIoU_epoch_95.pth'),
               ('/root/autodl-fs/luoyuan/stdc+convnextv2/Stdc+convnextv2.py',
                '/root/autodl-fs/luoyuan/stdc+convnextv2/best_mIoU_epoch_100.pth'),
                ('/root/autodl-fs/luoyuan/stdc+convnextv2+simAM/stdc+convnextv2+simAM.py',
                 '/root/autodl-fs/luoyuan/stdc+convnextv2+simAM/best_mIoU_epoch_100.pth'),
                 ('/root/autodl-fs/luoyuan/stdc+simAM/stdc+simAM.py',
                  '/root/autodl-fs/luoyuan/stdc+simAM/best_mIoU_epoch_100.pth'),
                  ('/root/autodl-fs/luoyuan/stdc+simAM_lite/stdc+simAM_lite.py',
                   '/root/autodl-fs/luoyuan/stdc+simAM_lite/best_mIoU_epoch_100.pth'),
                    ('/root/autodl-fs/luoyuan/mystdclite/stdclite.py',
                   '/root/autodl-fs/luoyuan/mystdclite/best_mIoU_epoch_100.pth')]  #模型列表
imgNameList = ['8192_7680.png' ,'12800_13824.png','9216_14848.png','30720_8192.png','30720_10240.png','16896_11264.png','20480_12800.png']  #图片列表
def draw_feature_map(config, checkpoint, imgName):
    # 1. 初始化模型
    model = init_model(config=config, checkpoint=checkpoint, device='cuda:0')

    # 2. 准备图片
    image = mmcv.imread(f'/root/autodl-fs/luoyuan/flood_results/original_images/{imgName}', channel_order='rgb')  #测试图片
    image = mmcv.imresize(image, (512, 512), return_scale=False)
    image_norm = np.float32(image) / 255
    input_tensor = preprocess_image(image_norm,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.to(device='cuda:0')

    # 3. 【核心修改】直接提取特征，不要 monkey patch
    # model.extract_feat 会返回一个 tuple，包含 backbone 输出的多层特征
    with torch.no_grad():
        feat_tuple = model.extract_feat(input_tensor)

    # 4. 查看输出了几层特征，并选择一层进行可视化
    print(f"提取到的特征层数量: {len(feat_tuple)}")
    for i, f in enumerate(feat_tuple):
        print(f"第 {i} 层特征图形状: {f.shape}")


    # [-1] 代表最后一层（语义最丰富，分辨率最低）
    # [0] 代表最浅层（细节最丰富，分辨率最高）
    feat = feat_tuple[-1]
    # 增加一个维度检查，确保万无一失
    if feat.ndim == 4: 
        # 如果形状是 [1, 128, 64, 64]，则去掉 Batch 维度
        feat = feat[0]
    print(f"输入 visualizer 的特征图形状: {feat.shape}") # 应该是 torch.Size([1024, 16, 16])

    # 5. 可视化
    visualizer = Visualizer()
    drawn_img = visualizer.draw_featmap(feat, image, channel_reduction='select_max')
    drawn_img = mmcv.rgb2bgr(drawn_img)
    # --- 修改点 2: 安全的路径处理 ---
    # 原来的 config 变量是 '/root/.../stdc+simAM.py'，直接拼接到路径里会报错
    config_filename = os.path.basename(config).split('.')[0] # 提取文件名 'stdc+simAM'
    save_dir = f'./vis/{config_filename}'

    # 确保文件夹存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'{imgName}.png')

    # 6. 保存结果
    mmcv.imwrite(drawn_img, save_path)
    print(f"特征图已保存至: {save_path}")

for (config, checkpoint) in modelList:
    for imgName in imgNameList:
        draw_feature_map(config, checkpoint, imgName)
