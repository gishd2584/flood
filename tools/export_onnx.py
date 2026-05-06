import torch
import torch.nn as nn
from mmseg.apis import init_model
from mmseg.models.backbones.gcnet import GCNet  # 新增导入

import argparse
import types

def export_onnx(config_path, checkpoint_path, output_file):
    # 1. 初始化模型
    print("正在加载模型...")
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    model.eval()
    # ✅ 修改1：如果是 GCNet，导出前必须先融合多分支结构
    if isinstance(model.backbone, GCNet):
        print("  检测到 GCNet，正在执行 switch_to_deploy()...")
        model.backbone.switch_to_deploy()
        # 验证融合是否成功
        for name, m in model.backbone.named_modules():
            if m.__class__.__name__ == 'GCBlock':
                assert hasattr(m, 'reparam_3x3'), f"{name} 融合失败！"
                assert not hasattr(m, 'path_3x3_1'), f"{name} 仍有训练分支残留！"
        print("  融合验证通过 ✓")
    # 2. 准备虚拟输入
    input_shape = (1, 3, 512, 512)
    dummy_input = torch.randn(input_shape).cuda()

    # 3. 定义新的 forward 函数，只包含核心计算逻辑
    # 这样可以绕过 MMSeg 复杂的 data_sample 包装
    def forward_impl(self, x):
        # 1. Backbone
        feat = self.backbone(x)
        
        # 2. Decode Head (产生预测 logits)
        # MMSeg 1.x 的 decode_head.forward 接收 inputs
        out = self.decode_head.forward(feat)
        
        # 3. Resize (如果输出尺寸不是 512x512，通常需要上采样回原图大小)
        # 注意：resize 在 ONNX 中有时比较慢，如果为了极致速度，
        # 可以只导出到 1/8 或 1/4 大小，在板卡上用 OpenCV 做最后的 resize
        # 这里为了方便，我们加上 resize 逻辑
        out = torch.nn.functional.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return out

    # 替换模型的方法
    model.forward = types.MethodType(forward_impl, model)

    # 4. 导出 ONNX
    print(f"正在导出到 {output_file} ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=['input'],
        output_names=['output'], # 输出即为类别概率图/Logits
        opset_version=11, 
        do_constant_folding=True,
        # 暂时关闭动态轴，先用固定尺寸跑通流程，Jetson 上固定尺寸性能最好
        # dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} 
    )
    print("导出成功！")

if __name__ == '__main__':
    # 请确认这里路径正确
    # exList = [
    #      ('/root/autodl-fs/luoyuan/bisenetv1/bisenetv1_fcn_4xb8-100epoch_luoyuanflood-512x512.py',
    #       '/root/autodl-fs/luoyuan/bisenetv1/best_mIoU_epoch_100.pth',
    #       'bisenetv1.onnx'),
    #       ('/root/autodl-fs/luoyuan/bisenetv2/bisenetv2_fcn_4xb8-100epoch_luoyuanflood-512x512.py',
    #        '/root/autodl-fs/luoyuan/bisenetv2/best_mIoU_epoch_100.pth',
    #        'bisenetv2.onnx'),
    #        ('/root/autodl-fs/luoyuan/ddrnet/ddrnet_23_in1k-pre_4xb8-100epoch_luoyuanflood-512x512.py',
    #         '/root/autodl-fs/luoyuan/ddrnet/best_mIoU_epoch_85.pth',
    #           'ddrnet.onnx'),
    #         ('/root/autodl-fs/luoyuan/gcnet/gcnet-s_4xb8-100epoch_opsfloodnet-512x512.py',
    #          '/root/autodl-fs/luoyuan/gcnet/best_mIoU_epoch_100.pth',
    #          'gcnet.onnx'),

             
    #          ('/root/autodl-fs/luoyuan/pidnet/pidnet-s_2xb8-100epoch_1024x1024-opsfloodnet-per.py',
    #             '/root/autodl-fs/luoyuan/pidnet/best_mIoU_epoch_100.pth',
    #           'pidnet.onnx'),
    #           ('/root/autodl-fs/luoyuan/stdc/stdc1_4xb8-100epoch_luyuanflood-512x512.py',
    #            '/root/autodl-fs/luoyuan/stdc/best_mIoU_epoch_95.pth',
    #            'stdc.onnx'),
    #            ('/root/autodl-fs/luoyuan/stdc+convnextv2/Stdc+convnextv2.py',
    #             '/root/autodl-fs/luoyuan/stdc+convnextv2/best_mIoU_epoch_100.pth',
    #             'stdc+convnextv2.onnx'),
    #             ('/root/autodl-fs/luoyuan/stdc+convnextv2+simAM/stdc+convnextv2+simAM.py',
    #              '/root/autodl-fs/luoyuan/stdc+convnextv2+simAM/best_mIoU_epoch_100.pth',
    #              'stdc+convnextv2+simAM.onnx'),
    #              ('/root/autodl-fs/luoyuan/stdc+simAM/stdc+simAM.py',
    #               '/root/autodl-fs/luoyuan/stdc+simAM/best_mIoU_epoch_100.pth',
    #               'stdc+simAM.onnx'),
    #               ('/root/autodl-fs/luoyuan/stdc+simAM_lite/stdc+simAM_lite.py',
    #                '/root/autodl-fs/luoyuan/stdc+simAM_lite/best_mIoU_epoch_100.pth',
    #                'stdc+simAM_lite.onnx')
    # ]

    exList = [
        
       
            ('/root/autodl-fs/luoyuan/swin/swin-tiny-patch4-window7_upernet_8xb2-100epoch_luoyuanflood-512x512.py',
             '/root/autodl-fs/luoyuan/swin/best_mIoU_epoch_65.pth',
             'swin-tiny.onnx'),
         
    ]
    
    for config, checkpoint, output in exList:
        export_onnx(config, checkpoint, output)
