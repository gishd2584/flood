#!/usr/bin/env python
"""完整诊断STDC+ConvNeXt V2的通道配置"""

import torch
import sys
sys.path.insert(0, '/root/flood')

print("="*70)
print("STDC + ConvNeXt V2 通道配置诊断工具")
print("="*70)

# 测试不同的配置
configs_to_test = [
    {
        'name': '配置1: last_in_channels=(1024, 512)',
        'last_in_channels': (1024, 512),
        'ffm_in_channels': 256 + 128,
    },
    {
        'name': '配置2: last_in_channels=(128, 128)',  
        'last_in_channels': (128, 128),
        'ffm_in_channels': 256 + 128,
    },
    {
        'name': '配置3: last_in_channels=(256, 128)',
        'last_in_channels': (256, 128),
        'ffm_in_channels': 256 + 128,
    },
]

# 尝试构建模型
try:
    from mmseg.registry import MODELS
    from mmengine.config import Config
    
    # 基础配置
    base_config = dict(
        type='STDCNetWithConvNeXtV2',
        stdc_type='STDCNet1',
        in_channels=3,
        channels=(32, 64, 256, 512, 1024),
        bottleneck_type='cat',
        num_convs=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        with_final_conv=False,
        use_grn=True,
        use_convnext_blocks=[False, False, True],
        drop_path_rate=0.1,
    )
    
    print("\n" + "="*70)
    print("步骤1: 测试STDCNet骨干网络")
    print("="*70)
    
    backbone = MODELS.build(base_config)
    backbone.eval()
    
    # 测试输入
    x = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        outs = backbone(x)
    
    print(f"\n✓ STDCNet构建成功")
    print(f"输出特征图数量: {len(outs)}")
    for i, out in enumerate(outs):
        print(f"  outs[{i}]: shape={out.shape}, channels={out.shape[1]}")
    
    # 根据实际输出确定正确配置
    actual_last_channels = (outs[-1].shape[1], outs[-2].shape[1])
    actual_spatial_channels = outs[0].shape[1]
    
    print(f"\n" + "="*70)
    print("推荐配置:")
    print("="*70)
    print(f"last_in_channels = {actual_last_channels}")
    print(f"ffm_cfg = dict(")
    print(f"    in_channels = {actual_spatial_channels} + 128,  # {actual_spatial_channels + 128}")
    print(f"    out_channels = 256,")
    print(f"    scale_factor = 4,")
    print(f"    use_grn = True")
    print(f")")
    
    print(f"\n" + "="*70)
    print("步骤2: 测试完整的STDCContextPathNet")
    print("="*70)
    
    # 构建完整网络
    full_config = dict(
        type='STDCContextPathNetWithConvNeXtV2',
        backbone_cfg=base_config,
        last_in_channels=actual_last_channels,
        out_channels=128,
        ffm_cfg=dict(
            in_channels=actual_spatial_channels + 128,
            out_channels=256,
            scale_factor=4,
            use_grn=True
        )
    )
    
    try:
        full_model = MODELS.build(full_config)
        full_model.eval()
        
        with torch.no_grad():
            full_outs = full_model(x)
        
        print(f"\n✓ 完整模型构建成功!")
        print(f"输出特征图数量: {len(full_outs)}")
        for i, out in enumerate(full_outs):
            print(f"  full_outs[{i}]: shape={out.shape}, channels={out.shape[1]}")
        
        print("\n" + "="*70)
        print("✅ 成功! 使用以下配置:")
        print("="*70)
        print(f"""
backbone=dict(
    type='STDCContextPathNetWithConvNeXtV2',
    backbone_cfg=dict(
        type='STDCNetWithConvNeXtV2',
        stdc_type='STDCNet1',
        in_channels=3,
        channels=(32, 64, 256, 512, 1024),
        bottleneck_type='cat',
        num_convs=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        with_final_conv=False,
        use_grn=True,
        use_convnext_blocks=[False, False, True],
        drop_path_rate=0.1,
    ),
    last_in_channels={actual_last_channels},
    out_channels=128,
    ffm_cfg=dict(
        in_channels={actual_spatial_channels + 128},
        out_channels=256,
        scale_factor=4,
        use_grn=True
    )
)
""")
        
    except Exception as e:
        print(f"\n✗ 完整模型构建失败: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"\n✗ 骨干网络构建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("诊断完成")
print("="*70)