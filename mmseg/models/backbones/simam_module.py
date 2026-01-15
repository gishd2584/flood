# Copyright (c) OpenMMLab. All rights reserved.
"""SimAM注意力模块
参考论文: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
"""
import torch
import torch.nn as nn


class SimAM(nn.Module):
    """SimAM注意力模块
    
    一个无参数的3D注意力模块，通过能量函数推导注意力权重
    
    Args:
        e_lambda (float): 能量函数中的λ参数，默认1e-4
    """
    
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        return f'{self.__class__.__name__}(lambda={self.e_lambda})'

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 [N, C, H, W]
            
        Returns:
            加权后的特征 [N, C, H, W]
        """
        b, c, h, w = x.size()
        
        # 空间维度的元素数量
        n = w * h - 1
        
        # 计算(t - u)^2，其中u是均值
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # 计算能量函数的倒数
        # E_inv = (x - μ)^2 / (4 * (σ^2 + λ)) + 0.5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 应用sigmoid激活并与原始特征相乘
        return x * self.activaton(y)