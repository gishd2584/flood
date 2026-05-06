import torch
import torch.nn.functional as F

x = torch.randn(1, 1, 16, 16)
pool1 = F.adaptive_avg_pool2d(x, 3)
pool2 = F.avg_pool2d(x, kernel_size=6, stride=5)

print("size 3 match:", torch.allclose(pool1, pool2))

pool3 = F.adaptive_avg_pool2d(x, 6)
# input=16, output=6. stride = floor(16/6) = 2. kernel_size = 16 - 5*2 = 6.
pool4 = F.avg_pool2d(x, kernel_size=6, stride=2)
print("size 6 match:", torch.allclose(pool3, pool4))
