import torch
from oodcls import OodCls

cls = OodCls()
imgs = torch.randn((1, 1, 28, 28))
result = cls.classify(imgs=imgs)
print(result)