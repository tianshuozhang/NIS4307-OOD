```python
import torch
from oodcls import OodCls

cls = OodCls()  # 定义oodcls类
imgs = torch.randn((3, 1, 28, 28))  # 输入的图片batch
result = cls.classify(imgs=imgs)  # 得到结果
print(result)
```

1.首先，导入 torch 库和 oodcls 类

2.创建 oodcls 实例: cls = oodcIs()，这里cls是导入最终生成的模型，模型存在当前目录中。

3.创建输入图片 imgs : 输入一个n12828的张量此处样例使用 torch.randn 函数创建了3张28*28随机图片

4.进行分类: 调用 ood cls.classify 方法对图像进行分类。

5.输出分类结果: 打印预测的类别结果，输出形式为n维tensor。