<div align="center">

# 🎯 NIS4307-OOD

**Out-of-distribution (OOD) aware classifier for MNIST**
**面向 MNIST 的分布外（OOD）检测分类器**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Topic](https://img.shields.io/badge/Topic-OOD%20Detection-blueviolet?style=flat-square)

</div>

---

## 📖 Overview · 简介

**English** — Course project for *NIS4307*. The classifier recognises the 10 MNIST digits and also rejects inputs that do not come from the training distribution. It is trained with **outlier exposure**: MNIST is the in-distribution set and CIFAR-10 images serve as outliers. Supported backbones are ShuffleNetV2, ResNet-18, WideResNet and an MLP.

**中文** — *NIS4307* 课程项目。分类器在识别 MNIST 十个数字的同时，能够识别并拒绝不属于训练分布的输入。训练采用 **Outlier Exposure** 思路：以 MNIST 作为分布内数据、CIFAR-10 作为异常样本。可选骨干网络有 ShuffleNetV2、ResNet-18、WideResNet 和 MLP。

## 🚀 Usage · 使用方法

```python
import torch
from oodcls import OodCls

cls = OodCls()                          # load the final model from the current directory
imgs = torch.randn((3, 1, 28, 28))      # a batch of n×1×28×28 images
result = cls.classify(imgs=imgs)        # n-dim tensor of predicted labels
print(result)
```

1. 导入 `torch` 与 `OodCls` 类 · import `torch` and `OodCls`
2. `cls = OodCls()` 加载当前目录下训练好的模型 · load the trained model
3. 构造输入 `imgs`（形状 `n×1×28×28`，示例为 3 张随机图片） · build an `n×1×28×28` input batch
4. 调用 `cls.classify` 进行分类 · call `classify`
5. 输出为 n 维 tensor 的预测类别 · output is an n-dim tensor of labels

## 🏋️ Training · 训练

```bash
python train.py --epochs 100 --lr 1e-3
python eval.py
```

## 📁 Structure · 目录结构

```text
NIS4307-OOD
├── oodcls.py     # OodCls inference wrapper (ShuffleNetV2)
├── model.py      # backbones: MLP / ShuffleNet / ResNet-18 / WideResNet
├── shufflenet.py # modified ShuffleNetV2
├── dataset.py    # MNIST (ID) + CIFAR-10 (OOD) datasets
├── train.py      # outlier-exposure training with cosine LR
├── eval.py       # evaluation
└── test.ipynb    # experiments
```
