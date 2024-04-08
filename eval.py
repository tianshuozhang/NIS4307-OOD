import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader

from torchvision.datasets import MNIST, CIFAR10, SVHN
from torchvision import transforms

from model import get_model

import matplotlib.pyplot as plt

import oodcls

def get_ood_score(out, T = 1.):
    with torch.no_grad():
        return -T * torch.logsumexp(out / T, dim=1).cpu().numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
]))

test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
]))

# test = MNIST('./data', train=False, download=True, transform=None)

dataloader_args = dict(shuffle=True, batch_size=64, num_workers=0, pin_memory=True)
test_loader = dataloader.DataLoader(test, **dataloader_args)

# print('[Test]')
# print(' - Numpy Shape:', len(test))
# print(' - Tensor Shape:', test[0][0].size())

model = get_model('shufflenet').to(device)
model.load_state_dict(torch.load('./models/shufflenet_2.pt', map_location=device))
model.eval()

cls = oodcls.OodCls()

correct = tot = 0
ood_scores = []
for batch_idx, (data, target) in enumerate(test_loader):
    
    # score = model(data.to(device))
    # pred = score.max(1)[1]
    
    # energy_score = get_ood_score(score)
    # ood_scores += energy_score.tolist()

    # pred[energy_score > -20.] = 10
    pred = cls.classify(data)

    d = pred.eq(target.to(device))
    correct += d.sum().item()
    tot += d.size()[0]

    if batch_idx % 20 == 1:
        print('\r Test: [{}/{} ({:.0f}%)]'.format(
            batch_idx * len(data), 
            len(test_loader.dataset),
            100. * batch_idx / len(test_loader)),
            end='')
print()

print('Accuracy:', correct / tot)

################################################################

test = SVHN('./data', split='test', download=True, transform=transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,)),
]))

# print('[Test]')
# print(' - Numpy Shape:', len(test))
# print(' - Tensor Shape:', test[0][0].size())

dataloader_args = dict(shuffle=True, batch_size=64, num_workers=0, pin_memory=True)
test_loader = dataloader.DataLoader(test, **dataloader_args)
correct = tot = 0
ood_scores1 = []
for batch_idx, (data, target) in enumerate(test_loader):
    
    # score = model(data.to(device))
    # pred = score.max(1)[1]
    #
    # energy_score = get_ood_score(score)
    # ood_scores1 += energy_score.tolist()
    #
    # pred[energy_score > -20.] = 10
    pred = cls.classify(data)

    d = pred.eq(target.to(device))
    correct += d.sum().item()
    tot += d.size()[0]


    if batch_idx % 20 == 1:
        print('\r Test: [{}/{} ({:.0f}%)]'.format(
            batch_idx * len(data), 
            len(test_loader.dataset),
            100. * batch_idx / len(test_loader)),
            end='')
print()

print('Accuracy:', correct / tot)


# draw
import numpy as np
import matplotlib.pyplot as plt

# 绘制频率分布直方图
plt.hist([ood_scores, ood_scores1], bins=50, density=True, alpha=0.7, color=['blue', 'green'])

# 添加标题和标签
plt.title('Frequency Distribution Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图形
plt.show()