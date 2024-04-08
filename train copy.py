import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse

from model import get_model
from utils import all_seed
from dataset import MyDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='shuffle_net', help='model type')
parser.add_argument('--train_epoch', type=int, default=10, help='training epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

all_seed(1234)
print('Set seed to 1234')

transform_minst = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
])

transform_cifar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
])

# train = MyDataset()

id_train = MNIST('./data', train=True, download=True, transform=transform_minst)
odd_train = CIFAR10('./data', train=True, download=True, transform=transform_cifar)

dataloader_args = dict(shuffle=True, batch_size=128, num_workers=8, pin_memory=True, drop_last=False)
id_train_loader = dataloader.DataLoader(id_train, **dataloader_args)
odd_train_loader = dataloader.DataLoader(odd_train, **dataloader_args)

# dataloader_args = dict(shuffle=True, batch_size=128, num_workers=8, pin_memory=True, drop_last=False)
# train_loader = dataloader.DataLoader(train, **dataloader_args)

print('[Train]')
print(' - Numpy Shape:', len(id_train))
print(' - Tensor Shape:', id_train[0][0].size())

model = get_model(args.model_type)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

writer = SummaryWriter()

model.train()
for epoch in range(args.train_epoch):
    for batch_idx, (data, target) in enumerate(id_train_loader):

        y_pred = model(data.cuda())

        loss = F.cross_entropy(y_pred, target.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('training_loss', loss, epoch * len(id_train_loader) + batch_idx)
        if batch_idx % 50 == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data), 
                len(id_train_loader.dataset),
                100. * batch_idx / len(id_train_loader), 
                loss), 
                end='')
    print()
    torch.save(model.state_dict(), f'./models/{args.model_type}_{epoch}.pt')

writer.close()
