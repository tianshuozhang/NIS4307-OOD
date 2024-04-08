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
import os

from model import get_model
from utils import all_seed

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='shufflenet', help='model type')
parser.add_argument('--epochs', type=int, default=100, help='training epoch number')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
args = parser.parse_args()

all_seed(1234)
print('Set seed to 1234')

transform_minst = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
])

transform_cifar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

id_train = MNIST('./data', train=True, download=True, transform=transform_minst)
odd_train = CIFAR10('./data', train=True, download=True, transform=transform_cifar)

dataloader_args = dict(shuffle=True, batch_size=64, num_workers=0, pin_memory=True, drop_last=True)
id_train_loader = dataloader.DataLoader(id_train, **dataloader_args)
odd_train_loader = dataloader.DataLoader(odd_train, **dataloader_args)

print('[Train]')
print(' - Numpy Shape:', len(id_train))
print(' - Tensor Shape:', id_train[0][0].size())

model = get_model(args.model_type)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(id_train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.lr))


if (not os.path.exists('./models')):
    os.makedirs('./models')

writer = SummaryWriter()
model.cuda()
model.train()
# start at a random point of the outlier dataset; this induces more randomness without obliterating locality
odd_train_loader.dataset.offset = np.random.randint(len(odd_train_loader.dataset))
for epoch in range(args.epochs):
    for i, (id_data, odd_data) in enumerate(zip(id_train_loader, odd_train_loader)):

        data = torch.cat((id_data[0], odd_data[0]), 0).cuda()
        target = id_data[1].cuda()

        x = model(data)

        loss = F.cross_entropy(x[:len(id_data[0])], target)
        # add regularization term to the loss to encourage energy score difference
        Ec_out = -torch.logsumexp(x[len(id_data[0]):], dim=1)
        Ec_in = -torch.logsumexp(x[:len(odd_data[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        writer.add_scalar('training_loss', loss, epoch * len(id_train_loader) + i)
        if i % 20 == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                i * len(data), 
                len(id_train_loader.dataset),
                100. * i / len(id_train_loader), 
                loss), 
                end='')
    print()
    torch.save(model.state_dict(), f'./models/{args.model_type}_{epoch}.pt')

writer.close()
