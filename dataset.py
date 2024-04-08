import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

class MyDataset(data.Dataset):    	
    def __init__(self, size=224):
        self.transform1 = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        ])
        self.transform2 = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)),
        ])
        
        self.id_data = MNIST('./data', train=True, download=True, transform=self.transform1)
        odd_data = CIFAR10('./data', train=True, download=True, transform=self.transform2)
        ood_len = int(0.1 * len(odd_data))
        self.odd_data, _ = data.random_split(odd_data, [ood_len, len(odd_data) - ood_len])

    def __getitem__(self, index):
        if (index < len(self.id_data)):
            img, target = self.id_data.__getitem__(index)
        else:
            img, target = self.odd_data.__getitem__(index - len(self.id_data))[0], 10
        
        return img, target

    def __len__(self):
        return len(self.id_data) + len(self.odd_data)