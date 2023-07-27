import os
import torch
import torchvision
from torchvision import datasets


class MNISTdata:
    def __init__(self):
        data_path = (os.path.dirname(__file__))
        transform = torchvision.transforms.Compose([
                     torchvision.transforms.Resize(64),
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.5,), (0.5,))
                     ])
        self.dataset = datasets.MNIST(root=data_path, download=True,
                                      transform=transform)

    def train_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

    def test_dataloader(self, batch_size):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)
