import random
import torch
import torchvision
import torchvision.transforms as transforms

# Cifar10データセットをdownlaod_pathにダウンロード
def load_cifar10(download_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True, transform=transform)
    return train, test
