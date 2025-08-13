import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            # 输入: 1x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x14x14
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x7x7
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x3x3
            nn.Dropout(0.25),
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return x

def train():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"cuda is available: {torch.cuda.is_available()}")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载 MNIST 数据集
    train_set = datasets.MNIST(
        root='datasets',
        train=True,
        download=False,
        transform=transform
    )
    test_set = datasets.MNIST(
        root='datasets',
        train=False,
        download=False,
        transform=transform
    )
    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True
    )
    test_loader = DataLoader(test_set, batch_size=1000)
    model = SimpleCnn()
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        start = time.time()
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)
            outputs = model(img)
            loss = criterion(outputs, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            train_loss += loss.item() * img.size(0)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        cost = time.time() - start
        with torch.no_grad():
            for img, target in test_loader:
                img, target = img.to(device), target.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Test Acc: {100 * correct / total:.2f}% | "
              f"Time cost: {cost} ")

    torch.save(model, "models/simplecnn.pth")
    print("模型已经保存")

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"cuda is available: {torch.cuda.is_available()}")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_set = datasets.MNIST(
        root='datasets',
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(test_set, batch_size=1000)
    model = torch.load("models/simplecnn.pth", weights_only=False)
    model = model.to(device)

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(device), target.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Acc: {100 * correct / total:.2f}% | ")


def main():

    # train()
    test()


if __name__=='__main__':
    main()