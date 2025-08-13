import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.hideLayer = nn.Linear(28*28, 128)
        self.activeLayer = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加Dropout
        self.outputLayer = nn.Linear(128, 10)


    def forward(self, x):
        x = self.hideLayer(x)
        x = self.activeLayer(x)
        x = self.dropout(x)
        x = self.outputLayer(x)
        return x

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(test_set, batch_size=1000)
    model = SimpleNetwork()
    model = model.to(device)

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
            img = torch.flatten(img, 1)

            img = img.to(device)
            target = target.to(device)
            outputs = model(img)
            loss = criterion(outputs, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        cost = time.time() - start
        with torch.no_grad():
            for img, target in test_loader:
                img = torch.flatten(img, 1)
                img = img.to(device)
                target = target.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Test Acc: {100 * correct / total:.2f}% | "
              f"Time cost: {cost} ")

    torch.save(model, "models/simplenn.pth")
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
    model = torch.load("models/simplenn.pth", weights_only=False)
    model = model.to(device)

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, target in test_loader:
            img = torch.flatten(img, 1)
            img = img.to(device)
            target = target.to(device)
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