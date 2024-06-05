import math
import time

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
import cv2
from torch.utils.data import TensorDataset, DataLoader


def Process(Data_type: str, nums, batch_size=32):
    data = pd.read_csv(f'./{Data_type}.txt',
                       sep=' ', header=None)
    file_name = data[0][:nums + 1]
    label = (torch.tensor(data[1], dtype=torch.float) * 20)[:nums + 1]
    dataset = []
    trans = transforms.ToTensor()
    for i in range(len(file_name)):
        im = cv2.imread('./IMG/' + file_name[i])
        im = trans(im).reshape(1, im.shape[2], im.shape[1], im.shape[0])
        im = im.to(torch.float)
        dataset.append(im)
    dataset = torch.cat(dataset, dim=0)
    dataset = TensorDataset(dataset, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def test_one_img(path: str):
    # Trans = transforms.ToTensor()
    im1 = Image.open(path)
    im1=im1.convert('RGB')
    # im1 = Trans(im1).reshape(1, im1.shape[2], im1.shape[1], im1.shape[0])
    transform = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    im1 = transform(im1)
    im1 = im1.reshape(1, 3, 350, 350)
    return im1


import torch.nn as nn
from d2l import torch as d2l


def evaluate_loss(net, data_iter, loss1, loss2):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_model_selection`"""
    metric = d2l.Accumulator(3)  # Sum of losses, no. of examples
    Total_l = 0
    num_batch = len(data_iter)
    for i, (X, y) in enumerate(data_iter):
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        out = net(X)
        y = y.reshape(-1, 1)
        l1 = loss1(out, y)
        l2 = loss2(out, y)
        # Total_l+=l.sum()
        metric.add(d2l.reduce_sum(l1), d2l.reduce_sum(l2), d2l.size(l1))
        print('Evaluating losses:%.2f%%\r' % (((i + 1) / num_batch) * 100), end='')
    print('\n', end='')
    return metric[0] / metric[2], metric[1] / metric[2]


import sys, os
from torcheval.metrics.functional import r2_score


def train(net, train_iter, test_iter, num_epochs: int, learning_rate: float, device: torch.device, model_path,
          is_retrain):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    test_loss = 0
    best = sys.maxsize
    if is_retrain == False:
        net.apply(init_weights)
    print('train on:', device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss = nn.MSELoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        Total_l = 0

        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            Total_l += l.item()
            l.backward()
            optimizer.step()

            print(f'\rEpoch:{epoch + 1} [' + '=' * int(10 * ((i + 1) / num_batches)) + '] %.2f%%' % (
                    ((i + 1) / num_batches) * 100), end='')
        print("\n", end='')
        test_loss, test_r2 = evaluate_loss(net, test_iter, loss1=nn.MSELoss(), loss2=r2_score)
        test_loss = math.sqrt(test_loss)
        train_loss = math.sqrt(Total_l / num_batches)
        end = time.time()
        min = (end - start) / 60
        sec = (end - start) % 60
        if os.path.exists(path=model_path):
            Last_r2 = torch.load(model_path)['R2_score']
            if abs(1 - best) > abs(1 - Last_r2):
                best = Last_r2
        if abs(1 - test_r2) < abs(1 - best):
            best = test_r2
            torch.save({
                'model_state_dict': net.state_dict(),
                'Train_loss': train_loss,
                'Test_loss': test_loss,
                'R2_score': test_r2,
            }, model_path)
        print('Epoch:%d Train Loss:%.4f Test Loss:%.4f R2_score:%.4f Time:%d:%d' %
              (epoch + 1, train_loss,
               test_loss, test_r2, min, sec))


from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open('./IMG/' + self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloader(nums, Data_type: str, batch_size: int):
    transform = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data = pd.read_csv(f'./{Data_type}.txt',
                       sep=' ', header=None)
    file_name = data[0][:nums + 1]
    labels = (torch.tensor(data[1], dtype=torch.float) * 20).view(-1, 1)[:nums + 1]
    dataset = CustomDataset(file_name, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
