import torch
import math
import numpy as np
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from data import data
from model import model

batch_size = 64
# train data load
trainset = data.Data()
train = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
train_epoch = trainset.__len__()/batch_size/100
# validate data load
valdata = data.Data()
val = DataLoader(valdata, batch_size=64, shuffle=True, num_workers=1)

net = model.Model()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range():
    # train 
    for i, (x, y) in enumerate(train):
        y_pred = net(x)
        loss = criterion(y_pred, y)
        criterion.zero_grad()
        loss.backward()
        criterion.step()
        steps = math.ceil(i/2/train_epoch)
        print(f"\r第{epoch}个epoch的进度: |{'='*steps}>{' '*(50-steps)}| {steps*2}%", end="")
    print(f'\n第{epoch}个epoch的损失为: {loss}')
    # save
    if epoch % 5 == 0:
        torch.save(net.state_dict(), f'./checkpoints/minist_{epoch}.pth')
    
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val):
            for i, (x, y) in enumerate(val):
                y_pred = net(x)
                total += y.shape[0]
                correct += (y_pred == y).sum().item()
        acc = correct/total*100
    print(f'训练至第{epoch}个epoch的准确率为{acc}%')