# -*- coding:utf-8 -*-
# @time : 2019.12.02
# @IDE : pycharm
# @author : wangzhebufangqi
# @github : https://github.com/wangzhebufangqi

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import models
import config
import torch.nn.functional as F

train_directory = config.TRAIN_DATASET_DIR
valid_directory = config.VALID_DATASET_DIR
num_epochs = config.NUM_EPOCHS
batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES
learning_rate = config.LEARNING_RATE

def loaddata():
    train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),  # 随机裁剪到256*256
         transforms.RandomRotation(degrees=15),  # 随机旋转
         transforms.RandomHorizontalFlip(),  # 随机水平翻转
         transforms.CenterCrop(size=224),  # 中心裁剪到224*224
         transforms.ToTensor(),  # 转化成张量
         transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
                              [0.229, 0.224, 0.225])
         ])

    val_transforms = transforms.Compose(
        [transforms.Resize((256,256)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    train_data_size = len(train_datasets)#训练数据总数

    val_datasets = datasets.ImageFolder(valid_directory, transform=val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True)
    valid_data_size = len(val_datasets)#验证数据总数

    print(train_data_size,valid_data_size)
    return train_data_size,train_dataloader,valid_data_size,val_dataloader


def ResNet50():
    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    fc_inputs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, num_classes)
    )
    return net

def GoogleNet():
    net = models.googlenet(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    fc_inputs = net.fc.in_features
    net.fc = nn.Linear(fc_inputs, num_classes)
        
    return net

class DenseNet121(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(DenseNet121, self).__init__()
        net = models.densenet121(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        fc_inputs = net.classifier.in_features
        self.features = net.features#必须是net.features不能是net
        self.classifier = nn.Sequential(
            nn.Linear(fc_inputs, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class VGGNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        self.features = net.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_and_valid(model,train_data_size,train_dataloader,valid_data_size,val_dataloader,loss_function, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#若有gpu可用则用gpu
    model.to(device)
    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):#训练num_epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, num_epochs))

        model.train()#训练

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            # 记得清零
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()#验证

            for j, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  :#记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
    print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    return model, record


if __name__=='__main__':
    model = ResNet50()
    # model = VGGNet()
    # model = DenseNet121()
    # model = GoogleNet()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_data_size,train_dataloader,valid_data_size,val_dataloader = loaddata()
    trained_model, record = train_and_valid(model,train_data_size,train_dataloader,valid_data_size,val_dataloader,loss_func, optimizer)
    torch.save(trained_model, config.TRAINED_MODEL)#保存模型

    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig('loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy.png')
    plt.show()
