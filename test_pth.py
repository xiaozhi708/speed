import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import models
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import sys
import os
from util import log
from tensorboardX import SummaryWriter

parser=argparse.ArgumentParser()
parser.add_argument('--epochs',default=50)
parser.add_argument('--bs',default=16,help='batch size',type=int)
parser.add_argument('--lr',default=0.0002,help='learning rate of optimizer',type=float)
parser.add_argument('--wd',default=0.0001,help='weight decay of optimizer')
parser.add_argument('--seed',default=41,help='',type=int)
parser.add_argument('--num_class',default=102,help='caltech101 ',type=int)
parser.add_argument('--model_type',default='resnet50',help='the choice of model option like:mobilenetv2,shufflenetv2_x1_0，vgg16_bn, resnet50,googlenet,densenet121')
parser.add_argument('--schedule',default=[100],help='到第几轮时降低学习率')
parser.add_argument('--gamma',default=0.01,help='学习率下降的倍数')
parser.add_argument('--log',default='./result_log/new',help='保存日志文件的地址')
parser.add_argument('--bar_interval',default=10,help='过多少轮batch更新一下进度条')
parser.add_argument('--gpu_id',default='1',help='')
parser.add_argument('--train_directory',default='./Caltech/train')
parser.add_argument('--valid_directory',default='./Caltech/val')
parser.add_argument('--save_img_dir',default='./result_img/',help='准确率和损失的曲线图的保存地址')
parser.add_argument('--model_save_dir',default='./trained_models/',help='保存最好模型的地址')
args=parser.parse_args()
state={k:v for k,v in args._get_kwargs()}

# 限定随机数
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

log_path=args.log+args.model_type+'_result.txt'
log(log_path,f'epochs:{args.epochs}')

writer=SummaryWriter(logdir='graphs',comment=args.model_type)

def adjust_lr(optimizer,epoch):
    global state
    if epoch in args.schedule:
        state['lr']*=args.gamma
        for params_group in optimizer.param_groups:
            params_group['lr']=state['lr']
            # print('params_groups:',params_group)

def loaddata():
    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }

    train_datasets = datasets.ImageFolder(args.train_directory, transform=image_transforms['train'])
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.bs, shuffle=True,num_workers=32)
    train_data_size = len(train_datasets)#训练数据总数

    val_datasets = datasets.ImageFolder(args.valid_directory, transform=image_transforms['valid'])
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=args.bs, shuffle=True,num_workers=32)
    valid_data_size = len(val_datasets)#验证数据总数

    print('train_data_size : '+str(train_data_size),'valid_data_size : '+str(valid_data_size))
    return train_data_size,train_dataloader,valid_data_size,val_dataloader

def ResNet50():
    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    fc_inputs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, args.num_class),
        nn.LogSoftmax(dim=1)
    )
    return net
def MobileNetV2():
    net = models.mobilenet_v2(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(net.last_channel, args.num_class),
    )
    return net

def ShuffleNetV2_x1_0():
    net = models.shufflenet_v2_x1_0(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    fc_inputs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 4096),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(4096,2048),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(2048, args.num_class),
        nn.LogSoftmax(dim=1)
    )
    return net


def GoogleNet():
    net = models.googlenet(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    fc_inputs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, args.num_class),
        nn.LogSoftmax(dim=1)
    )
    return net

class DenseNet121(nn.Module):
    def __init__(self, num_classes=args.num_class):
        super(DenseNet121, self).__init__()
        net = models.densenet121(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        fc_inputs = net.classifier.in_features
        self.features = net.features#必须是net.features不能是net
        self.classifier = nn.Sequential(
            nn.Linear(fc_inputs, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class VGG16_BN(nn.Module):
    def __init__(self, num_classes=args.num_class):
        super(VGG16_BN, self).__init__()
        net = models.vgg16_bn(pretrained=True)
        for param in net.parameters():
            param.requires_grad = False
        self.features = net.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 268),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(268, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_and_valid(model,train_data_size,train_dataloader,valid_data_size,val_dataloader,loss_function, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
    model.to(device)

    record = [] #用于保存每轮的训练集损失，验证集损失，训练集准确率，验证集准确率
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        epoch_start = time.time() # 开始时间
        # adjust_lr(optimizer, epoch)

        log(log_path, 'epoch {}'.format(epoch + 1))

        # --------------------------------训练--------------------------------------------
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) # 定义进度条
        for i,(inputs,labels) in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

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

            # 显示进度条
            if i % args.bar_interval == 0:
                done = i * len(inputs)
                percentage = 100. * i / len(train_dataloader)
                pbar.set_description(
                    f'Train Epoch: {epoch} [{done:5}/{len(train_dataloader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        writer.add_scalar('train_loss', avg_train_loss, epoch)
        writer.add_scalar('train_accuracy', avg_train_acc, epoch)

        # --------------------------------验证--------------------------------------------
        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
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

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        writer.add_scalar('valid_loss', avg_valid_loss, epoch)
        writer.add_scalar('valid_accuracy', avg_valid_acc, epoch)

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if avg_valid_acc > best_acc  :#记录最高准确性的模型
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            best_model = model

        epoch_end = time.time() # 结束时间

        log(log_path, 'Train Loss: {:.6f}, Acc: {:.6f}'.format(avg_train_loss, avg_train_acc * 100))
        log(log_path, 'Valid Loss: {:.6f}, Acc: {:.6f}'.format(avg_valid_loss, avg_valid_acc * 100))

        print("Epoch: {:03d}/{:03d}, Training: Loss: {:.6f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.6f}, Accuracy: {:.6f}%, Time: {:.6f}s".format(
                epoch + 1, args.epochs,avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start))
    log(log_path,"Best Accuracy for validation : {:.6f} at epoch {:03d}".format(best_acc, best_epoch))
    print("Best Accuracy for validation : {:.6f} at epoch {:03d}".format(best_acc, best_epoch))

    return best_model, record
def only_valid(model,valid_data_size,val_dataloader,loss_function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #若有gpu可用则用gpu
    model.to(device)
    # --------------------------------验证--------------------------------------------
    model.eval()
    valid_loss = 0.0
    valid_acc = 0.0
    with torch.no_grad():
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

    avg_valid_loss = valid_loss / valid_data_size
    avg_valid_acc = valid_acc / valid_data_size
    print('avg_valid_loss',avg_valid_loss,'avg_valid_acc',avg_valid_acc)
    


def save_acc_loss_image(record):
    record = np.array(record)
    plt.plot(record[:, 0:2])
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(args.save_img_dir+args.model_type+'_loss.png')
    plt.show()

    plt.plot(record[:, 2:4])
    plt.legend(['Train Accuracy', 'Valid Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(args.save_img_dir+args.model_type+'_acc.png')
    plt.show()

if __name__=='__main__':
    # 加载数据
    train_data_size,train_dataloader,valid_data_size,val_dataloader = loaddata()
    # 选择模型
    if args.model_type == 'vgg16_bn':
        model = VGG16_BN()
    elif args.model_type == 'resnet50':
        model = ResNet50()
    elif args.model_type == 'densenet121':
        model = DenseNet121()
    elif args.model_type == 'googlenet':
        model = GoogleNet()
    elif args.model_type == 'shufflenetv2_x1_0':
        model = ShuffleNetV2_x1_0()
    elif args.model_type == 'mobilenetv2':
        model = MobileNetV2()
    else:
        print('error:no model')
        sys.exit()
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    #
    model.load_state_dict(torch.load(args.model_save_dir+'resnet50.pth'))
    # 验证
    only_valid(model,valid_data_size,val_dataloader,loss_func)

    