import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from models.vgg import vgg16_bn
from models.resnet import resnet50
from models.densenet import densenet121
from models.googlenet import googlenet
import matplotlib.pyplot as plt
import argparse
from tensorboardX import SummaryWriter
from util import log
from tqdm import tqdm
import os

parser=argparse.ArgumentParser()
parser.add_argument('--epochs',default=200)
parser.add_argument('--bs',default=128,help='batch size',type=int)
parser.add_argument('--lr',default=0.01,help='learning rate of optimizer',type=float)
parser.add_argument('--wd',default=0.0001,help='weight decay of optimizer')
parser.add_argument('--seed',default=41,help='',type=int)
parser.add_argument('--num_class',default=102,help='caltech101 ',type=int)
parser.add_argument('--model_type',default='densenet121',help='the choice of model option like:vgg16_bn, resnet50,googlenet')
parser.add_argument('--schedule',default=[100])
parser.add_argument('--gamma',default=0.01)
parser.add_argument('--log',default='./results/new')
parser.add_argument('--log_interval',default=30)
parser.add_argument('--gpu_id',default='1',help='')

args=parser.parse_args()
state={k:v for k,v in args._get_kwargs()}


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
path=args.log+args.model_type+'_result.txt'
log(path,f'epochs:{args.epochs}')

writer=SummaryWriter(logdir='graphs',comment=args.model_type)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    device='cuda'
    torch.cuda.manual_seed(args.seed)
else:
    device='cpu'
device=torch.device(device)

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
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dir = '/home/tongxueqing/tong/quantization_/train'
    train_datasets = datasets.ImageFolder(train_dir, transform=image_transforms['train'])
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.bs, shuffle=True,num_workers=32)

    val_dir = '/home/tongxueqing/tong/quantization_/test'
    val_datasets = datasets.ImageFolder(val_dir, transform=image_transforms['valid'])
    # print(val_datasets)
    # print(len(val_datasets))
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=args.bs, shuffle=False,num_workers=32)

    return train_datasets,train_dataloader,val_datasets,val_dataloader
    
    
# --------------------训练过程---------------------------------
def train_evaluation(train_datasets,train_dataloader,val_datasets,val_dataloader,model,epochs,model_type):

    model = model.to(device)
    test_accu_list = []
    train_accu_list = []
    best=0
    for epoch in range(epochs):
        adjust_lr(optimizer,epoch)
        print('epoch {}'.format(epoch + 1))
        log(path,'epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        model.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            train_loss += loss.item()
            pred = torch.max(output, 1)[1]##需要注意
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(batch_x)
                percentage = 100. * batch_idx / len(train_dataloader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_dataloader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')
        
        writer.add_scalar('train_loss',train_loss / (len(train_datasets)),epoch)
        writer.add_scalar('train_accuracy',train_acc / (len(train_datasets)),epoch)
    
        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                eval_loss += loss.item()
                pred = torch.max(output, 1)[1]##需要注意
                num_correct = (pred == batch_y).sum()
                eval_acc += num_correct.item()
        if eval_acc > best :
            best=eval_acc
            torch.save(model,f'./saves/{model_type}best.pth')
        writer.add_scalar('eval_loss',eval_loss / (len(val_datasets)),epoch)
        writer.add_scalar('eval_accuracy',eval_acc / (len(val_datasets)),epoch)
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_datasets)), train_acc / (len(train_datasets))))
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                val_datasets)), eval_acc / (len(val_datasets))))
        log(path,'Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                train_datasets)), train_acc / (len(train_datasets))))
        log(path,'Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                val_datasets)), eval_acc / (len(val_datasets))))
        train_accu_list.append(train_acc / (len(train_datasets)))
        test_accu_list.append(eval_acc / (len(val_datasets)))
    
    
    return train_accu_list,test_accu_list,model     




if __name__ == "__main__":
    train_datasets,train_dataloader,val_datasets,val_dataloader=loaddata()
    ###choose vgg_bn model 
    log(path,f'{args.epochs}{args.schedule}{args.model_type}')
    if args.model_type=='vgg16_bn':
        model = vgg16_bn(pretrained=True)  # 使用VGG16 网络预训练好的模型
        for parma in model.parameters():  # 设置自动梯度为false#????????
            parma.requires_grad = False

        model.classifier = torch.nn.Sequential(  # 修改全连接层 自动梯度会恢复为默认值
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, args.num_class))
        print('vgg16_bn_new')

    #choose resnet50
    elif args.model_type=='resnet50':
        model = resnet50(pretrained=True)  # 使用VGG16 网络预训练好的模型
        for parma in model.parameters():  # 设置自动梯度为false#????????
            parma.requires_grad = False

        model.fc =  nn.Linear(512 * 4, args.num_class)# 修改全连接层 自动梯度会恢复为默认值
        print('resnet50')

    #choose densenet121
    elif args.model_type=='densenet121':
        model = densenet121(pretrained=True)  # 使用VGG16 网络预训练好的模型
        for parma in model.parameters():  # 设置自动梯度为false#????????
            parma.requires_grad = False

        model.classifier = nn.Linear(1024,args.num_class)# 修改全连接层 自动梯度会恢复为默认值

        print('densenet121')
    elif args.model_type=='googlenet':
        model = googlenet(pretrained=True)  # 使用VGG16 网络预训练好的模型
        # print(model)
        for parma in model.parameters():  # 设置自动梯度为false#????????
            parma.requires_grad = False
        model.fc = nn.Linear(1024,args.num_class)# 修改全连接层 自动梯度会恢复为默认值
        print('googlenet')
    else:
        print('error:no model')

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.wd)#vgg16_bn
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    train_accu_list,test_accu_list,model = train_evaluation(train_datasets,train_dataloader,val_datasets,val_dataloader,model,args.epochs,args.model_type)
    save_path='./saves/'+args.model_type+'.pth'
    torch.save(model,save_path)
    plt.plot(train_accu_list,label='train_accuracy')
    plt.plot(test_accu_list,label='test_accuracy')
    plt.savefig('./r'+args.model_type+'.png')