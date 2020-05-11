"""
(experimental) Static Quantization with Eager Mode in PyTorch
=========================================================

**Author**: `Raghuraman Krishnamoorthi <https://github.com/raghuramank100>`_

**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_

This tutorial shows how to do post-training static quantization, as well as illustrating
two more advanced techniques - per-channel quantization and quantization-aware training -
to further improve the model's accuracy. Note that quantization is currently only supported
for CPUs, so we will not be utilizing GPUs / CUDA in this tutorial.

By the end of this tutorial, you will see how quantization in PyTorch can result in
significant decreases in model size while increasing speed. Furthermore, you'll see how
to easily apply some advanced quantization techniques shown
`here <https://arxiv.org/abs/1806.08342>`_ so that your quantized models take much less
of an accuracy hit than they would otherwise.

Warning: we use a lot of boilerplate code from other PyTorch repos to, for example,
define the ``MobileNetV2`` model archtecture, define data loaders, and so on. We of course
encourage you to read it; but if you want to get to the quantization features, feel free
to skip to the "4. Post-training static quantization" section.

We'll start by doing the necessary imports:
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
import re
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from models.utils import load_state_dict_from_url

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

######################################################################
# 1. Model architecture
# ---------------------
#
# We first define the MobileNetV2 model architecture, with several notable modifications
# to enable quantization:
#
# - Replacing addition with ``nn.quantized.FloatFunctional``
# - Insert ``QuantStub`` and ``DeQuantStub`` at the beginning and end of the network.
# - Replace ReLU6 with ReLU
#
# Note: this code is taken from
# `here <https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py>`_.

from torch.quantization import QuantStub, DeQuantStub

# class DenseNet121(nn.Module):
#     def __init__(self, num_classes=102):
#         super(DenseNet121, self).__init__()
#         net = models.densenet121(pretrained=True)
#         for param in net.parameters():
#             param.requires_grad = False
#         fc_inputs = net.classifier.in_features
#         self.features = net.features#必须是net.features不能是net
#         self.quant = QuantStub()
#         self.dequant = DeQuantStub()
#         self.classifier = nn.Sequential(
#             nn.Linear(fc_inputs, 1024),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, num_classes)
#         )
#
#     def forward(self, x):
#         x = self.quant(x)
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         out = self.dequant(out)
#         return out
#
#     # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
#     # This operation does not change the numerics
#     def fuse_model(self):
#         for m in self.modules():
#             if type(m) == DenseNet121:
#                 torch.quantization.fuse_modules(m.features, ['conv0', 'norm0', 'relu0'], inplace=True)
#                 torch.quantization.fuse_modules(m.classifier, ['0', '1', ], inplace=True)
#                 torch.quantization.fuse_modules(m.classifier, ['3', '4', ], inplace=True)

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):#本来是*prev_features
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)#本来是*prev_features
        else:
            bottleneck_output = bn_function(*prev_features)#本来是*prev_features
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)#本来是*features
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=102, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # Linear layer
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.quant(x)
        f = self.features(x)
        # f = self.quant(f)
        out = F.relu(f, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.dequant(out)
        return out
    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == DenseNet:
                # RuntimeError: Could not run 'aten::thnn_conv2d_forward'with arguments from the 'QuantizedCPUTensorId' backend.
                # 'aten::thnn_conv2d_forward' is only available for these backends: [CPUTensorId, CUDATensorId, VariableTensorId].
                torch.quantization.fuse_modules(m.features, ['conv0', 'norm0', 'relu0'])#加inplace=True就报上面的错
                torch.quantization.fuse_modules(m.classifier, ['0', '1', ], inplace=True)
                torch.quantization.fuse_modules(m.classifier, ['3', '4', ], inplace=True)

def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
# def load_my_state_dict
def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)
######################################################################
# 2. Helper functions
# -------------------
#
# We next define several helper functions to help with model evaluation. These mostly come from
# `here <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_.

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = densenet121()
    # model = nn.DataParallel(model)
    # cudnn.benchmark = True
    state_dict = torch.load(model_file,map_location='cpu')
    model.load_state_dict(state_dict)
    model.to('cpu')
    # print(model)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

######################################################################
# 3. Define dataset and data loaders
# ----------------------------------

###改动二 把data_loader改一下
def loaddata(train_directory,valid_directory,train_batch_size,eval_batch_size):
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

    train_datasets = datasets.ImageFolder(train_directory, transform=image_transforms['train'])
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True,num_workers=32)
    train_data_size = len(train_datasets)#训练数据总数

    val_datasets = datasets.ImageFolder(valid_directory, transform=image_transforms['valid'])
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=eval_batch_size, shuffle=True,num_workers=32)
    valid_data_size = len(val_datasets)#验证数据总数

    print('train_data_size : '+str(train_data_size),'valid_data_size : '+str(valid_data_size))
    return train_dataloader,val_dataloader

######################################################################
# Next, we'll load in the pre-trained MobileNetV2 model. We provide the URL to download the data from in ``torchvision``
# `here <https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py#L9>`_.

saved_model_dir = '../trained_models/'
float_model_file = 'densenet121.pth'
scripted_float_model_file = 'densenet121_quantization_scripted.pth'
scripted_quantized_model_file = 'densenet121_quantization_scripted_quantized.pth'
train_directory='../Caltech/train'
valid_directory='../Caltech/val'
train_batch_size = 30
eval_batch_size = 30

data_loader, data_loader_test = loaddata(train_directory,valid_directory,train_batch_size,eval_batch_size)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file)#.to('cpu')

######################################################################
# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

print('\n float_model.features: Before fusion \n\n', float_model.features[2])
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n float_model.features: After fusion\n\n',float_model.features[2])

######################################################################
# Finally to get a "baseline" accuracy, let's see the accuracy of our un-quantized model
# with fused modules

num_eval_batches = 40

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
# torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)
# params = float_model.state_dict()
# for k, v in params.items():
#     if 'denseblock' not in k and 'transition' not in k and 'norm5' not in k:
#         print(k)
# print("*×*×*×*×保存前的dict×*×*×*×*×")
torch.save(float_model.state_dict(),saved_model_dir+scripted_float_model_file)
######################################################################
# We see 78% accuracy on 300 images, a solid baseline for ImageNet,
# especially considering our model is just 14.0 MB.
#
# This will be our baseline to compare to. Next, let's try different quantization methods
#
# 4. Post-training static quantization
# ------------------------------------
#
# Post-training static quantization involves not just converting the weights from float to int,
# as in dynamic quantization, but also performing the additional step of first feeding batches
# of data through the network and computing the resulting distributions of the different activations
# (specifically, this is done by inserting `observer` modules at different points that record this
# data). These distributions are then used to determine how the specifically the different activations
# should be quantized at inference time (a simple technique would be to simply divide the entire range
# of activations into 256 levels, but we support more sophisticated methods as well). Importantly,
# this additional step allows us to pass quantized values between operations instead of converting these
# values to floats - and then back to ints - between every operation, resulting in a significant speed-up.

num_calibration_batches = 40

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)#本来是inplace=True的
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print("Size of quantized model")
print_size_of_model(per_channel_quantized_model)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
# torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)
params = per_channel_quantized_model.state_dict()
for k, v in params.items():
    if 'denseblock' not in k and 'transition' not in k and 'norm5' not in k:
        print(k)
print("*×*×*×*×保存前的dict×*×*×*×*×")
torch.save(per_channel_quantized_model.state_dict(),saved_model_dir+scripted_quantized_model_file)
######################################################################
# Changing just this quantization configuration method resulted in an increase
# of the accuracy to over 76%! Still, this is 1-2% worse than the baseline of 78% achieved above.
# So lets try quantization aware training.
#

#####################################################################
# Here, we just perform quantization-aware training for a small number of epochs. Nevertheless,
# quantization-aware training yields an accuracy of over 71% on the entire imagenet dataset,
# which is close to the floating point accuracy of 71.9%.
#
# More on quantization-aware training:
#
# - QAT is a super-set of post training quant techniques that allows for more debugging.
#   For example, we can analyze if the accuracy of the model is limited by weight or activation
#   quantization.
# - We can also simulate the accuracy of a quantized model in floating point since
#   we are using fake-quantization to model the numerics of actual quantized arithmetic.
# - We can mimic post training quantization easily too.
#
# Speedup from quantization
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Finally, let's confirm something we alluded to above: do our quantized models actually perform inference
# faster? Let's test:

def run_benchmark_float(model_file, img_loader):
    elapsed = 0

    model = densenet121()
    state_dict = torch.load(model_file, map_location='cpu')
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段保持，如果有，则需要删除 module.
    for k, v in state_dict.items():
        if '.0.' in k:
            k = k.replace('0.','',1)
        else:
            k = k
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    # model.load_state_dict(state_dict)
    model.eval()
    num_batches = 5
    with torch.no_grad():
        # Run the scripted model on a few batches of images
        for i, (images, target) in enumerate(img_loader):
            if i < num_batches:
                start = time.time()
                output = model(images)
                end = time.time()
                elapsed = elapsed + (end-start)
            else:
                break
        num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

def run_benchmark_quantized(model_file, img_loader):
    elapsed = 0

    model = load_model(saved_model_dir + float_model_file)
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    evaluate(model, criterion, data_loader, num_calibration_batches)
    torch.quantization.convert(model, inplace=True)

    # model.load_state_dict(state_dict)
    model.eval()
    num_batches = 5
    with torch.no_grad():
        # Run the scripted model on a few batches of images
        for i, (images, target) in enumerate(img_loader):
            if i < num_batches:
                start = time.time()
                output = model(images)
                end = time.time()
                elapsed = elapsed + (end-start)
            else:
                break
        num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

print("*×*×*×*×*×*×*float model consume time ×*×**×*×*×*×*×*×*×*×*×*×*×*×*")
run_benchmark_float(saved_model_dir + scripted_float_model_file, data_loader_test)
print("*×*×*×*×*×*×*quantized model consume time ×*×**×*×*×*×*×*×*×*×*×*×*×*×*")
run_benchmark_quantized(saved_model_dir + scripted_quantized_model_file, data_loader_test)

######################################################################
# Running this locally on a MacBook pro yielded 61 ms for the regular model, and
# just 20 ms for the quantized model, illustrating the typical 2-4x speedup
# we see for quantized models compared to floating point ones.
#
# Conclusion
# ----------
#
# In this tutorial, we showed two quantization methods - post-training static quantization,
# and quantization-aware training - describing what they do "under the hood" and how to use
# them in PyTorch.
#
# Thanks for reading! As always, we welcome any feedback, so please create an issue
# `here <https://github.com/pytorch/pytorch/issues>`_ if you have any.
