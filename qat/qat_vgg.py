import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
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

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
cfg=cfgs['D']
class VGG16_bn(nn.Module):
    def __init__(self, num_classes=102, init_weights=True):
        super(VGG16_bn, self).__init__()
        self.features = make_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 268),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(268, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.quant=QuantStub()
        self.dequant=DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            # print(m,type(m))
            if type(m) == VGG16_bn:
                for idx in range(len(m.features)):
                    if type(m.features[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.features, [str(idx), str(idx + 1),str(idx + 2)], inplace=True)                   
                torch.quantization.fuse_modules(m.classifier, ['0', '1',], inplace=True)
                torch.quantization.fuse_modules(m.classifier, ['3', '4',], inplace=True)
                



def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


    

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
    model = VGG16_bn()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

######################################################################

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

# data_path = '/home/tongxueqing/tong/tutorials/advanced_source/data/imagenet_1k'
saved_model_dir = '/home/tong/speed/trained_models/'
float_model_file = 'vgg16_bn.pth'
scripted_float_model_file = 'vgg16_bn_quantization_scripted.pth'
scripted_quantized_model_file = 'vgg16_bn_quantization_scripted_quantized.pth'
scripted_qat_model_file='vgg16_bn_qat_scripted.pth'
train_directory='/home/tong/speed/Caltech/train'
valid_directory='/home/tong/speed/Caltech/val'
train_batch_size = 30
eval_batch_size = 30


# data_loader, data_loader_test = prepare_data_loaders(data_path)
data_loader, data_loader_test = loaddata(train_directory,valid_directory,train_batch_size,eval_batch_size)
criterion = nn.CrossEntropyLoss()
float_model = load_model(saved_model_dir + float_model_file).to('cpu')

######################################################################
# Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
# while also improving numerical accuracy. While this can be used with any model, this is
# especially common with quantized models.

print('\n float_model.features: Before fusion \n\n', float_model.features[1])
float_model.eval()

# Fuses modules
float_model.fuse_model()

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n float_model.features: After fusion\n\n',float_model.features[1])

######################################################################
# Finally to get a "baseline" accuracy, let's see the accuracy of our un-quantized model
# with fused modules

num_eval_batches = 40

print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

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

num_calibration_batches = 10

myModel = load_model(saved_model_dir + float_model_file).to('cpu')
myModel.eval()

# Fuse Conv, bn and relu
myModel.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
myModel.qconfig = torch.quantization.default_qconfig
print(myModel.qconfig)
torch.quantization.prepare(myModel, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', myModel.classifier)

# Calibrate with the training set
evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(myModel, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.classifier)

print("Size of model after quantization")
print_size_of_model(myModel)

top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

######################################################################
# For this quantized model, we see a significantly lower accuracy of just ~62% on these same 300
# images. Nevertheless, we did reduce the size of our model down to just under 3.6 MB, almost a 4x
# decrease.
#
# In addition, we can significantly improve on the accuracy simply by using a different
# quantization configuration. We repeat the same exercise with the recommended configuration for
# quantizing for x86 architectures. This configuration does the following:
#
# - Quantizes weights on a per-channel basis
# - Uses a histogram observer that collects a histogram of activations and then picks
#   quantization parameters in an optimal manner.
#

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)
evaluate(per_channel_quantized_model,criterion, data_loader, num_calibration_batches)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
print("Size of ptq channel model")
print_size_of_model(per_channel_quantized_model)
torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

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

def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            torch.cuda.synchronize()
            start = time.time()
            output = model(images)
            torch.cuda.synchronize()
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)



def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

######################################################################
# We fuse modules as before

qat_model = load_model(saved_model_dir + float_model_file)
qat_model.fuse_model()

optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

######################################################################
# Finally, ``prepare_qat`` performs the "fake quantization", preparing the model for quantization-aware
# training

torch.quantization.prepare_qat(qat_model, inplace=True)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',qat_model.classifier)

######################################################################
# Training a quantized model with high accuracy requires accurate modeling of numerics at
# inference. For quantization aware training, therefore, we modify the training loop by:
#
# - Switch batch norm to use running mean and variance towards the end of training to better
#   match inference numerics.
# - We also freeze the quantizer parameters (scale and zero-point) and fine tune the weights.

num_train_batches = 20

# Train and check accuracy after each epoch
for nepoch in range(5):
    train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))

print("Size of quantized model")
print_size_of_model(quantized_model)
torch.jit.save(torch.jit.script(quantized_model), saved_model_dir + scripted_qat_model_file)

run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)
run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)
run_benchmark(saved_model_dir + scripted_qat_model_file, data_loader_test)

