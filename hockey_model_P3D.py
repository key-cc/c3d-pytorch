#encoding:utf-8

import torch
import torch.nn as nn
import torch.utils.data as dataf
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import h5py
import datetime
from torchsummary import summary
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet152
from log import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_ids = [0,1,2,3]
logger = get_logger('P3D_hockeyFight.log')

##############################
#         P3D
##############################
class P3D_Block(nn.Module):

    def __init__(self, blockType, inplanes, planes, stride=1):
        super(P3D_Block, self).__init__()
        self.expansion = 4
        self.blockType=blockType
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        if self.blockType=='A':
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride),
                                   padding=(0,1,1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3,1,1), stride=(stride,1,1),
                                    padding=(1,0,0), bias=False)
        elif self.blockType == 'B':
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride,
                                    padding=(0, 1, 1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=stride,
                                    padding=(1, 0, 0), bias=False)
        else:
            self.conv2D = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride,
                                    padding=(0, 1, 1), bias=False)
            self.conv1D = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1,
                                    padding=(1, 0, 0), bias=False)
        self.bn2D = nn.BatchNorm3d(planes)
        self.bn1D = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride

        if self.stride != 1 or inplanes!= planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )
        else:
            self.downsample=None


    def forward(self, x):
        x_branch = self.conv1(x)
        x_branch = self.bn1(x_branch)
        x_branch = self.relu(x_branch)

        if self.blockType=='A':
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch = self.conv1D(x_branch)
            x_branch = self.bn1D(x_branch)
            x_branch = self.relu(x_branch)
        elif self.blockType=='B':
            x_branch2D = self.conv2D(x_branch)
            x_branch2D = self.bn2D(x_branch2D)
            x_branch2D = self.relu(x_branch2D)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch=x_branch1D+x_branch2D
            x_branch=self.relu(x_branch)
        else:
            x_branch = self.conv2D(x_branch)
            x_branch = self.bn2D(x_branch)
            x_branch = self.relu(x_branch)
            x_branch1D = self.conv1D(x_branch)
            x_branch1D = self.bn1D(x_branch1D)
            x_branch=x_branch+x_branch1D
            x_branch=self.relu(x_branch)

        x_branch = self.conv3(x_branch)
        x_branch = self.bn3(x_branch)

        if self.downsample is not None:
            x = self.downsample(x)

        x =x+ x_branch
        x = self.relu(x)
        return x

class P3D (nn.Module):
    # input size: 16 x 160 x 160
    def __init__(self, num_class=2):
        super(P3D, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.conv2 = nn.Sequential(P3D_Block('A',64,64,2),
                                    P3D_Block('B', 64 * self.expansion, 64),
                                    P3D_Block('C', 64 * self.expansion, 64))
        self.conv3 = nn.Sequential(P3D_Block('A', 64 * self.expansion, 128, 2),
                                   P3D_Block('B', 128 * self.expansion, 128),
                                   P3D_Block('C', 128 * self.expansion, 128),
                                   P3D_Block('A', 128 * self.expansion, 128))
        self.conv4 = nn.Sequential(P3D_Block('B', 128 * self.expansion, 256, 2),
                                   P3D_Block('C', 256 * self.expansion, 256),
                                   P3D_Block('A', 256 * self.expansion, 256),
                                   P3D_Block('B', 256 * self.expansion, 256),
                                   P3D_Block('C', 256 * self.expansion, 256),
                                   P3D_Block('A', 256 * self.expansion, 256))
        self.conv5 = nn.Sequential(P3D_Block('B', 256 * self.expansion, 512, 2),
                                   P3D_Block('C', 512 * self.expansion, 512),
                                   P3D_Block('A', 512 * self.expansion, 512))
        self.average_pool=nn.AvgPool3d((1,3,3))
        self.fc=nn.Linear(512 * self.expansion,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.average_pool(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x

# ---------------------------------------------------------------------------------------------

# train
def train():
    
    model.train()

    train_loss_data = 0.
    train_acc = 0.

    starttime = datetime.datetime.now()
    logger.info('start training!')

    
    for batch_idx, train_data in enumerate(train_loader):
        
        # data loading
        train_video, train_label = train_data
        train_video, train_label = train_video.cuda(device_ids[0]), train_label.cuda(device_ids[0])

        train_out = model(train_video)

        # calculate the loss
        train_loss = criterion(train_out, train_label)
        # train_loss = L2loss(train_out,train_label,batch_size)
        train_loss_data += train_loss.item()

        # calculate the accuracy
        train_pred = torch.max(train_out, 1)[1]
        train_correct = (train_pred == train_label).sum()
        train_acc += train_correct.item()

        # update the grad
        optimizer.zero_grad()
        train_loss.backward()
        #optimizer.module.step()
        optimizer.step()

        # Print log
        logger.info('Train Epoch: {}  [{}/{} ({:.0f}%)]     Batch_Loss: {:.6f}       Batch_Acc: {:.3f}%'.format(
            epoch + 1, 
            batch_idx * len(train_video), 
            len(train_dataset),
            100. * batch_idx / len(train_loader),
            train_loss.item() / batch_size,
            100. * train_correct.item() / batch_size
            )
        )

        # for param_lr in optimizer.module.param_groups: 
        #     param_lr['lr'] = param_lr['lr'] * 0.999

        logger.info('-----------------------------------------------------------------------------------------------------------')

    endtime = datetime.datetime.now()
    time = (endtime-starttime).seconds
    logger.info('###############################################################################################################\n')
    logger.info(('Train Epoch: [{}\{}]\ttime: {}s').format(epoch+1,num_epoches,time))
    
    #for param_lr in optimizer.module.param_groups: 
    for param_lr in optimizer.param_groups:
        logger.info('lr_rate: ' + str(param_lr['lr']) + '\n')

    logger.info('Train_Loss: {:.6f}      Train_Acc: {:.3f}%\n'.format(train_loss_data / (len(train_dataset)),
                                                            100. * train_acc / (len(train_dataset))
                                                            )
          )
    logger.info('-----------------------------------------------------------------------------------------------------------')

# test
def test():
    
    model.eval()

    test_loss_data = 0.
    test_acc = 0.

    for test_data in test_loader:
        
        # data loading
        test_video, test_label = test_data
        test_video, test_label = test_video.cuda(device_ids[0]), test_label.cuda(device_ids[0])
        test_out = model(test_video)

        #calculate the loss
        test_loss = criterion(test_out, test_label)
        test_loss_data += test_loss.item()

        # calculate the accuracy
        test_pred = torch.max(test_out, 1)[1]
        test_correct = (test_pred == test_label).sum()
        test_acc += test_correct.item()

    # Log test performance
    logger.info('Test_Loss: {:.6f}      Test_Acc: {:.3f}%\n'.format(test_loss_data / (len(test_dataset)),
                                                            100. * test_acc / (len(test_dataset))
                                                            )
          )
    logger.info('--------------------------------------------------------')

# ---------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------
# training dataset
f = h5py.File('hockey_train.h5','r')
train_video = f['data'][()]
# train_video, train_label = create_train(1000,60,90)

train_video = train_video.transpose((0,2,1,3,4))     
train_label = f['label'][()]

train_video = torch.from_numpy(train_video)
train_label = torch.from_numpy(train_label)

# test dataset
f1 = h5py.File('hockey_test.h5','r')                           
test_video = f1['data'][()]    
test_video = test_video.transpose((0,2,1,3,4))     
test_label = f1['label'][()] 

test_video = torch.from_numpy(test_video)
test_label = torch.from_numpy(test_label)


# ---------------------------------------------------------------------------------------------

# parameters 
batch_size = 50
learning_rate = 1e-4
num_epoches = 200

# ---------------------------------------------------------------------------------------------

# training dataset 
train_dataset = dataf.TensorDataset(train_video, train_label)
train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test dataset
test_dataset = dataf.TensorDataset(test_video, test_label)
test_loader = dataf.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
model = P3D()
#model.apply(weights_init)
model = model.cuda(device_ids[0])
#model = nn.DataParallel(model, device_ids=device_ids)
summary(model,(3,16,160,160))

# loss and optimization 
criterion = nn.CrossEntropyLoss()

# def L2loss(x,y,batchsize):
#     loss_batch = 0
#     for i in range(0, batchsize):
#         temp_label = y[i].item()
#         temp_data = x[i].item()
#         loss_batch += (temp_data[temp_label] - temp_label) ** 2
#     loss_batch = torch.div(loss_batch, batchsize)
#     return loss_batch

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
#optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

# ----------------------------------------------------------------------------------------------

for epoch in range(num_epoches):
    train()
    test()
    # Save model checkpoint
    if epoch % 5 == 0:
        os.makedirs("model_checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"model_checkpoints/{model.__class__.__name__}_{epoch}.pth")
