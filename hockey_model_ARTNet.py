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
logger = get_logger('ARTNet_hockeyFight.log')

##############################
#         ARTNet
##############################
class SMART_block(nn.Module):

    def __init__(self, in_channel,out_channel,kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super(SMART_block, self).__init__()

        self.appearance_conv=nn.Conv3d(in_channel, out_channel, kernel_size=(1,kernel_size[1],kernel_size[2]),stride= stride,padding=(0, padding[1], padding[2]),bias=False)
        self.appearance_bn=nn.BatchNorm3d(out_channel)

        self.relation_conv=nn.Conv3d(in_channel, out_channel,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relation_bn1=nn.BatchNorm3d(out_channel)
        self.relation_pooling=nn.Conv3d(out_channel,out_channel//2,kernel_size=1,stride=1,groups=out_channel//2,bias=False)
        nn.init.constant_(self.relation_pooling.weight,0.5)
        self.relation_pooling.weight.requires_grad=False
        self.relation_bn2 = nn.BatchNorm3d(out_channel//2)

        self.reduce=nn.Conv3d(out_channel+out_channel//2,out_channel,kernel_size=1,bias=False)
        self.reduce_bn=nn.BatchNorm3d(out_channel)

        self.relu = nn.ReLU()
        if in_channel != out_channel or stride[0] != 1 or stride[1] != 1:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=1,
                                                       stride=stride,
                                                       bias=False),
                                             nn.BatchNorm3d(out_channel))
        else:
            self.down_sample = None

    def forward(self, x):
        appearance=x
        relation=x
        appearance=self.appearance_conv(appearance)
        appearance=self.appearance_bn(appearance)
        relation=self.relation_conv(relation)
        relation=self.relation_bn1(relation)
        relation=torch.pow(relation,2)
        relation=self.relation_pooling(relation)
        relation=self.relation_bn2(relation)
        stream=self.relu(torch.cat([appearance,relation],1))
        stream=self.reduce(stream)
        stream=self.reduce_bn(stream)
        if self.down_sample is not None:
            x=self.down_sample(x)

        return self.relu(stream+x)


class ARTNet(nn.Module):
    # Input size: 16x112x112
    def __init__(self, num_class=2, init_weights=True):
        super(ARTNet, self).__init__()

        self.conv1=SMART_block(3,64,kernel_size=(3,7,7),stride=(2,2,2),padding=(1,3,3))
        self.conv2=nn.Sequential(SMART_block(64,64),
                                 SMART_block(64, 64))
        self.conv3=nn.Sequential(SMART_block(64,128,stride=(2,2,2)),
                                 SMART_block(128, 128))
        self.conv4 = nn.Sequential(SMART_block(128, 256, stride=(2,2,2)),
                                   SMART_block(256, 256))
        self.conv5 = nn.Sequential(SMART_block(256, 512, stride=(2,2,2)),
                                   SMART_block(512, 512))
        self.avg_pool=nn.AvgPool3d(kernel_size=(1,7,7))
        self.linear=nn.Linear(512,num_class)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        return self.linear(x.view(x.size(0),-1))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

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
model = ARTNet()
#model.apply(weights_init)
model = model.cuda(device_ids[0])
#model = nn.DataParallel(model, device_ids=device_ids)
summary(model,(3,16,112,112))

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
