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
from log import *
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_ids = [0,1,2,3]
logger = get_logger('C3D_hockeyFight.log')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(6144, 512)
        self.fc7 = nn.Linear(512, 100)
        self.fc8 = nn.Linear(100, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 6144)
        h = self.relu(self.fc6(h))
        # h = self.dropout(h)
        h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        logits = self.fc8(h)
        probs = F.softmax(logits, dim=1)

        return probs

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
model = C3D()
model.apply(weights_init)
model = model.cuda(device_ids[0])
#model = nn.DataParallel(model, device_ids=device_ids)
summary(model,(3,16,70,110))

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
