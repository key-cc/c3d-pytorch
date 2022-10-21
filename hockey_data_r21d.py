# coding:utf-8

# output hdf5 file
# video data: N * C * L * H * W
# label: N * 1

import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import h5py
import matplotlib
import matplotlib.pyplot as plt


def create_trainh5(number,length,height,width):
        # get the frames and ids

        video = np.zeros((number, length, 3, height, width), dtype=np.float32)
        label = np.zeros(number, dtype=np.int64)

        with open('./train.txt', 'r') as f:
                num = 0
                for line in f:
                        line = line.strip()
                        dir_name, frame, lbl = line.split()
                        images = np.zeros((length, 3, height, width), dtype=np.float32)
                        for i in range(int(frame),int(frame)+length):
                          
                                img_name = str(i) + '.jpg'
                                img = Image.open(os.path.join(dir_name,img_name))
                                #img = img.resize((112,112))
                          

                                img = transform(img)
                                img = img.numpy()
        
                                images[i-int(frame)] = img

                        video[num] = images 
                        label[num] = int(lbl)
                        num = num + 1   


        video = np.asarray(video,dtype=np.float32)   
        print(video.shape)
        label = np.asarray(label,dtype=np.int64)
        print(label.shape)

        f = h5py.File('hockey_train.h5','w')
        f['data'] = video                
        f['label'] = label         
        f.close()

# -------------------------------------------------------------------------------------------------

def create_testh5(number,height,width):
        # get frames and ids
        video = []
        label = []
        video = np.zeros((number, 8, 3, height, width), dtype=np.float32)
        label = np.zeros(number, dtype=np.int64)

        with open('./test.txt', 'r') as f:
                num = 0
                for line in f:
                        line = line.strip()
                        dir_name, lbl = line.split()
                        images = np.zeros((8, 3, height, width), dtype=np.float32)
                        for i in range(1,9):
                                img_name = str(i) + '.jpg'
                                img = Image.open(os.path.join(dir_name,img_name))
                                #img = img.resize((112,112))

                                img = transform(img)
                                img = img.numpy()
                                #plt.savefig('1.jpg',img)
                                print(num)

        
                                images[i-1] = img

                        video[num] = images 
                        label[num] = int(lbl)
                        num = num + 1   


        video = np.asarray(video,dtype=np.float32)   
        print(video.shape)
        label = np.asarray(label,dtype=np.int64)
        print(label.shape)

        f = h5py.File('hockey_test.h5','w')
        f['data'] = video                
        f['label'] = label         
        f.close()

# --------------------------------------------------------------------------------------------------
#frame transform
transform = transforms.Compose([
        transforms.CenterCrop((210,330)),
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]
)


#create_trainh5(1000,16,70,110)
create_trainh5(1000,8,112,112)
#create_testh5(200,70,110)
create_testh5(200,112,112)