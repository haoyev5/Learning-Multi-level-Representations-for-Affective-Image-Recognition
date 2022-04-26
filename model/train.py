

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:17:57 2020

@author: haoye
"""
import sys
import os
#sys.path.append(os.getcwd())
os.system('dir')
os.system('ls')
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
#import gluoncv
import time
import d2lzh as d2l
sys.path.append(os.getcwd())
import utils
import matplotlib.pyplot as plt
from gluoncv import model_zoo
#from mxnet.gluon import model_zoo

# some superparameters that we can fine-tuning
batch_size          = 16
num_epochs, lr, wd  = 60, 0.0001, 1e-4
lr_period, lr_decay = 10, 0.5
epsilon, momentum   = 2e-5, 0.9


# modual-net parameters
num_classes = 8


#data file name
data_dir   = 'D:/data/FI'
train_dir, test_dir = 'train', 'valid'

# try to train model on GPU
#ctx = d2l.try_gpu()
ctx = mx.gpu(0)


# Preprocessing data
transform_train = gdata.vision.transforms.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和
    # 宽均为224像素的新图像
    gdata.vision.transforms.Resize(256),
    gdata.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
    gdata.vision.transforms.RandomFlipLeftRight(),
    gdata.vision.transforms.RandomColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
    #gdata.vision.transforms.RandomColorJitter(brightness=0.4, contrast=0.4),
    gdata.vision.transforms.RandomLighting(0.1),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.Resize(224),
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])])


# load data
train_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, train_dir), flag=1)
if test_dir is not None:
    test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, test_dir), flag=1)
else:
    test_ds = None
#test_ds = gdata.vision.ImageFolderDataset(os.path.join(data_dir, test_dir), flag=1)
train_iter = gdata.DataLoader(train_ds.transform_first(transform_train), batch_size=batch_size, shuffle=True, last_batch='keep')
print('train iter complete!')
if test_ds is not None:
    test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), batch_size=batch_size, shuffle=False, last_batch='keep')
    print('test iter complete!')
else:
    test_iter = None
    print('No test iter! Go ahead---->')
#test_iter = gdata.DataLoader(test_ds.transform_first(transform_test), batch_size=batch_size, shuffle=False, last_batch='keep')
    
############################################################################
class Gram(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Gram, self).__init__(**kwargs)        
        
    def forward(self, X):
        B,C,H,W = X.shape 
        X = X.reshape((B, C, H*W))
        Y = X.transpose((0,2,1))
        return nd.linalg_gemm2(X, Y) / (C*H*W)    

def get_resnet_features_extractor(ctx):
    # get pretrained model from mxnet model_zoo
    #resnet50_v1 resnet152_v2 resnet101_v2 resnext50_32x4d
    resnet = model_zoo.get_model('resnet50_v2', pretrained=True, ctx=ctx)
    features_extractor = resnet.features
    return features_extractor

def get_net(num_classes, ctx):
    class ResNet(nn.HybridBlock):
        def __init__(self, ctx, **kwargs):
            super( ResNet, self).__init__(**kwargs)
            
            embed_size = 2048 
            reduction  = 1
            
            self.feature_extractor = get_resnet_features_extractor(ctx)
            #self.feature_extractor.initialize(init.Xavier(), ctx=ctx)

            with self.name_scope():
                '''
                self.branch_0 = nn.HybridSequential(prefix='branch_0')
                with self.branch_0.name_scope():            
                    self.branch_0.add(nn.Conv2D(channels=128, kernel_size=11, strides=2, padding=5, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      Gram(),
                                      nn.Flatten(),
                                      nn.Dense(embed_size, activation='relu'),
                                      nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.branch_0.initialize(init.Xavier(), ctx=ctx)
                '''
                self.branch_1 = nn.HybridSequential(prefix='branch_1')
                with self.branch_1.name_scope():
                    self.branch_1.add(#nn.Conv2D(channels=64//reduction, kernel_size=1, strides=1, padding=0, activation='relu'),
                                      #nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      Gram(),
                                      nn.Flatten(),
                                      nn.Dense(64//reduction, activation='relu'))
                self.branch_1.initialize(init.Xavier(), ctx=ctx)
                    
                self.branch_2 = nn.HybridSequential(prefix='branch_2')
                with self.branch_2.name_scope():
                    self.branch_2.add(#nn.Conv2D(channels=256//reduction, kernel_size=1, strides=1, padding=0, activation='relu'),
                                      nn.MaxPool2D(pool_size=3,strides=2,padding=1),
                                      #nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      Gram(),
                                      nn.Flatten(),
                                      nn.Dense(256//reduction, activation='relu'))
                self.branch_2.initialize(init.Xavier(), ctx=ctx)
                    
                self.branch_3 = nn.HybridSequential(prefix='branch_3')
                with self.branch_3.name_scope():
                    self.branch_3.add(#nn.Conv2D(channels=512//reduction , kernel_size=1, strides=1, padding=0, activation='relu'),
                                      nn.MaxPool2D(pool_size=3,strides=2,padding=1),
                                      #nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                      Gram(),
                                      nn.Flatten(),
                                      nn.Dense(512//reduction, activation='relu'))
                self.branch_3.initialize(init.Xavier(), ctx=ctx)
    
                self.styleVector = nn.HybridSequential(prefix='styleVector')
                with self.styleVector.name_scope():
                    self.styleVector.add(nn.Dropout(.8),
                                         nn.Dense(embed_size//reduction, activation='relu'),
                                         nn.BatchNorm(epsilon=2e-5, momentum=0.9))
                self.styleVector.initialize(init.Xavier(), ctx=ctx)
                
                #self.feature_extractor2 = get_resnet_features_extractor(ctx)
    
                self.output = nn.HybridSequential(prefix='output')
                with self.output.name_scope():
                    self.output.add(#nn.Dropout(.5),
                                    nn.BatchNorm(epsilon=2e-5, momentum=0.9),
                                    nn.Dense(num_classes))
                self.output.initialize(init.Xavier(), ctx=ctx)
                                     
            #self.flat = nn.Flatten()
               

        def hybrid_forward(self, F, X, *args, **kwargs):
            #feature_extractor = get_resnet_features_extractor(ctx)
            # branch_0           
            #branch_0 = self.branch_0(X)
            
            # branch_1
            y = self.feature_extractor[:4](X)            
            branch_1 = self.branch_1(y)
            #print('gram1 shape:', gram1.shape)                    
            
            # branch_2
            y = self.feature_extractor[4:5](y)
            branch_2 = self.branch_2(y) 
            
            # branch_3
            y = self.feature_extractor[5:6](y)
            branch_3 = self.branch_3(y)
            
            #style feature representation
            style = nd.concat(branch_1, branch_2, branch_3, dim=1)
            style = self.styleVector(style)            
            
            # high-level feature representation: backbone feature
            y = self.feature_extractor[6:](y)
            #y = self.flat(y)
            
            # integrate low- and high-level feature and output
            out = nd.concat(y, style, dim=1)
            #out = nd.broadcast_mul(y, style)
            #out = nd.broadcast_add(y, style)
            out = self.output(out)                    
            
            return out

    return ResNet(ctx)

net = get_net(num_classes, ctx)

#Perform a computational graph operation before training to prevent bugs from appearing
X = nd.random.uniform(shape=(16,3,224,224)).as_in_context(ctx)
print('Input shape:', X.shape)
X = net(X)
print('Output shape:', X.shape)

print("==============================================================")
print("==============================================================")

############################################################################
  
# train net
utils.train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)


# save the model
saved_filename = 'resnet50_v2_energy8.params'
net.save_parameters(saved_filename)
