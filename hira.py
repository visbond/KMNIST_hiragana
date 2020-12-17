# KMNIST Hiragana, deep learning models

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.input_to_out = nn.Linear(in_features = 784, out_features = 10) # 784 = 28*28
            # will encode the 784 inputs into 10 outputs (since have 10 classes)
        
    def forward(self, x):
        # DEBUG
        # print("The input shape of x is",x.shape)
        x = x.view(x.shape[0], -1)
            # this keeps the first dimension (which indexes batches) and flattens the rest (channel*width*height)
            # hardcoding 64 in the first dimension also works, but x.shape[0] is more flexible

        x = self.input_to_out(x)
        return F.log_softmax(x, dim = 1)
        

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        
        # 30 hidden nodes gave 80% accuracy
        # self.input_to_hid1 = nn.Linear(784, 30)  
        # self.hid1_to_hid2 = nn.Linear(30, 30)
        # self.hid2_to_out = nn.Linear(30,10) # outputs to 10 classes
        
        # 50 hidden nodes gave 83% accuracy
        # self.input_to_hid1 = nn.Linear(784, 50)
        # self.hid1_to_hid2 = nn.Linear(50, 50) 
        # self.hid2_to_out = nn.Linear(50,10) 
        
        # 60 hidden nodes gave 84.5% accuracy
        # self.input_to_hid1 = nn.Linear(784, 60)
        # self.hid1_to_hid2 = nn.Linear(60, 60) 
        # self.hid2_to_out = nn.Linear(60,10)
        
        # 90 hidden nodes give 85% accuracy
        self.input_to_hid1 = nn.Linear(784, 90)
        self.hid1_to_hid2 = nn.Linear(90, 90) 
        self.hid2_to_out = nn.Linear(90,10)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # x = F.tanh(self.input_to_hid1(x)) # F.tanh giving deprecation warning, asks to use torch.tanh
        # x = F.tanh(self.hid1_to_hid2(x))
        x = torch.tanh(self.input_to_hid1(x)) # with F.tanh was getting a deprecation warning
        x = torch.tanh(self.hid1_to_hid2(x))        
        x = self.hid2_to_out(x)
        return F.log_softmax(x, dim = 1)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
# more versions of this that were tried are in comments at the end of the file
    def __init__(self):
        super(NetConv, self).__init__()
        self.imag_to_convo1 = nn.Conv2d(1, 96, 3) # key idea, small kernel first for edge-detection
        self.convo1_to_2 = nn.Conv2d(96, 128, 5) # bigger kernel later for making sense of shapes

        # DEBUG
        # to confirm the exact dimensions after convolution
        x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
            # we are sending a 28 by 28 pixel grid now, so not flattening to 784
            # -1 takes care of any arbitrary batch size
        self.flattened = None # will hold flattened dimensions
        self.convolve(x)

        self.convo2_to_fc = nn.Linear(self.flattened, 96) 
        self.fc_to_out = nn.Linear(96, 10)
        
    '''does max pooling, and confirms size'''
    def convolve(self,x, pool = False):
        if pool:
            x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
            x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
        else:
            x = F.relu(self.imag_to_convo1(x))
            x = F.relu(self.convo1_to_2(x))
        if self.flattened is None:
            self.flattened = 1
            for i in (x[0].size()):
                self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convolve(x)
        x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
        x = self.fc_to_out(x)
        return F.log_softmax(x, dim=1) # initially used softmax instead of log_softmax and wasted several hours


#for part 4
class NetConv_Custom(nn.Module):
    #3 convolutional layers
    def __init__(self):
        super(NetConv_Custom, self).__init__()
        self.imag_to_convo1 = nn.Conv2d(1, 96, 3) # key idea, small kernel first for edge-detection
        self.convo1_to_2 = nn.Conv2d(96, 96, 5) # bigger kernel later for making sense of shapes
        self.convo2_to_3 = nn.Conv2d(96, 64, 7)

        # DEBUG
        # to confirm the exact dimensions after convolution
        x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
            # we are sending a 28 by 28 pixel grid now, so not flattening to 784
            # -1 takes care of any arbitrary batch size
        self.flattened = None # will hold flattened dimensions
        self.convolve(x)

        self.convo3_to_fc = nn.Linear(self.flattened, 96) 
        self.fc_to_out = nn.Linear(96, 10)
        
    '''does max pooling, and confirms size'''
    def convolve(self,x, pool = False):
        if pool:
            x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
            x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
            x = F.max_pool2d(F.relu(self.convo2_to_3(x)), (2,2))
        else:
            x = F.relu(self.imag_to_convo1(x))
            x = F.relu(self.convo1_to_2(x))
            x = F.relu(self.convo2_to_3(x))
        if self.flattened is None:
            self.flattened = 1
            for i in (x[0].size()):
                self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convolve(x)
        x = F.relu(self.convo3_to_fc(x.view(-1, self.flattened)))
        x = self.fc_to_out(x)
        return F.log_softmax(x, dim=1) # initially used softmax instead of log_softmax and wasted several hours


#############VARIOUS RUNS OF SUBPART 3 WITH DIFFERENT ACCURACY, FOR REFERENCE##########

#93%, and reasonably fast (compared to earlier ones)
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 96, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(96, 128, 5) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution
#         x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
#             # we are sending a 28 by 28 pixel grid now, so not flattening to 784
#             # -1 takes care of any arbitrary batch size
#         self.flattened = None # will hold flattened dimensions
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 96) 
#         self.fc_to_out = nn.Linear(96, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.log_softmax(x, dim=1) # initially used softmax instead of log_softmax and wasted several hours


# got 94% as soon as used log_softmax
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 128, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(128, 172, 6) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution

#         x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
#             # we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None # will hold flattened dimensions
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 160) 
#         self.fc_to_out = nn.Linear(160, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.log_softmax(x, dim=1) # initially used softmax instead of log_softmax and wasted several hours


#82%. Even worse shifting of class 6 to class 3
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 256, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(256, 256, 4) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution

#         x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
#             # we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None # will hold flattened dimensions
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 216) 
#         self.fc_to_out = nn.Linear(216, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1) # initially used softmax instead of log_softmax and wasted several hours


# DEBUG, 85%. Class 6 being seen as 3
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 160, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(160, 256, 10) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution

#         x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
#             # we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None # will hold flattened dimensions
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 216) 
#         self.fc_to_out = nn.Linear(216, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)


## 85%, all classes detected from early on, but 6 mistaken for 3
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 128, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(128, 160, 8) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution

#         x = torch.randn(28,28).view(-1,1,28,28) # note that this is convolution, 
#             # we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None # will hold flattened dimensions
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 160) 
#         self.fc_to_out = nn.Linear(160, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)


# this gives 84% accuracy. All classes turn on by the end, but poor accuracy on 6th class\
# (is being seen as 3rd)
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 128, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(128, 128, 7) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution
#         x = torch.randn(28,28).view(-1,1,28,28) # note that we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 128) 
#         self.fc_to_out = nn.Linear(128, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)


# DEBUG
# reached 75%, 9th column never turned on. 5th one did midway. 6th class heavily predicted as 3rd
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 96, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(96, 128, 8) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution
#         x = torch.randn(28,28).view(-1,1,28,28) # note that we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 64) 
#         self.fc_to_out = nn.Linear(64, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x, pool = False):
#         if pool:
#             x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#             x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         else:
#             x = F.relu(self.imag_to_convo1(x))
#             x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)

#DEBUG
# the following got to 80%, and also started out stronger at 57%, but third column still empty
# uses idea of smaller initial kernel and bigger later one
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 64, 3) # key idea, small kernel first for edge-detection
#         self.convo1_to_2 = nn.Conv2d(64, 96, 6) # bigger kernel later for making sense of shapes

#         # DEBUG
#         # to confirm the exact dimensions after convolution
#         x = torch.randn(28,28).view(-1,1,28,28) # note that we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 64) 
#         self.fc_to_out = nn.Linear(64, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x):
#         # x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2)) # first run, with pooling and 
#             # 32 nodes, 4x4 conv kernels gave 49% accuracy
#         # x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         x = F.relu(self.imag_to_convo1(x))
#         x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)


#DEBUG
# the following got till 66%, but second and third columns of confusion matrix
#  completely empty, and bad spread in 6th row
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 64, 6)
#         self.convo1_to_2 = nn.Conv2d(64, 64, 6)

#         # DEBUG
#         # to confirm the exact dimensions after convolution
#         x = torch.randn(28,28).view(-1,1,28,28) # note that we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 64) 
#         self.fc_to_out = nn.Linear(64, 10)
        
#     '''does max pooling, and confirms size'''
#     def convolve(self,x):
#         # x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2)) # first run, with pooling and 
#             # 32 nodes, 4x4 conv kernels gave 49% accuracy
#         # x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         x = F.relu(self.imag_to_convo1(x))
#         x = F.relu(self.convo1_to_2(x))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)

# DEBUG
# the following started with 20% and reached only till 49% accuracy, and negative loss
# third, fifth and ninth column of confusion matrix completely empty
# class NetConv(nn.Module):
#     # two convolutional layers and one fully connected layer,
#     # all using relu, followed by log_softmax
#     def __init__(self):
#         super(NetConv, self).__init__()
#         self.imag_to_convo1 = nn.Conv2d(1, 32, 4)
#         self.convo1_to_2 = nn.Conv2d(32, 32, 4)

#         # DEBUG
#         # to confirm the exact dimensions after convolution
#         x = torch.randn(28,28).view(-1,1,28,28) # note that we are sending a 28 by 28 pixel grid now, so not flattening to 784
#         self.flattened = None
#         self.convolve(x)

#         self.convo2_to_fc = nn.Linear(self.flattened, 32) 
#         self.fc_to_out = nn.Linear(32, 10)
        
    
#     '''does max pooling, and confirms size'''
#     def convolve(self,x):
#         x = F.max_pool2d(F.relu(self.imag_to_convo1(x)), (2,2))
#         x = F.max_pool2d(F.relu(self.convo1_to_2(x)), (2,2))
#         if self.flattened is None:
#             self.flattened = 1
#             for i in (x[0].size()):
#                 self.flattened *= i # x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
#         return x

#     def forward(self, x):
#         x = self.convolve(x)
#         x = F.relu(self.convo2_to_fc(x.view(-1, self.flattened)))
#         x = self.fc_to_out(x)
#         return F.softmax(x, dim=1)
