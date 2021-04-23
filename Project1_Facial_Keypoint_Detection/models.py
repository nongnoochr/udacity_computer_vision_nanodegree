## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # maxpool layer which will be added after every CONV+Activation
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # Layer#1
        # Image size [1, 224, 224]
        # Output W, H -> (W - F + 2P)/S + 1 -> (224 - 5 + 4)/1 + 1 = 224
        # Output Depth = 32
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.dropout1 = nn.Dropout2d(p=0.1)
        
        # Layer#2
        # Input size (afterPooling) [32, 112, 112]
        # Output W, H -> (W - F + 2P)/S + 1 -> (112 - 3 + 2)/1 + 1 = 112
        # Output Depth = 64
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.dropout2 = nn.Dropout2d(p=0.2)
        
        # Layer#3
        # Input size (afterPooling) [64, 56, 56]
        # Output W -> (W - F + 2P)/S + 1 -> (56 - 3 + 2)/1 + 1 = 56
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.dropout3 = nn.Dropout2d(p=0.3)
        
        # Layer#4 -> 1st FC
        # Input size (afterPooling) [128, 28, 28]
        
        # 128 outputs * the 28*28 filtered/pooled map size
        self.fc1 = nn.Linear(128*28*28, 1000)
        self.fc1_drop = nn.Dropout(p=0.5)
        
        # Layer#5
        # It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        self.fc2 = nn.Linear(1000, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        # Layer#1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Layer#2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Layer#3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # Layer#4 - First FC
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        
        # Layer#5 -> last FC
        x = self.fc2(x)
        
        return x
