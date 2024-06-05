#importing libraries
import torch 
import torch.nn as nn
import torch optim as optim 
import torchvision 
import torchvision.transform as transforms 
from torch.utils.data import dataloader 

#defining the transformation of data
transform= transforms.compose([transform.ToTensor(),transform.normalize((0.5,)(0.5))])