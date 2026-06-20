from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet-LDEM':
            if mode == 'train':
                self.model = DNANet_LDEM(mode='train')
            else:
                self.model = DNANet_LDEM(mode='test')
        elif model_name == 'DNANet-LDEM-Gate':
            if mode == 'train':
                self.model = DNANet_LDEM_Gate(mode='train')
            else:
                self.model = DNANet_LDEM_Gate(mode='test')
        elif model_name == 'DNANet-LDEM-Gate-Stable':
            if mode == 'train':
                self.model = DNANet_LDEM_Gate_Stable(mode='train')
            else:
                self.model = DNANet_LDEM_Gate_Stable(mode='test')
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ResUNet':
            self.model = ResUNet()
        elif model_name == 'ResUNet-CBAM':
            self.model = ResUNet_CBAM()
        elif model_name == 'ResUNet-GCA':
            self.model = ResUNet_GCA()
        elif model_name == 'ResUNet-GCA-DSPG':
            self.model = ResUNet_GCA_DSPG()
        elif model_name == 'ResUNet-GCA-NoGate':
            self.model = ResUNet_GCA_NoGate()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        
    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
