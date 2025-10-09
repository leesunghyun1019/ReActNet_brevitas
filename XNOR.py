import torch
import torch.nn as nn
from BNN2 import * # 양자화 함수들

import brevitas.nn as qnn

import torch.nn.functional as F

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1 , downsample=None):
        super(BasicBlock,self).__init__()

        self.act1 = BinaryActivation()
        self.conv1 = Bconv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride



    def forward(self,x):

        residual = x 
        out = x
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class ResNet(nn.Module):

    def __init__(self,block,layers,imagenet=False,num_classes=10,input_sign=True):
        super(ResNet,self).__init__()
        
        self.inplanes=64
        self.quant_inp = Quantinput(input_sign) 
        
        if imagenet:
            self.conv1 = firstconv3x3(3, 64, stride=2, kernel_size=7, padding=3)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        else:
            self.conv1 = firstconv3x3(3, 64, stride=1)
            self.maxpool = nn.Identity()

        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = Int8ReLU()

        self.layer1= self._make_layer(block,64,layers[0])
        self.layer2= self._make_layer(block,128,layers[1],stride=2)
        self.layer3= self._make_layer(block,256,layers[2],stride=2)
        self.layer4= self._make_layer(block,512,layers[3],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Int8Linear(512*block.expansion, num_classes)

        self.o_quant = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = True)

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Bconv1x1(self.inplanes,planes*block.expansion,stride=stride),
                nn.BatchNorm2d(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        
        self.inplanes = planes*block.expansion
        
        for _ in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        
        x= self.quant_inp(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.prelu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.o_quant(x)

        return F.log_softmax(x, dim = 1)

def ResNet18(num_classes = 10, imagenet = False, input_sign=True):
    model = ResNet(BasicBlock, [4, 4, 4, 4], num_classes=num_classes, imagenet=imagenet,input_sign=input_sign)
    return model
