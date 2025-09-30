import torch
import torch.nn as nn
from BNN2 import * # 양자화 함수들

import brevitas.nn as qnn

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2



# 기존의 LearnableBias
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

# brevitas과 호환된 LearnableBias
# class LearnableBias(nn.Module):
    
#     def __init__(self, out_chn):
#         super(LearnableBias, self).__init__()
#         self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
    
#     def forward(self,x):
#         if hasattr(x,'value'):
#             out = x.value + self.bias.expand_as(x.value)
#             return x.set(value=out)
#         else:
#             return x+ self.bias.expand_as(x)


# class firstconv3x3(nn.Module):
#     def __init__(self, inp, oup, stride):
#         super(firstconv3x3, self).__init__()

#         self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(oup)

#     def forward(self, x):

#         out = self.conv1(x)
#         out = self.bn1(out)

#         return out


class BasicBlock(nn.Module):

    def __init__(self,inplanes,planes,stride=1):
        super(BasicBlock,self).__init__()

        
        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = Bconv3x3(inplanes,inplanes,stride=stride)
        self.move12 = LearnableBias(inplanes)
        # self.bn1 = qnn.BatchNorm2dToQuantScaleBias(inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes) 

        self.prelu1  = Int8ReLU()

        self.move13 = LearnableBias(inplanes)
        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = Bconv1x1(inplanes, planes)
            #self.bn2 = qnn.BatchNorm2dToQuantScaleBias(planes)
            self.bn2 = nn.BatchNorm2d(planes) 
        else:
            self.binary_pw_down1 = Bconv1x1(inplanes, inplanes)
            self.binary_pw_down2 = Bconv1x1(inplanes, inplanes)
            #self.bn2_1 = qnn.BatchNorm2dToQuantScaleBias(inplanes)
            #self.bn2_2 = qnn.BatchNorm2dToQuantScaleBias(inplanes)
            self.bn2_1 = nn.BatchNorm2d(inplanes)
            self.bn2_2 = nn.BatchNorm2d(inplanes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = Int8ReLU()

        self.move23 = LearnableBias(planes)
        self.binary_activation = BinaryActivation()

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            # self.pooling = QAvgPool2d(2,2)
            self.pooling = nn.AvgPool2d(2,2)

    def forward(self,x):

        x_in= x

        out1 = self.move11(x_in)
        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x_in)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out1_in = out1
        out2 = self.move21(out1_in)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 = out2 + out1
        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)

            out2_1 = out2_1+ out1
            out2_2 = out2_2+ out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class Reactnet(nn.Module):
    def __init__(self, num_classes=1000,imagenet=True,input_sign=True):
        super(Reactnet,self).__init__()
        self.quant_inp = Quantinput(input_sign)    
        self.feature = nn.ModuleList()

        for i in range(len(stage_out_channel)):
            if i == 0:
                if imagenet:
                    self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
                else:
                    self.feature.append(firstconv3x3(3, stage_out_channel[i], 1))
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))

        # self.pool1 = QAdaptiveAvgPool2d(1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = Int8Linear(1024,num_classes)
        
        
    
    def forward(self,x):
        
        x= self.quant_inp(x)
        
        for _,block in enumerate(self.feature):
            x = block(x)

        x= self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x







