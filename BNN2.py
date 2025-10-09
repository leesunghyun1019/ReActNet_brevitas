import torch
import torch.nn as nn
import brevitas.nn as qnn

#from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.inject.enum import RestrictValueType
from brevitas.inject import value


from brevitas.quant import Int8WeightPerTensorFloat,Int8ActPerTensorFloat,Uint8ActPerTensorFloat

# Binary Convolution
class BinaryWeightPerChannel(Int8WeightPerTensorFloat):
     quant_type = QuantType.BINARY
     bit_width = 1
     bit_width_impl_type = BitWidthImplType.CONST
     scaling_impl_type = ScalingImplType.STATS
     scaling_stats_op = StatsOp.AVE
     scaling_per_output_channel = True
     narrow_range = False
     
     @value
     def scaling_init(module):
         return torch.mean(torch.abs(module.weight), dim=(1, 2, 3), keepdim=True)
    
   

def Bconv1x1(in_planes,out_planes,stride=1):
    """1x1 binary convolution without padding"""
    return qnn.QuantConv2d(in_planes,out_planes,kernel_size=1,weight_quant=BinaryWeightPerChannel, stride=stride,padding=0,bias=False)

def Bconv3x3(in_planes,out_planes,stride=1):
    """3x3 binary convolution with padding"""
    return qnn.QuantConv2d(in_planes,out_planes,kernel_size=3,weight_quant=BinaryWeightPerChannel, stride=stride,padding=1,bias=False)


# Binary Activation
class BinaryActPerTensor(Int8ActPerTensorFloat):
    quant_type = QuantType.BINARY
    min_val = -1.0
    max_val = 1.0
    bit_width = 1
    restrict_scaling_type = RestrictValueType.FP

# class BinaryActivation(nn.Module):
#     def __init__(self,alpha=3,return_quant_tensor=False):
#         super(BinaryActivation,self).__init__()
#         self.quant = qnn.QuantSigmoid(act_quant=BinaryActPerTensor,return_quant_tensor=return_quant_tensor)
#         self.alpha = alpha
#     def forward(self,x):
#         return self.quant(self.alpha*x)

# BianryActivation 수정
class BinaryActivation(nn.Module):
    def __init__(self,return_quant_tensor=True):
        super(BinaryActivation,self).__init__()
        self.quant = qnn.QuantIdentity(act_quant=BinaryActPerTensor,return_quant_tensor=return_quant_tensor)
        

    def forward(self,x):
        return self.quant(torch.sign(x))  

class firstconv3x3(nn.Module):
    def __init__(self,in_planes,out_planes,stride,kernel_size=3, padding=1):
        super(firstconv3x3,self).__init__()
        self.conv=qnn.QuantConv2d(in_planes,out_planes,kernel_size=kernel_size,weight_bit_width=8,weight_quant=Int8WeightPerTensorFloat,weight_scaling_min_val=2e-16,restrict_scaling_type=RestrictValueType.FP,stride=stride,padding=padding,bias=False)

    def forward(self,x):
        out=self.conv(x)
        return out

# #Int8 Activation
# class Int8Activation(nn.Module):
#     def __init__(self):
#         super(Int8Activation,self).__init__()
#         self.quant= qnn.QuantSigmoid(bit_width=8, return_quant_tensor= False, act_quant=Int8ActPerTensorFloat, scaling_min_val = 2e-16, restrict_scaling_type = RestrictValueType.FP)
#     def forward(self,x):
#         return self.quant(x)


class Int8ReLU(nn.Module):
    def __init__(self):
        super(Int8ReLU,self).__init__()
        self.relu = qnn.QuantReLU(bit_width = 8, return_quant_tensor = False)
    def forward(self, x):
        return self.relu(x)

class QAvgPool2d(nn.Module):
    def __init__(self, kernel_size,stride=None,padding=0):
        super(QAvgPool2d,self).__init__()
        self.Avgpool2d= qnn.TruncAvgPool2d(kernel_size = kernel_size, stride=stride, padding = padding, return_quant_tensor = False)

    def forward(self,x):
        return self.Avgpool2d(x)

class QAdaptiveAvgPool2d(nn.Module):
    def __init__(self,output_size):
        super(QAdaptiveAvgPool2d,self).__init__()
        self.AdaptiveAvgPool2d=qnn.TruncAdaptiveAvgPool2d(output_size=output_size, return_quant_tensor = False)
    
    def forward(self,x):
        return self.AdaptiveAvgPool2d(x)


class Int8Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Int8Linear,self).__init__()
        self.linear=qnn.QuantLinear(in_features=in_features,out_features=out_features,weight_quant= Int8WeightPerTensorFloat,weight_bit_width= 8, return_quant_tensor= False)
    def forward(self,x):
        return self.linear(x)


class Quantinput(nn.Module):
    def __init__(self, input_sign=True):
        super(Quantinput,self).__init__()

        if (input_sign):
            self.quant_inp = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = False, act_quant = Uint8ActPerTensorFloat, scaling_min_val = 2e-16, 
                            restrict_scaling_type = RestrictValueType.FP)
        else:
            self.quant_inp = qnn.QuantIdentity(bit_width = 8, return_quant_tensor = False,
                         act_quant = Int8ActPerTensorFloat, scaling_min_val = 2e-16, 
                                        restrict_scaling_type = RestrictValueType.FP)

    def forward(self,x):
        return self.quant_inp(x)





