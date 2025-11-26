import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
import os
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Uint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias
from brevitas.core.restrict_val import RestrictValueType

from configure_ibex_new import get_quantnet_details
from configure_ibex_new import get_int_params
from configure_ibex_new import generate_og_c_code_cnn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(BasicBlock, self).__init__()

        self.conv1 = qnn.QuantConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
            cache_inference_bias = True
        )

        self.act1 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                       return_quant_tensor=True)
        
        self.conv2 = qnn.QuantConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
            cache_inference_bias = True
        )

        self.act2 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                       return_quant_tensor=True)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.act1(out)
        out=self.conv2(out)
        out=self.act2(out)
        return out

class ResidualBlock(nn.Module):
    """Residual block with downsample"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResidualBlock, self).__init__()
        
        # Main branch
        self.conv1 = qnn.QuantConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
            cache_inference_bias = True
        )
        self.act1 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                       return_quant_tensor=True)

        
        self.conv2 = qnn.QuantConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
            cache_inference_bias = True
        )
        
        # Downsample branch
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                qnn.QuantConv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0, bias=True,
                    weight_quant=Int8WeightPerTensorFloat,
                    bias_quant=Int32Bias,
                    return_quant_tensor=True,
                    cache_inference_bias = True
                )
            )
        else:
            self.downsample=None

        self.add= qnn.QuantEltwiseAdd(output_quant=None,tie_input_output_quant=True,return_quant_tensor=True)

        self.act2 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                       return_quant_tensor=True)
        
    
    def forward(self, x):
        identity = x
        
        # Main branch
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)

        # Downsample branch
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add
        out = self.add(identity,out)
        out = self.act2(out)

        
        return out


class SimpleTestCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SimpleTestCNN, self).__init__()
        
        
        # Stem: Conv + ReLU
        self.conv1 = qnn.QuantConv2d(
            3, 16,
            kernel_size=3, stride=2, padding=1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
            cache_inference_bias = True
        )
        self.act1 = qnn.QuantIdentity(act_quant=Int8ActPerTensorFloat,
                                       return_quant_tensor=True)
        
        # Residual Block (shortcut without downsample)
        self.residual_block1 = ResidualBlock(16, 16,stride=1)
        #self.residual_block1 = BasicBlock(16, 16,stride=1) #basic block        
        # Residual Block (with downsample)
        self.residual_block2 = ResidualBlock(16, 32, stride=2)
        
        # Linear
        self.fc = qnn.QuantLinear(
            32*8*8,  # 2048
            num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
            cache_inference_bias = True
        )

        
    def forward(self, x):
        
        # Stem
        x = self.conv1(x)
        x = self.act1(x)
        
        # Basic block
        x = self.residual_block1(x)
        
        # Residual block
        x = self.residual_block2(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Linear
        x = self.fc(x)
        
        
        return x

class Quant_Model(nn.Module):
    def __init__(self,quant_model,input_sign=True):
        super(Quant_Model, self).__init__()

        if input_sign:
            self.quant_inp = qnn.QuantIdentity(
                bit_width=8, 
                return_quant_tensor=True,
                act_quant=Uint8ActPerTensorFloat, 
                scaling_min_val=2e-16, 
                restrict_scaling_type=RestrictValueType.LOG_FP
            )
        
        else:
            self.quant_inp = qnn.QuantIdentity(
                bit_width=8, 
                return_quant_tensor=True,
                act_quant=Int8ActPerTensorFloat, 
                scaling_min_val=2e-16, 
                restrict_scaling_type=RestrictValueType.LOG_FP
            )

        self.sequential = quant_model

        self.o_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

    def forward(self,x):
        x = self.quant_inp(x)
        x = self.sequential(x)
        x = self.o_quant(x)
        return F.log_softmax(x, dim=1)






if __name__ == "__main__":

    print("=== Creating Model ===")
    simplenet=SimpleTestCNN(num_classes=10)
    model = Quant_Model(quant_model=simplenet,input_sign=True)

    model.eval()

    print("=== Running Inference with Dummy Input ===")
    x = torch.randn(1, 3, 32, 32).abs()  # Uint8이므로 양수
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}\n")

    print("=== Extracting Network Details ===")
    details, act = get_quantnet_details(simplenet)
    
    for i, layer in enumerate(details):
        print(f"{i}: {layer['layer_type']}", end="")
        if layer['layer_type'] == 'Conv2d':
            print(f" - {layer['in_channels']}→{layer['out_channels']}, "
                f"stride={layer['stride']}, kernel={layer['kernel_size']}")
        elif layer['layer_type'] == 'Downsample_Conv2d':
            print(f" - {layer['in_channels']}→{layer['out_channels']}, "
                f"stride={layer['stride']} (DOWNSAMPLE)")
        elif layer['layer_type'] == 'Shortcut':
            print(f" - length={layer['length']}, has_downsample={layer['has_downsample']}")
        else:
            print()
    
    print(f"\n=== Activation Details ===")
    print(f"Total activations: {act}")

   
    print("\n=== Extracting Parameters ===")
    int_weights, int_biases, f_int_biases, shift_biases, mul_vals, shift_vals = get_int_params(model,details)
    print(f"Extracted {len(mul_vals)} mul_vals")
    print(f"Extracted {len(shift_vals)} shift_vals")

    print("\n=== Saving Quantized Parameters ===")


    print("\n=== Generating C Code ===")
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    input_data = x.numpy()
    generate_og_c_code_cnn(
        path=output_dir,
        name="simple_test_cnn_1",
        input=input_data,
        cnn_details=details,
        act_details=act,
        int_weights=int_weights
    )
    print("C code generated successfully!")
