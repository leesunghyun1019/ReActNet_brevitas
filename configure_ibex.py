import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import shutil


def quantize_multiplier(real_multiplier):
    s = 0
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        s += 1

    q = int(round(real_multiplier * (1 << 7)))

    # Handle the special case when the real multiplier was so close to 1
    # that its fixed-point approximation was undistinguishable from 1.
    # We handle this by dividing it by two, and remembering to decrement
    # the right shift amount.

    if q == (1 << 7):
        q //= 2
        s -= 1

    quantized_multiplier = int(q)
    right_shift = s

    return quantized_multiplier, right_shift

            
def get_int_params(quant_net):

    int_weights = []
    weight_scales = []
    act_scales = []
    layer_types = []  # 각 레이어의 유형 추적 (binary 또는 int8)

    def extract_quant_params(module):

        for name,submodule in module.named_children():

             # Check if the submodule has weights and append them if present
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                int_weights.append(submodule.int_weight().cpu().detach().numpy())
                weight_scales.append(submodule.quant_weight_scale().cpu().detach().numpy())

                # check layer type(1bit or 8bit)
                is_binary = False
                if hasattr(submodule, 'weight_quant') and hasattr(submodule.weight_quant, 'bit_width'):
                    is_binary = (submodule.weight_quant.bit_width().item() == 1)
                layer_types.append('binary' if is_binary else 'int8')

            # Check if the submodule has activation scale and append it if present
            if hasattr(submodule, 'quant_act_scale') and submodule.quant_act_scale is not None:
                act_scales.append(submodule.quant_act_scale().cpu().detach().numpy())

            # Recursively extract parameters from the children modules
            extract_quant_params(submodule)

    # Start extraction from the top-level module
    extract_quant_params(quant_net)

    mul_vals,shift_vals= [],[]
    for i in range(len(act_scales)-1):
        
        weight_scale= weight_scales[i]
        input_scale = act_scales[i]
        next_act_scale = act_scales[i+1]

        print(f"\nLayer {i}:")
        print(f"  weight_scale shape: {weight_scale.shape}") #check if weight scale is {per channel} or {per tensor}
        print(f"  input_scale shape: {input_scale.shape}, {type(input_scale)}")
        print(f"  next_act_scale shape: {next_act_scale.shape}, {type(next_act_scale)}")

        
        if weight_scale.ndim > 1:
            # if {per channel} : (C,1,1,1) -> (C,)
            weight_scale_flat = weight_scale.squeeze()
        else:
            weight_scale_flat = weight_scale

        print(f"  Processed weight_scale: {weight_scale_flat.shape}")


        output_scale=weight_scale_flat*input_scale    
        M=output_scale/next_act_scale

        if M.ndim == 0 : #scalar
            print("scalar")
            mul, shift = quantize_multiplier(float(M))
            mul_vals.append(mul)
            shift_vals.append(shift)
        
        #채널별로 양자화된 곱셈기/시프트 계산
        else: #배열
            print("not scalar")
            channel_muls = []
            channel_shifts = []
            for j in range(M.shape[0]):
                mul, shift = quantize_multiplier(M[j])
                channel_muls.append(mul)
                channel_shifts.append(shift)
            
            mul_vals.append(np.array(channel_muls))
            shift_vals.append(np.array(channel_shifts))

    return int_weights,layer_types,mul_vals,shift_vals
