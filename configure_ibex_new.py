import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import shutil
import brevitas.nn as qnn
from typing import Dict, List, Any, Optional, Tuple


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

def get_int_params(quant_net, details):
    # details 0~8    act_scale inp 1~6 oup  in scale c0~ dc5
    int_weights = []
    int_bias = []
    in_scales = []
    act_scales = []
    add_scales = []
    
    def extract_quant_params(module):
        for name, submodule in module.named_children():
            # Check if the submodule has weights and append them if present
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                int_weights.append(submodule.int_weight().cpu().detach().numpy())
                int_bias.append(submodule.int_bias().cpu().detach().numpy())
                in_scales.append(submodule.quant_bias_scale().cpu().detach().numpy())

            # Check if the submodule has activation scale and append it if present
            if hasattr(submodule, 'quant_act_scale') and submodule.quant_act_scale is not None:
                act_scales.append(submodule.quant_act_scale().cpu().detach().numpy())

            if isinstance(submodule, qnn.QuantEltwiseAdd):
                add_scale=submodule.output_quant.scale().cpu().detach().numpy()
                if add_scale.ndim == 0:
                    add_scale = np.array([add_scale])
                add_scales.append(add_scale)

            # Recursively extract parameters from the children modules
            extract_quant_params(submodule)

    # Start extraction from the top-level module
    extract_quant_params(quant_net)

    print(f"Total weights: {len(int_weights)}")
    print(f"Total act_scales: {len(act_scales)}")
    print(f"Total add_scales: {len(add_scales)}")
    
    mul_vals, shift_vals = [], []
    M_num=-1
    conv_idx=0
    act_idx=0 # Input quantization activation (act_scales[0])
    add_idx=0

    for detail_idx,layer in enumerate(details):
        if layer['layer_type'] in ['Conv2d', 'Linear']:
            # if next layer is shortcut
            next_is_shortcut_related=(detail_idx +1 <len(details) and 
                                 details[detail_idx + 1]['layer_type'] in ['Shortcut', 'Downsample_Conv2d'])

            if next_is_shortcut_related:
                # Shortcut 직전 main path 마지막 conv 의 scale/ add scale
                M= in_scales[conv_idx] / add_scales[add_idx]
                M_num+=1
                print(f"Mnum: {M_num} => Conv {conv_idx} (before Shortcut): M = in_scales[{conv_idx}] / add_scales[{add_idx}]")
            else:
                # 일반 conv/linear -> 바로 다음 activation
                act_idx+=1
                M = in_scales[conv_idx] / act_scales[act_idx]
                M_num+=1
                print(f"Mnum: {M_num} => Conv or Linear {conv_idx} (normal): M = in_scales[{conv_idx}] / act_scales[{act_idx}]")
            
            mul, shift = quantize_multiplier(M[0])
            mul_vals.append(mul)
            shift_vals.append(shift)
            conv_idx += 1
        
        elif layer['layer_type'] == 'Downsample_Conv2d':
            # Downsample conv → add scale 사용
            M = in_scales[conv_idx] / add_scales[add_idx]
            M_num+=1
            print(f"Mnum: {M_num} => Downsample Conv {conv_idx}: M = in_scales[{conv_idx}] / add_scales[{add_idx}]")

            mul, shift = quantize_multiplier(M[0])
            mul_vals.append(mul)
            shift_vals.append(shift)
            conv_idx += 1

        elif layer['layer_type'] == 'Shortcut':
            if not layer['has_downsample']:
                # Identity shortcut → 분기점 activation scale / add scale
                # 분기점 = main path 시작 전 activation 이 부분은 추후에 보자
                branch_act_idx = act_idx - (layer['length']-1)

                M = act_scales[branch_act_idx] / add_scales[add_idx]
                M_num+=1
                print(f"Mnum: {M_num} => Identity Shortcut: M = act_scales[{branch_act_idx}] / add_scales[{add_idx}]")

                mul, shift = quantize_multiplier(M[0])
                mul_vals.append(mul)
                shift_vals.append(shift)
            
            # Add 이후 requantization: add_scale → activation
            act_idx += 1
            M_add = add_scales[add_idx] / act_scales[act_idx]
            M_num+=1
            print(f"Mnum: {M_num} => Add requant: M = add_scales[{add_idx}] / act_scales[{act_idx}]")

            mul, shift = quantize_multiplier(M_add[0])
            mul_vals.append(mul)
            shift_vals.append(shift)

            add_idx += 1
    
    int_biases = []
    f_int_biases = []
    shift_biases = []

    for int_b in int_bias:
        shift_bias = np.clip(np.log2(abs(int_b + 1e-10)).astype(np.int32) - 6, a_max = None, a_min = 0)
        r_bias = np.right_shift(int_b, shift_bias)
        f_int_biases.append(r_bias)
        l_bias = np.left_shift(r_bias, shift_bias)
        shift_biases.append(shift_bias)
        int_biases.append(l_bias)
    
    print(f"\nTotal M values: {len(mul_vals)}")
    return int_weights, int_biases, f_int_biases, shift_biases, mul_vals, shift_vals

def extract_input(model,testloader):
    for test_imgs, _ in testloader:
        t = (torch.round(Variable(test_imgs).float()/model.quant_inp.extract_quant_act_scale().cpu()))
        t = t.detach().cpu().numpy().astype(np.int16)[0]
    t = np.expand_dims(t, axis = 0)

    return t

def save_2d_inputs(path, input):
    with open(path + '/ibex_inputs.h', 'w') as f:
        f.write('#ifndef IBEX_INPUTS_H\n#define IBEX_INPUTS_H\n\n')
        test_batch_X_cnn_new = np.transpose(input, (2, 3, 1, 0))
        dims = np.shape(test_batch_X_cnn_new)
        st = 'static const int input[' + str(dims[0]) + '][' + str(dims[1]) + '][' + str(dims[2]) + ']['
        st += str(dims[3]) + '] = {\n'
        f.write(st)
        for n in range(dims[0]):
            f.write('\t{\n')

            for m in range(dims[1]):
                f.write('\t\t{\n')

                for k in range(dims[2]):
                    f.write('\t\t\t{')
                    for l in range(dims[3]-1):
                        f.write(str(test_batch_X_cnn_new[n][m][k][l]) + ', ')
                    if(dims[3] != 1):
                        f.write(str(test_batch_X_cnn_new[n][m][k][l+1]) + '}')
                    else:
                        f.write(str(test_batch_X_cnn_new[n][m][k][0]) + '}')

                    if(k != dims[2]-1):
                        f.write(',')
                f.write('\n')

                f.write('\t\t}')
                if(m != dims[1]-1):
                    f.write(',')

                f.write('\n')

            f.write('\t}')
            if(n != dims[0]-1):
                f.write(',')
            f.write('\n')

        f.write('};\n\n\n')
        f.write('#endif /* IBEX_INPUTS_H */')
    
    return 

def save_cnn_net_params(path, int_weights, int_biases, mul_vals, shift_vals, shift_biases = None):
    wi = 0
    bi = 0
    fi = 0

    # Open a text file for writing
    with open(path + '/cnn_weights.h', 'w') as f:
        f.write('#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n\n')
        for k in range(len(int_weights)):
            dims = np.shape(int_weights[k])
            mat = int_weights[k]   
            
            if(len(dims) == 2 or ((len(dims) == 4) and dims[2] == dims[3] == 1)):
                f.write('static const int ')
                if(len(dims) == 2):
                    wi += 1
                    f.write('W' + str(wi))                
                else:
                    mat = np.squeeze(mat, axis = (2,3))
                    fi += 1
                    f.write('F' + str(fi))
                    
                st = '[' + str(dims[0]) + ']' + '[' + str(dims[1]) + '] = {\n'
                f.write(st)
                for n in range(dims[0]):
                    f.write('\t{')
                    for m in range(dims[1] - 1):
                        f.write(str(mat[n][m]) + ', ')
                    if(dims[1] == 1):
                        f.write(str(mat[n][0]) + '}')
                    else:
                        f.write(str(mat[n][m+1]) + '}')
                    if(n != dims[0]-1):
                        f.write(',')
                    f.write('\n')
                f.write('};\n\n')
            
            elif (len(dims) == 3):
                dims = np.shape(mat)
                fi += 1
                st = 'static const int F' + str(fi) + '[' + str(dims[0]) + '][' + str(dims[1])
                st += '][' + str(dims[2]) + '] = {\n'
                f.write(st)

                for n in range(dims[0]):
                    f.write('\t{\n')
                    for l in range(dims[1]):
                        f.write('\t\t{')
                        for h in range(dims[2] - 1):
                            f.write(str(mat[n][l][h]) + ', ')
                        if dims[2] != 1:
                            f.write(str(mat[n][l][dims[2] - 1]) + '}')
                        else:
                            f.write(str(mat[n][l][0]) + '}')
                        if (l != dims[1] - 1):
                            f.write(',')
                        f.write('\n')
                    f.write('\t}')
                    if n != dims[0] - 1:
                        f.write(',')
                    f.write('\n')
                f.write('};\n\n')
            
            elif(len(dims) == 4):
                mat = np.transpose(mat, (0, 2, 3, 1))
                dims = np.shape(mat)
                fi += 1
                st = 'static const int F' + str(fi) + '[' + str(dims[0]) + '][' + str(dims[1])
                st += '][' + str(dims[2]) + '][' + str(dims[3]) + '] = {\n'
                f.write(st)

                for n in range(dims[0]):
                    f.write('\t{\n')
                    for m in range(dims[1]):
                        f.write('\t\t{\n')
                        for l in range(dims[2]):
                            f.write('\t\t\t{')
                            for h in range(dims[3] - 1):
                                f.write(str(mat[n][m][l][h]) + ', ')
                            if(dims[3] != 1):
                                f.write(str(mat[n][m][l][h+1]) + '}')
                            else:
                                f.write(str(mat[n][m][l][0]) + '}')
                            if (l != dims[2]-1):
                                f.write(',')
                            f.write('\n')
                        f.write('\t\t}')
                        if (m != dims[1] - 1):
                            f.write(',')
                        f.write('\n')
                    f.write('\t}')
                    if (n != dims[0] - 1):
                        f.write(',')
                    f.write('\n')
                f.write('};\n\n')
                
        for k in range(len(int_biases)):
            dims = np.shape(int_biases[k])
            mat = int_biases[k]
            bi += 1
            st = 'static const int B' + str(bi) + '[' + str(dims[0]) + '] = {\n\t'
            f.write(st)

            for n in range(dims[0]):
                f.write(str(mat[n]))
                if(n != dims[0] - 1):
                    f.write(', ')
            f.write('\n};\n\n')

        f.write('\n')
        f.write('#endif /* CNN_WEIGHTS_H */')

    if('original' in path):
        with open(path + '/ibex_cnn_params.h', 'w') as f:
            f.write('#ifndef IBEX_CNN_PARAMS_H\n#define IBEX_CNN_PARAMS_H\n\n')
            for i, mul_v in enumerate(mul_vals):
                f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                f.write('#define SV' + str(i+1) + ' ' + str(shift_v+7) + '\n')

            f.write('\n')

            for i, mul_v in enumerate(mul_vals):
                f.write('#define SB' + str(i+1) + ' ' + str(0) + '\n')
            
            f.write('\n#endif /* IBEX_CNN_PARAMS_H */')

    else:
        bi = 0
        with open(path + '/ibex_cnn_params.h', 'w') as f:
            f.write('#ifndef IBEX_CNN_PARAMS_H\n#define IBEX_CNN_PARAMS_H\n\n')

            for i, mul_v in enumerate(mul_vals):
                f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                f.write('#define SV' + str(i+1) + ' ' + str(shift_v) + '\n')
            
            f.write('\n')
            
            for k in range(len(shift_biases)):
                dims = np.shape(shift_biases[k])
                mat = shift_biases[k]
                bi += 1
                st = 'static const int SB' + str(bi) + '[' + str(dims[0]) + '] = {\n\t'
                f.write(st)

                for n in range(dims[0]):
                    f.write(str(mat[n]))
                    if(n != dims[0] - 1):
                        f.write(', ')
                f.write('\n};\n\n')
                
            f.write('#endif /* IBEX_CNN_PARAMS_H */')

    return


def get_quantnet_details(module,details=None,act=None):

    if details is None:
        details = []
        act = []

    # residual block
    if hasattr(module,'downsample'):
        block_start_idx = len(details)

        main_layers = []
        for name, layer in module.named_children():
            if name == 'downsample':
                continue

            if isinstance(layer, (nn.Conv2d, qnn.QuantConv2d)):
                layer_info = {
                    "layer_type": "Conv2d",
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "groups": layer.groups,
                }
                details.append(layer_info)
                main_layers.append(len(details) - 1)

            # elif isinstance(layer, (nn.BatchNorm2d, qnn.BatchNorm2dToQuantScaleBias)):
            #     pass
            
            elif isinstance(layer, (nn.ReLU, nn.ReLU6, qnn.QuantReLU)):
                act.append(1)
            elif isinstance(layer, (nn.Identity,qnn.QuantIdentity)):
                act.append(0)
            
            # Recursive
            else:
                get_quantnet_details(layer, details, act)

        #downsample
        has_downsample = False
        if module.downsample is not None:
            has_downsample = True

            for ds_layer in module.downsample.children():
                if isinstance(ds_layer, (nn.Conv2d, qnn.QuantConv2d)):
                    ds_info = {
                        "layer_type": "Downsample_Conv2d",
                        "in_channels": ds_layer.in_channels,
                        "out_channels": ds_layer.out_channels,
                        "kernel_size": ds_layer.kernel_size,
                        "stride": ds_layer.stride,
                        "padding": ds_layer.padding,
                        "groups": ds_layer.groups,
                    }
                    details.append(ds_info)
                    act.append(0)
        
        details.append({
            "layer_type": "Shortcut",
            "length": len(main_layers),
            "has_downsample": has_downsample,
        })
        act.append(0)

    #basic block
    else:
        for name, layer in module.named_children():
            
            if isinstance(layer, (nn.Conv2d, qnn.QuantConv2d)):
                details.append({
                    "layer_type": "Conv2d",
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "groups": layer.groups,
                })

            elif isinstance(layer, (nn.Linear, qnn.QuantLinear)):
                details.append({
                    "layer_type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                })
            
            elif isinstance(layer, nn.MaxPool2d):
                details.append({
                    "layer_type": "MaxPool2d",
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                })
            
            elif isinstance(layer, nn.AvgPool2d):
                
                details.append({
                "layer_type": "AvgPool2d",
                "kernel_size": layer.kernel_size,
                "stride": layer.stride,
                "padding": layer.padding
                })
            
            elif isinstance(layer, (nn.ReLU, nn.ReLU6, qnn.QuantReLU)):
                act.append(1)
            
            elif isinstance(layer, nn.Identity):
                act.append(0) 
             # Recursive
            else:
                get_quantnet_details(layer, details, act)

    return details, act

def generate_og_c_code_cnn(path, name, input, cnn_details, act_details, int_weights):

    #Step 1: Pre-parse Block Structure
    block_info = {}   # Shortcut index
    conv_counter = 0  # Conv/Downsample count

    for idx,detail in enumerate(cnn_details):
        
        if detail["layer_type"] in ["Conv2d", "Downsample_Conv2d"]:
            conv_counter+=1

        if detail["layer_type"] == "Shortcut":
            length = detail["length"]
            has_ds = detail.get("has_downsample", False)

            #The index of the first Conv layer in the main branch
            main_start = idx - length - (1 if has_ds else 0)

            #Block input layer
            input_layer = main_start - 1 

            # Calculate filter index(fi) of downsample
            if has_ds:
                ds_fi = conv_counter # current fi of downsample

                # Find the fi of the input channel (number of Conv layers up to the input layer)
                if input_layer < 0:
                    input_fi = 0
                else:
                    input_fi = sum(1 for i, d in enumerate(cnn_details[:input_layer+1]) 
                                   if d["layer_type"] in ["Conv2d", "Downsample_Conv2d"])
            
            else:
                ds_fi = None
                input_fi = None
            
            # idx when layer_type shortcut
            block_info[idx]={
                'main_start': main_start,
                'input_layer': input_layer,
                'ds_fi': ds_fi,
                'input_fi': input_fi,
                'has_downsample': has_ds,
                'ds_layer_idx': idx - 1 if has_ds else None

            }

    with open(path + '/' + name + '.c', 'w') as f:
        # ========== Header file ==========
        f.write('#include "simple_system_common.h"\n')
        f.write('#include "cnn_weights.h"\n')
        f.write('#include "fully_connected.h"\n')
        f.write('#include "ibex_cnn_params.h"\n')
        f.write('#include "ibex_inputs.h"\n')
        f.write('#include "conv2d.h"\n')

        # check if DWS conv is needed
        for detail in cnn_details[:-1]:
            if detail["layer_type"] in ["Conv2d", "Downsample_Conv2d"]:
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    f.write('#include "dws_conv.h"\n')
                    break
        
        f.write('\n')

        # ========== Basic Macro Definition ==========
        f.write('#define IMG_SZ ' + str(np.shape(input)[2]) + '\n')
        f.write('#define NUM_FIL0 ' + str(np.shape(input)[1]) + '\n\n')

        # Filter Size Definition
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4):
                f.write('#define FILTER' + str(i) + ' ' + str(w.shape[2]) + '\n')
                i += 1
        f.write('\n')

        #NUM_FIL Definition
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4):
                f.write('#define NUM_FIL' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1
        f.write('\n')

        # Stride Definition
        i = 1
        for detail in cnn_details:
            if detail["layer_type"] in ["Conv2d", "Downsample_Conv2d"]:
                 f.write('#define STRIDE' + str(i) + ' ' + str(detail["stride"][0]) + '\n')
                 i += 1
        f.write('\n')

        # Padding Definition
        i = 1
        for detail in cnn_details:
            if detail["layer_type"] in ["Conv2d", "Downsample_Conv2d"]:
                if(detail["padding"] == 'same'):
                    f.write('#define PAD_TB' + str(i) + ' ' + str((detail["kernel_size"][0] - 1)//2) + '\n')
                    f.write('#define PAD_LR' + str(i) + ' ' + str((detail["kernel_size"][0] - 1)//2) + '\n')
                elif(detail["padding"] == 'valid'):
                    f.write('#define PAD_TB' + str(i) + ' 0\n')
                    f.write('#define PAD_LR' + str(i) + ' 0\n')
                else:
                    f.write('#define PAD_TB' + str(i) + ' ' + str(detail["padding"][0]) + '\n')
                    f.write('#define PAD_LR' + str(i) + ' ' + str(detail["padding"][0]) + '\n')
                f.write('\n')
                i += 1

        # POOL Definition
        i = 1
        for detail in cnn_details:
            if ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('#define POOL_STRIDE' + str(i) + ' ' + str(detail["stride"]) + '\n')
                f.write('#define POOL_SIZE' + str(i) + ' ' + str(detail["kernel_size"]) + '\n')
                f.write('\n')
                i += 1

        # DENSE 정의
        i = 1
        for w in int_weights[:-1]:
            if(len(np.shape(w)) == 2):
                f.write('#define DENSE_DIM' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1

        f.write('#define OUT_DIM ' + str(int_weights[-1].shape[0]) + '\n\n')
        f.write('#define SAMPLES 1\nint outs[SAMPLES][OUT_DIM];\n\n')
        f.write('void ' + name + '() {\n\n')

        # ========== Dimension Calculation ==========
        i = 1  # cnn_details (1,2,3...)
        fi = 1 # filter index
        st = 1
        flatten = 0

        for idx, detail in enumerate(cnn_details):
            if detail["layer_type"] == "Conv2d":
                f.write('\tint dout' + str(i) + ' = NUM_FIL' + str(fi) + ';\n')
                if(i == 1):
                    f.write('\tint hout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_TB1)/STRIDE1) + 1;\n')
                    f.write('\tint wout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_LR1)/STRIDE1) + 1;\n')
                else:
                    f.write('\tint hout' + str(i) + ' = ((hout' + str(i-1) + ' - FILTER' + str(fi))
                    f.write(' + 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                    f.write('\tint wout' + str(i) + ' = ((wout' + str(i-1) + ' - FILTER' + str(fi))
                    f.write(' + 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                fi += 1

            elif detail["layer_type"] == "Downsample_Conv2d":
                # Find the Input Layer in the Block Information
                input_layer_idx = None
                for shortcut_idx, info in block_info.items():
                    if info['ds_layer_idx'] == idx:
                        input_layer_idx = info['input_layer']
                        break
            
                f.write('\tint dout' + str(i) + '_ds = NUM_FIL' + str(fi) + ';\n')

                if input_layer_idx is None or input_layer_idx < 0:
                    # IMG input
                    f.write('\tint hout' + str(i) + '_ds = ((IMG_SZ - FILTER' + str(fi))
                    f.write(' + 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                    f.write('\tint wout' + str(i) + '_ds = ((IMG_SZ - FILTER' + str(fi))
                    f.write(' + 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                
                else:
                    c_input_idx = input_layer_idx + 1
                    f.write('\tint hout' + str(i) + '_ds = ((hout' + str(c_input_idx) + ' - FILTER' + str(fi))
                    f.write(' + 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                    f.write('\tint wout' + str(i) + '_ds = ((wout' + str(c_input_idx) + ' - FILTER' + str(fi))
                    f.write(' + 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('\tint dout' + str(i) + ' = dout' + str(i-1) + ';\n')
                f.write('\tint hout' + str(i) + ' = ((hout' + str(i-1) + ' - POOL_SIZE' + str(st) + ')/POOL_STRIDE' + str(st) + ') + 1;\n')
                f.write('\tint wout' + str(i) + ' = ((wout' + str(i-1) + ' - POOL_SIZE' + str(st) + ')/POOL_STRIDE' + str(st) + ') + 1;\n')
                st += 1

            elif(detail["layer_type"] == "Shortcut"):
                if detail.get("has_downsample"):
                    ds_idx = i - 1
                    f.write('\tint dout' + str(i) + ' = dout' + str(ds_idx) + '_ds;\n') 
                    f.write('\tint hout' + str(i) + ' = hout' + str(ds_idx) + '_ds;\n')
                    f.write('\tint wout' + str(i) + ' = wout' + str(ds_idx) + '_ds;\n')
                else:
                    f.write('\tint dout' + str(i) + ' = dout' + str(i-1) + ';\n')
                    f.write('\tint hout' + str(i) + ' = hout' + str(i-1) + ';\n')
                    f.write('\tint wout' + str(i) + ' = wout' + str(i-1) + ';\n')
            
            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\tint flatten_dim = dout' + str(i-1) + ' * hout' + str(i-1) + ' * wout' + str(i-1) + ';\n')
                    flatten = 1
                break
            
            f.write('\n')
            i += 1
        
        f.write('\n')


        # ========== Memory Allocation ==========
        i = 1
        fi = 1
        dn = 1
        flatten = 0

        f.write('\tint in[IMG_SZ][IMG_SZ][NUM_FIL0];\n')
        f.write('\tint inp_dim[3] = {IMG_SZ, IMG_SZ, NUM_FIL0};\n\n')

        for idx, detail in enumerate(cnn_details):
            if detail["layer_type"] == "Conv2d":
                f.write('\tint out' + str(i) + '[hout' + str(i) + '][wout' + str(i) + '][dout' + str(i) + '];\n')
                f.write('\tint pad_' + str(i) + '[4] = {PAD_TB' + str(fi) + ', PAD_TB' + str(fi))
                f.write(', PAD_LR' + str(fi) + ', PAD_LR' + str(fi) + '};\n')
                f.write('\tint outp_dim' + str(i) + '[3] = {hout' + str(i) + ', wout' + str(i))
                f.write(', dout' + str(i) + '};\n')
                f.write('\tint f_dim' + str(i) + '[4] = {NUM_FIL' + str(fi) + ', FILTER' + str(fi))
                f.write(', FILTER' + str(fi) + ', NUM_FIL' + str(fi-1) + '};\n')
                fi += 1
            
            elif detail["layer_type"] == "Downsample_Conv2d":
                input_fi = None
                for shortcut_idx, info in block_info.items():
                    if info['ds_layer_idx'] == idx:
                        input_fi = info['input_fi']
                        break

                f.write('\tint out' + str(i) + '_ds[hout' + str(i) + '_ds][wout' + str(i) + '_ds][dout' + str(i) + '_ds];\n')
                f.write('\tint pad_' + str(i) + '_ds[4] = {PAD_TB' + str(fi) + ', PAD_TB' + str(fi))
                f.write(', PAD_LR' + str(fi) + ', PAD_LR' + str(fi) + '};\n')
                f.write('\tint outp_dim' + str(i) + '_ds[3] = {hout' + str(i) + '_ds, wout' + str(i) + '_ds')
                f.write(', dout' + str(i) + '_ds};\n')
                f.write('\tint f_dim' + str(i) + '_ds[4] = {NUM_FIL' + str(fi) + ', FILTER' + str(fi))
                f.write(', FILTER' + str(fi) + ', NUM_FIL' + str(input_fi) + '};\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d") or detail["layer_type"] == "Shortcut"):
                f.write('\tint out' + str(i) + '[hout' + str(i) + '][wout' + str(i) + '][dout' + str(i) + '];\n')
                f.write('\tint outp_dim' + str(i) + '[3] = {hout' + str(i) + ', wout' + str(i))
                f.write(', dout' + str(i) + '};\n')

            elif detail["layer_type"] == "Linear":
                if flatten == 0:
                    f.write('\tint out' + str(i) + '[flatten_dim];\n')
                    flatten = 1
                else:
                    f.write('\tint out' + str(i) + '[DENSE_DIM' + str(dn) + '];\n')
                    dn += 1
            
            f.write('\n')
            i += 1
        
        f.write('\n\tint out[OUT_DIM];\n\n\tfor (int iter = 0; iter < SAMPLES; iter++){\n\n')

        # ========== Copy Input Data ==========
        f.write('\t\tfor(int i = 0; i < IMG_SZ; i++){\n')
        f.write('\t\t\tfor(int j = 0; j < IMG_SZ; j++){\n')
        f.write('\t\t\t\tfor(int k = 0; k < NUM_FIL0; k++){\n')
        f.write('\t\t\t\t\tin[i][j][k] = input[i][j][k][iter];\n')
        f.write('\t\t\t\t}\n\t\t\t}\n\t\t}\n\n\t\tpcount_enable(1);\n\n')

        # ========== Execute Layer ==========
        i = 1
        fi = 1
        st = 1
        dn = 1
        flatten = 0
        act_num = 0

        for idx, detail in enumerate(cnn_details[:-1]):
            
            if detail["layer_type"] == "Conv2d":
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    conv_type = 'dw_conv'
                elif(detail["kernel_size"][0] == 1):
                    conv_type = 'pw_conv'
                else:
                    conv_type = "conv2"

                
                relu_flag = act_details[act_num] if act_num < len(act_details) else 0

                if(i == 1):
                    f.write('\t\t' + conv_type + '(inp_dim, f_dim1, outp_dim1, in, F1, B1, ')
                    f.write('out1, STRIDE1, pad_1, SB1, MV1, SV1, ' + str(relu_flag) + ');\n')
                else:
                    f.write('\t\t' + conv_type + '(outp_dim' + str(i-1) + ', f_dim' + str(i) + ', outp_dim' + str(i))
                    f.write(', out' + str(i-1) + ', F' + str(fi) + ', B' + str(fi) + ', out' + str(i))
                    f.write(', STRIDE' + str(fi) + ', pad_' + str(i) + ', SB' + str(fi))
                    f.write(', MV' + str(fi) + ', SV' + str(fi) + ', ' + str(relu_flag) + ');\n')

                fi += 1
                act_num += 1
            
            elif detail["layer_type"] == "Downsample_Conv2d":
                conv_type = 'pw_conv' if detail["kernel_size"][0] == 1 else 'conv2'

                input_layer_idx = None
                for shortcut_idx, info in block_info.items():
                    if info['ds_layer_idx'] == idx:
                        input_layer_idx = info['input_layer']
                        break
                
                if input_layer_idx is None or input_layer_idx < 0:
                    input_name = 'in'
                    input_dim = 'inp_dim'
                else:
                    c_input_idx = input_layer_idx + 1
                    input_name = 'out' + str(c_input_idx)
                    input_dim = 'outp_dim' + str(c_input_idx)
                
                f.write('\t\t// Downsample branch (parallel with main branch)\n')
                f.write('\t\t' + conv_type + '(' + input_dim + ', f_dim' + str(i) + '_ds, outp_dim' + str(i) + '_ds')
                f.write(', ' + input_name + ', F' + str(fi) + ', B' + str(fi) + ', out' + str(i) + '_ds')
                f.write(', STRIDE' + str(fi) + ', pad_' + str(i) + '_ds, SB' + str(fi))
                f.write(', MV' + str(fi) + ', SV' + str(fi) + ', 0);\n')
                
                fi += 1
                act_num += 1
            
            elif detail["layer_type"] == "MaxPool2d":
                f.write('\t\tmaxpool2(outp_dim' + str(i-1) + ', outp_dim' + str(i))
                f.write(', out' + str(i-1) + ', out' + str(i) + ', POOL_SIZE' + str(st) + ', POOL_STRIDE')
                f.write(str(st) + ');\n')
                st += 1
            
            elif(detail["layer_type"] == "AvgPool2d"):
                f.write('\t\tavgpool2(outp_dim' + str(i-1) + ', outp_dim' + str(i))
                f.write(', out' + str(i-1) + ', out' + str(i) + ', POOL_SIZE' + str(st) + ', POOL_STRIDE')
                f.write(str(st) + ');\n')
                st += 1
            
            elif detail["layer_type"] == "Shortcut":
                if detail.get("has_downsample"):
                    ds_idx = i - 1
                    main_idx = ds_idx - 1
                    f.write('\t\tshortcut(outp_dim' + str(i) + ', out' + str(main_idx))
                    f.write(', out' + str(ds_idx) + '_ds, out' + str(i) + ');\n')
                
                else:
                    shortcut_length = detail["length"]
                    f.write('\t\tshortcut(outp_dim' + str(i) + ', out' + str(i-1))
                    f.write(', out' + str(i-shortcut_length-1) + ', out' + str(i) + ');\n')
            
            elif detail["layer_type"] == "Linear":
                relu_flag = act_details[act_num] if act_num < len(act_details) else 0

                if flatten == 0:
                    f.write('\t\tflatten(outp_dim' + str(i-1) + ', out' + str(i-1) + ', out' + str(i) + ');\n\n')
                    i += 1
                    f.write('\t\tmlp_layer(out' + str(i-1) + ', out' + str(i) + ', flatten_dim, DENSE_DIM1')
                    f.write(', W1, B' + str(fi + dn - 1) +  ', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
                    f.write(', SV' + str(fi + dn - 1) + ', ' + str(relu_flag) + ');\n')
                    dn += 1
                    flatten = 1
                else:
                    f.write('\t\tmlp_layer(out' + str(i-1) + ', out' + str(i) + ', DENSE_DIM' + str(dn-1))
                    f.write(', DENSE_DIM' + str(dn) + ', W' + str(dn) + ', B' + str(fi + dn - 1))
                    f.write(', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
                    f.write(', SV' + str(fi + dn - 1) + ', ' + str(relu_flag) + ');\n')
                    dn += 1
                
                act_num += 1

            f.write('\n')
            i += 1
        
        # ========== final layer ==========
        if flatten == 0:
            f.write('\t\tflatten(outp_dim' + str(i-1) + ', out' + str(i-1) + ', out' + str(i) + ');\n\n')
            i += 1
            f.write('\t\tmlp_layer(out' + str(i-1) + ', out, flatten_dim, OUT_DIM, ')
            f.write('W1, B' + str(fi + dn - 1) +  ', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
            f.write(', SV' + str(fi + dn - 1) + ', 1);\n')
        else:
            f.write('\t\tmlp_layer(out' + str(i-1) + ', out, DENSE_DIM' + str(dn-1))
            f.write(', OUT_DIM, W' + str(dn) + ', B' + str(fi + dn - 1))
            f.write(', SB' + str(fi + dn - 1) + ', MV' + str(fi + dn - 1))
            f.write(', SV' + str(fi + dn - 1) + ', 1);\n')

        
        # ========== output ==========
        f.write('\n\t\tpcount_enable(0);\n\n')
        f.write('\t\tputs("Output Layer Values:\\n");\n')
        f.write('\t\tfor(int i = 0; i < OUT_DIM; i++) {\n')
        f.write('\t\t\tputhex(out[i]);\n')
        f.write('\t\t\tputs("\\n");\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('}\n\n')
        
        f.write('int main(void) {\n\n')
        f.write('\tpcount_enable(0);\n\n')
        f.write('\t' + name + '();\n\n')
        f.write('\treturn 0;\n}')

    return
            



