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

def get_int_params(quant_net):
    
    int_weights = []
    int_bias = []
    in_scales = []
    act_scales = []
    
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

            # Recursively extract parameters from the children modules
            extract_quant_params(submodule)

    # Start extraction from the top-level module
    extract_quant_params(quant_net)
    
    mul_vals, shift_vals = [], []
    
    for i in range(len(act_scales)-1):
        M = in_scales[i]/act_scales[i+1]
        mul, shift = quantize_multiplier(M[0])
        mul_vals.append(mul)
        shift_vals.append(shift)
      
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
    with open(path + '/' + name + '.c', 'w') as f:
        # ========== 헤더 파일 ==========
        f.write('#include "simple_system_common.h"\n')
        f.write('#include "cnn_weights.h"\n')
        f.write('#include "fully_connected.h"\n')
        f.write('#include "ibex_cnn_params.h"\n')
        f.write('#include "ibex_inputs.h"\n')
        f.write('#include "conv2d.h"\n')

        # DWS conv 필요 여부 체크
        for detail in cnn_details[:-1]:
            if detail["layer_type"] in ["Conv2d", "Downsample_Conv2d"]:
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    f.write('#include "dws_conv.h"\n')
                    break
        
        f.write('\n')
        
        # ========== 기본 매크로 정의 ==========
        f.write('#define IMG_SZ ' + str(np.shape(input)[2]) + '\n')
        f.write('#define NUM_FIL0 ' + str(np.shape(input)[1]) + '\n\n')
        
        # FILTER 크기 정의
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4):
                f.write('#define FILTER' + str(i) + ' ' + str(w.shape[2]) + '\n')
                i += 1
        f.write('\n')
        
        # NUM_FIL 정의
        i = 1
        for w in int_weights:
            if(len(np.shape(w)) == 4):
                f.write('#define NUM_FIL' + str(i) + ' ' + str(w.shape[0]) + '\n')
                i += 1
        f.write('\n')

        # STRIDE 정의
        i = 1
        for detail in cnn_details:
            if detail["layer_type"] in ["Conv2d", "Downsample_Conv2d"]:
                f.write('#define STRIDE' + str(i) + ' ' + str(detail["stride"][0]) + '\n')
                i += 1
        f.write('\n')

        # PADDING 정의
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

        # POOL 정의
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

        # ========== 차원 계산 ==========
        i = 1
        fi = 1
        st = 1
        flatten = 0
        ds_indices = {}  # downsample 레이어 인덱스 저장
        block_start = {}  # 각 block의 시작 인덱스

        for idx, detail in enumerate(cnn_details):
            if detail["layer_type"] == "Conv2d":
                f.write('\tint dout' + str(i) + ' = NUM_FIL' + str(fi) + ';\n')
                if(i == 1):
                    f.write('\tint hout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_TB1)/STRIDE1) + 1;\n')
                    f.write('\tint wout' + str(i) + ' = ((IMG_SZ - FILTER1 + 2 * PAD_LR1)/STRIDE1) + 1;\n')
                else:
                    f.write('\tint hout' + str(i) + ' = ((hout' + str(i-1) + ' - FILTER' + str(fi))
                    f.write('+ 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                    f.write('\tint wout' + str(i) + ' = ((wout' + str(i-1) + ' - FILTER' + str(fi))
                    f.write('+ 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                fi += 1
                
                # Block 시작점 저장 (Shortcut 직전의 여러 Conv 중 첫 번째)
                if idx + 1 < len(cnn_details):
                    next_detail = cnn_details[idx + 1]
                    # 다음이 Conv이면 아직 block 중간
                    if next_detail["layer_type"] not in ["Conv2d"]:
                        # Shortcut 길이만큼 뒤로 가서 block 시작점 찾기
                        for j in range(idx + 1, len(cnn_details)):
                            if cnn_details[j]["layer_type"] == "Shortcut":
                                block_length = cnn_details[j]["length"]
                                block_start[j] = i - block_length + 1
                                break
            
            elif detail["layer_type"] == "Downsample_Conv2d":
                ds_indices[i] = True
                f.write('\tint dout' + str(i) + '_ds = NUM_FIL' + str(fi) + ';\n')
                
                # Downsample은 해당 block의 시작점에서 계산
                # Shortcut을 찾아서 block 시작점 얻기
                shortcut_idx = None
                for j in range(idx + 1, len(cnn_details)):
                    if cnn_details[j]["layer_type"] == "Shortcut":
                        shortcut_idx = j
                        break
                
                if shortcut_idx in block_start:
                    start_idx = block_start[shortcut_idx]
                else:
                    start_idx = i - cnn_details[shortcut_idx]["length"]
                
                f.write('\tint hout' + str(i) + '_ds = ((hout' + str(start_idx-1) + ' - FILTER' + str(fi))
                f.write('+ 2 * PAD_TB' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                f.write('\tint wout' + str(i) + '_ds = ((wout' + str(start_idx-1) + ' - FILTER' + str(fi))
                f.write('+ 2 * PAD_LR' + str(fi) + ')/STRIDE' + str(fi) + ')+1;\n')
                fi += 1
            
            elif ((detail["layer_type"] == "MaxPool2d") or (detail["layer_type"] == "AvgPool2d")):
                f.write('\tint dout' + str(i) + ' = dout' + str(i-1) + ';\n')
                f.write('\tint hout' + str(i) + ' = ((hout' + str(i-1) + ' - POOL_SIZE' + str(st) + ')/POOL_STRIDE' + str(st) + ') + 1;\n')
                f.write('\tint wout' + str(i) + ' = ((wout' + str(i-1) + ' - POOL_SIZE' + str(st) + ')/POOL_STRIDE' + str(st) + ') + 1;\n')
                st += 1
            
            elif(detail["layer_type"] == "Shortcut"):
                
                if detail.get("has_downsample"):
                    # Downsample 레이어의 차원 사용
                    ds_idx = i - 1  # 바로 직전이 downsample
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
        
        # ========== 메모리 할당 ==========
        i = 1  #Layer Index
        fi = 1 #Filter Index
        dn = 1 #Dense/Linear Index
        flatten = 0 #Flatten Flag

        f.write('\tint in[IMG_SZ][IMG_SZ][NUM_FIL0];\n')
        f.write('\tint inp_dim[3] = {IMG_SZ, IMG_SZ, NUM_FIL0};\n\n')

        for detail in cnn_details:
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
                f.write('\tint out' + str(i) + '_ds[hout' + str(i) + '_ds][wout' + str(i) + '_ds][dout' + str(i) + '_ds];\n')
                f.write('\tint pad_' + str(i) + '_ds[4] = {PAD_TB' + str(fi) + ', PAD_TB' + str(fi))
                f.write(', PAD_LR' + str(fi) + ', PAD_LR' + str(fi) + '};\n')
                f.write('\tint outp_dim' + str(i) + '_ds[3] = {hout' + str(i) + '_ds, wout' + str(i) + '_ds')
                f.write(', dout' + str(i) + '_ds};\n')
                
                # f_dim의 in_channels는 block 시작점의 채널 사용
                shortcut_idx = None
                for j in range(len(cnn_details)):
                    if j > cnn_details.index(detail) and cnn_details[j]["layer_type"] == "Shortcut":
                        shortcut_idx = j
                        break
                
                if shortcut_idx and shortcut_idx in block_start:
                    start_fi = fi - cnn_details[shortcut_idx]["length"] - 1
                else:
                    start_fi = fi - 1
                
                f.write('\tint f_dim' + str(i) + '_ds[4] = {NUM_FIL' + str(fi) + ', FILTER' + str(fi))
                f.write(', FILTER' + str(fi) + ', NUM_FIL' + str(start_fi) + '};\n')
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
       
        # ========== 입력 데이터 복사 ==========
        f.write('\t\tfor(int i = 0; i < IMG_SZ; i++){\n')
        f.write('\t\t\tfor(int j = 0; j < IMG_SZ; j++){\n')
        f.write('\t\t\t\tfor(int k = 0; k < NUM_FIL0; k++){\n')
        f.write('\t\t\t\t\tin[i][j][k] = input[i][j][k][iter];\n')
        f.write('\t\t\t\t}\n\t\t\t}\n\t\t}\n\n\t\tpcount_enable(1);\n\n')

        # ========== 레이어 실행 ==========
        i = 1
        fi = 1
        st = 1
        dn = 1
        flatten = 0
        act_num = 0
        shortcut_start_id = 1  # Block 시작점 추적

        for idx, detail in enumerate(cnn_details[:-1]):
            if detail["layer_type"] == "Conv2d":
                # Conv 타입 결정
                if(detail["in_channels"] == detail["out_channels"] == detail["groups"] != 1):
                    conv_type = 'dw_conv'
                elif(detail["kernel_size"][0] == 1):
                    conv_type = 'pw_conv'
                else:
                    conv_type = "conv2"
                
                # ReLU 플래그
                relu_flag = act_details[act_num] if act_num < len(act_details) else 0
                
                if(i == 1):
                    f.write('\t\t' + conv_type + '(inp_dim, f_dim1, outp_dim1, in, F1, B1, ')
                    f.write('out1, STRIDE1, pad_1, SB1, MV1, SV1, ' + str(relu_flag) + ');\n')
                    shortcut_start_id = 1
                else:
                    f.write('\t\t' + conv_type + '(outp_dim' + str(i-1) + ', f_dim' + str(i) + ', outp_dim' + str(i))
                    f.write(', out' + str(i-1) + ', F' + str(fi) + ', B' + str(fi) + ', out' + str(i))
                    f.write(', STRIDE' + str(fi) + ', pad_' + str(i) + ', SB' + str(fi))
                    f.write(', MV' + str(fi) + ', SV' + str(fi) + ', ' + str(relu_flag) + ');\n')
                
                fi += 1
                act_num += 1
                
                # 다음이 Downsample이면 이게 block 시작점
                if idx + 1 < len(cnn_details):
                    for j in range(idx + 1, len(cnn_details)):
                        if cnn_details[j]["layer_type"] == "Shortcut":
                            block_length = cnn_details[j]["length"]
                            if i >= block_length:
                                shortcut_start_id = i - block_length + 1
                            break
            
            elif detail["layer_type"] == "Downsample_Conv2d":
                # Downsample은 항상 1x1 conv
                conv_type = 'pw_conv' if detail["kernel_size"][0] == 1 else 'conv2'

                shortcut_idx = None
                for j in range(idx + 1, len(cnn_details)):
                    if cnn_details[j]["layer_type"] == "Shortcut":
                        shortcut_idx = j
                        break

                if shortcut_idx and shortcut_idx in block_start:
                    input_idx = block_start[shortcut_idx] - 1
                else:
                    input_idx = i - cnn_details[shortcut_idx]["length"] - 1
                
                f.write('\t\t// Downsample branch (parallel with main branch)\n')
                f.write('\t\t' + conv_type + '(outp_dim' + str(input_idx) + ', f_dim' + str(i) + '_ds, outp_dim' + str(i) + '_ds')
                f.write(', out' + str(input_idx) + ', F' + str(fi) + ', B' + str(fi) + ', out' + str(i) + '_ds')
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
                shortcut_length = detail["length"]
                
                if detail.get("has_downsample"):
                    # Downsample 결과와 add
                    ds_idx = i - 1
                    main_idx = ds_idx - 1
                    f.write('\t\tshortcut(outp_dim' + str(i) + ', out' + str(main_idx))
                    f.write(', out' + str(ds_idx) + '_ds, out' + str(i) + ');\n')
                else:
                    # Identity shortcut
                    f.write('\t\tshortcut(outp_dim' + str(i) + ', out' + str(i-1))
                    f.write(', out' + str(i-shortcut_length-1) + ', out' + str(i) + ');\n')
                
                shortcut_start_id = i + 1  # 다음 block 시작

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

        # ========== 최종 레이어 ==========
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

        # ========== 출력 ==========
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

