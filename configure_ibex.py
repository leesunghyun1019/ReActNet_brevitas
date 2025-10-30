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
    weight_bit_width = []  # 각 레이어의 bit 추적 binary 또는 int8
    int_biases = []

    def extract_quant_params(module):

        for name,submodule in module.named_children():

             # Check if the submodule has weights and append them if present
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                int_weights.append(submodule.int_weight().cpu().detach().numpy())
                
                bias_int = submodule.int_bias() #BatchNorm2d의 bias 추출
                if bias_int is not None:
                    int_biases.append(bias_int.cpu().detach().numpy())

                weight_scales.append(submodule.quant_weight_scale().cpu().detach().numpy())
                # check weight_bit_width (1bit or 8bit)
                is_binary = False
                if hasattr(submodule, 'weight_quant') and hasattr(submodule.weight_quant, 'bit_width'):
                    is_binary = (submodule.weight_quant.bit_width().item() == 1)
                weight_bit_width.append( 1 if is_binary else 8 )

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
            print(f"not scalar M의 shape :  {M.shape}") #(64,) #(128,) 나옴
            channel_muls = []
            channel_shifts = []
            for j in range(M.shape[0]):
                mul, shift = quantize_multiplier(M[j])
                channel_muls.append(mul)
                channel_shifts.append(shift)
            
            mul_vals.append(np.array(channel_muls))
            shift_vals.append(np.array(channel_shifts))

    return int_weights,weight_bit_width,mul_vals,shift_vals,int_biases

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


def save_cnn_net_params(path, int_weights,mul_vals, shift_vals, int_biases):
    
    wi = 0
    bi = 0
    fi = 0
    bni = 0  

    # Open a text file for writing
    with open(path + '/cnn_weights.h', 'w') as f:
        f.write('#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n\n')
        
        for k in range(len(int_weights)):
            dims = np.shape(int_weights[k])
            mat = int_weights[k]   
            
            
            # 1D 배열 - BatchNorm 파라미터
            if len(dims) == 1:
                bni += 1
                f.write('static const int BN' + str(bni) + '[' + str(dims[0]) + '] = {\n\t')
                for n in range(dims[0] - 1):
                    f.write(str(mat[n]) + ', ')
                f.write(str(mat[dims[0]-1]) + '\n};\n\n')

            
            elif(len(dims) == 2 or ((len(dims) == 4) and dims[2] == dims[3] == 1)):
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

        
        for k in range(len(int_biases)): #BatchNorm의 bias
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

    # MV,SV 저장
    if('original' in path):
        with open(path + '/ibex_cnn_params.h', 'w') as f:
            f.write('#ifndef IBEX_CNN_PARAMS_H\n#define IBEX_CNN_PARAMS_H\n\n')
            for i, mul_v in enumerate(mul_vals):
                if isinstance(mul_v,(int,float)) or mul_v.ndim == 0: #scalar
                    f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
                else: #배열
                    dims = mul_v.shape[0]
                    f.write('static const int MV' + str(i+1) + '[' + str(dims) + '] = {\n\t')

                    for j in range(dims - 1):
                        f.write(str(mul_v[j]) + ', ')
                    f.write(str(mul_v[dims-1]) + '\n};\n\n')
            
            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                if isinstance(shift_v, (int, float)) or shift_v.ndim == 0:  # 스칼라
                    f.write('#define SV' + str(i+1) + ' ' + str(shift_v + 7) + '\n')
                else: #배열
                    dims = shift_v.shape[0]
                    f.write('static const int SV' + str(i+1) + '[' + str(dims) + '] = {\n\t')

                    for j in range(dims - 1):
                        f.write(str(shift_v[j] + 7) + ', ')
                    f.write(str(shift_v[dims-1] + 7) + '\n};\n\n')


            f.write('\n')

            
            f.write('\n#endif /* IBEX_CNN_PARAMS_H */')

    else:
        bi = 0
        with open(path + '/ibex_cnn_params.h', 'w') as f:
            f.write('#ifndef IBEX_CNN_PARAMS_H\n#define IBEX_CNN_PARAMS_H\n\n')

            for i, mul_v in enumerate(mul_vals):
                if isinstance(mul_v,(int,float)) or mul_v.ndim == 0: #scalar
                    f.write('#define MV' + str(i+1) + ' ' + str(mul_v) + '\n')
                else:
                    dims = mul_v.shape[0]
                    f.write('static const int MV' + str(i+1) + '[' + str(dims) + '] = {\n\t')

                    for j in range(dims - 1):
                        f.write(str(mul_v[j]) + ', ')
                    f.write(str(mul_v[dims-1]) + '\n};\n\n')

            f.write('\n')
            
            for i, shift_v in enumerate(shift_vals):
                if isinstance(shift_v, (int, float)) or shift_v.ndim == 0:  # 스칼라
                    f.write('#define SV' + str(i+1) + ' ' + str(shift_v) + '\n')
                else: #배열
                    dims = shift_v.shape[0]
                    f.write('static const int SV' + str(i+1) + '[' + str(dims) + '] = {\n\t')

                    for j in range(dims - 1):
                        f.write(str(shift_v[j] + 7) + ', ')
                    f.write(str(shift_v[dims-1] + 7) + '\n};\n\n')
            
            f.write('\n')
            f.write('#endif /* IBEX_CNN_PARAMS_H */')

    return

def generate_Makefile(path, name):
    with open(path + '/Makefile', 'w') as f:
        f.write('# Copyright lowRISC contributors.\n')
        f.write('# Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
        f.write('# SPDX-License-Identifier: Apache-2.0\n')
        f.write('#\n# Generate a baremetal application\n\n')
        f.write('# Name of the program $(PROGRAM).c will be added as a source file\n\n')

        f.write('PROGRAM = ' + name + '\n')
        f.write('PROGRAM_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))\n')
        f.write('# Any extra source files to include in the build. Use the upper case .S\n')
        f.write('# extension for assembly files\nEXTRA_SRCS :=\n\n')
        f.write('include ${PROGRAM_DIR}/../../common/common.mk')

    shutil.copy(path + '/Makefile', path + '/../optimized')
    return



