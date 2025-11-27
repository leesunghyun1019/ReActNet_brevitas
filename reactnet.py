import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, SignedBinaryActPerTensorConst
from brevitas.core.quant import QuantType
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.stats import StatsOp

import mpq_quantize

# ============================================================
# Quantizer 정의
# ============================================================

class BinaryWeightPerTensor(Int8WeightPerTensorFloat):
    quant_type = QuantType.BINARY
    bit_width = 1
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.AVE
    scaling_per_output_channel = False
    narrow_range = False
    restrict_scaling_type = RestrictValueType.LOG_FP


# ============================================================
# 기본 컴포넌트
# ============================================================

stage_out_channel = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        return x + self.bias


# ============================================================
# BasicBlock
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        # === 첫 번째 branch (3x3 conv) ===
        self.move11 = LearnableBias(inplanes)
        self.binary_act1 = qnn.QuantIdentity(act_quant=SignedBinaryActPerTensorConst)
        self.binary_3x3 = qnn.QuantConv2d(
            inplanes, inplanes,
            kernel_size=3, stride=stride, padding=1, bias=False,
            weight_quant=BinaryWeightPerTensor,
            input_quant=None,
            output_quant=None
        )
        self.residual_add1 = qnn.QuantEltwiseAdd(
            input_quant=None,
            output_quant=None,
            return_quant_tensor=False
        )

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        # === 두 번째 branch (1x1 conv) ===
        self.move21 = LearnableBias(inplanes)
        self.binary_act2 = qnn.QuantIdentity(act_quant=SignedBinaryActPerTensorConst)

        if inplanes == planes:
            self.binary_pw = qnn.QuantConv2d(
                inplanes, planes,
                kernel_size=1, stride=1, padding=0, bias=False,
                weight_quant=BinaryWeightPerTensor,
                input_quant=None,
                output_quant=None
            )
            self.residual_add2 = qnn.QuantEltwiseAdd(
            input_quant=None,
            output_quant=None,
            return_quant_tensor=False
            )

        else:
            self.binary_pw_down1 = qnn.QuantConv2d(
                inplanes, inplanes,
                kernel_size=1, stride=1, padding=0, bias=False,
                weight_quant=BinaryWeightPerTensor,
                input_quant=None,
                output_quant=None
            )
            self.binary_pw_down2 = qnn.QuantConv2d(
                inplanes, inplanes,
                kernel_size=1, stride=1, padding=0, bias=False,
                weight_quant=BinaryWeightPerTensor,
                input_quant=None,
                output_quant=None
            )
            self.residual_add2_1 = qnn.QuantEltwiseAdd(
                input_quant=None,
                output_quant=None,
                return_quant_tensor=False
                )

            self.residual_add2_2 = qnn.QuantEltwiseAdd(
                input_quant=None,
                output_quant=None,
                return_quant_tensor=False
            )

            self.concat = qnn.QuantCat(
                input_quant=None,
                output_quant=None,
                return_quant_tensor=False
            )

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        if self.stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # === 첫 번째 branch ===
        out1 = self.move11(x)
        out1 = self.binary_act1(out1)
        out1 = self.binary_3x3(out1)

        shortcut = self.pooling(x) if self.stride == 2 else x
        out1 = self.residual_add1(out1, shortcut)

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        # === 두 번째 branch ===
        out2 = self.move21(out1)
        out2 = self.binary_act2(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.residual_add2(out2, out1)
        else:
            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.residual_add2_1(out2_1, out1)
            out2_2 = self.residual_add2_2(out2_2, out1)
            out2 = self.concat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

# ============================================================
# ReActNet
# ============================================================

class ReActNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ReActNet, self).__init__()
        
        self.feature = nn.ModuleList()
        
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(
                    nn.Conv2d(3, stage_out_channel[i], kernel_size=3, stride=2, padding=1, bias=False)
                )
            elif stage_out_channel[i-1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], stride=2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], stride=1))

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = qnn.QuantLinear(1024, num_classes, weight_quant=None, bias=True)

    def forward(self, x):
        for block in self.feature:
            x = block(x)

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
