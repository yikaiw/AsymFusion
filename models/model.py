"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from utils.helpers import maybe_download
from .parallel import ModuleParallel, BatchNorm2dParallel

data_info = {
    21: 'VOC',
}

models_urls = {
    '101_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/Owmttk9bdPROwc6/download',

    '50_imagenet' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_stages, num_parallel):
        super(CRPBlock, self).__init__()
        for i in range(num_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes, out_planes))
        self.stride = 1
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=5, stride=1, padding=2))

    def forward(self, x):
        top = x
        for i in range(self.num_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = [x[l] + top[l] for l in range(self.num_parallel)]
        return x


stages_suffixes = {0 : '_conv', 1 : '_conv_relu_varout_dimred'}

class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, num_stages, num_parallel):
        super(RCUBlock, self).__init__()
        for i in range(num_blocks):
            for j in range(num_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, bias=(j == 0)))
        self.stride = 1
        self.num_blocks = num_blocks
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
    
    def forward(self, x):
        for i in range(self.num_blocks):
            residual = x
            for j in range(self.num_stages):
                x = self.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x = [x[l] + residual[l] for l in range(self.num_parallel)]
        return x



class ShiftShuffle(nn.Module):
    def __init__(self, reverse=False):
        super(ShiftShuffle, self).__init__()
        self.pos = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        if reverse:
            self.pos = self.pos[::-1]

    def forward(self, x):
        if len(x) != 2:
            return x, [0, 0]
        shift_group = x[0].shape[1] // 5
        shuffle_channel = shift_group * 4
        x1a, x1b = x[0][:, :shuffle_channel], x[1][:, shuffle_channel:]
        x2a, x2b = x[1][:,:shuffle_channel], x[0][:, shuffle_channel:]
        shuffle = [torch.cat([x1a, x1b], dim=1), torch.cat([x2a, x2b], dim=1)]

        h, w = x1a.shape[-2:]
        pad = (1, 1, 1, 1)
        x1a_, x2a_, x1b_, x2b_ = [], [], [], []
        for idx, i in enumerate(range(0, shuffle_channel, shift_group)):
            posh, posw = self.pos[idx][0] + 1, self.pos[idx][1] + 1
            x1a_.append(F.pad(x1a[:, i: i + shift_group], pad)[:, :, posh: h + posh, posw: w + posw])
            x2a_.append(F.pad(x2a[:, i: i + shift_group], pad)[:, :, posh: h + posh, posw: w + posw])
        x1a_.append(torch.zeros_like(x1b))
        x2a_.append(torch.zeros_like(x2b))
        shift = [torch.cat(x1a_, dim=1), torch.cat(x2a_, dim=1)]

        return shuffle, shift


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, stride=1, downsample=None, use_shuffle=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride
        self.use_shuffle = use_shuffle
        if self.use_shuffle:
            self.shiftshuffle1 = ShiftShuffle(reverse=False)
            self.shiftshuffle2 = ShiftShuffle(reverse=True)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        if self.use_shuffle and len(x) > 1:
            out, shift = self.shiftshuffle1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.use_shuffle and len(x) > 1:
            out[0] = out[0] + shift[1]
            out[1] = out[1] + shift[0]
            out, _ = self.shiftshuffle2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, block, layers, num_parallel, num_classes=21):
        self.inplanes = 64
        self.num_parallel = num_parallel
        super(RefineNet, self).__init__()
        self.dropout = ModuleParallel(nn.Dropout(p=0.5))
        self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False))
        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256)
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.clf_conv = conv3x3(256, num_classes, bias=True)
        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.alpha = nn.Parameter(torch.ones([1, num_parallel, 157, 157], requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def _make_crp(self, in_planes, out_planes, num_stages):
        layers = [CRPBlock(in_planes, out_planes, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, num_blocks, num_stages):
        layers = [RCUBlock(in_planes, out_planes, num_blocks, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            use_shuffle = True if i >= num_blocks - 1 else False
            layers.append(block(self.inplanes, planes, self.num_parallel, use_shuffle=use_shuffle))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout(l4)
        l3 = self.dropout(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = [nn.Upsample(size=l3[0].size()[2:], mode='bilinear', align_corners=True)(x4_) for x4_ in x4]

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = [x3[l] + x4[l] for l in range(self.num_parallel)]
        x3 = self.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = [nn.Upsample(size=l2[0].size()[2:], mode='bilinear', align_corners=True)(x3_) for x3_ in x3]

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = [x2[l] + x3[l] for l in range(self.num_parallel)]
        x2 = self.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = [nn.Upsample(size=l1[0].size()[2:], mode='bilinear', align_corners=True)(x2_) for x2_ in x2]

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = [x1[l] + x2[l] for l in range(self.num_parallel)]
        x1 = self.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.dropout(x1)

        out = self.clf_conv(x1)
        ens = 0
        alpha_soft = F.softmax(self.alpha)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * out[l].detach()
        out.append(ens)
        return out, alpha_soft


def refinenet(num_layers, num_classes, num_parallel, **kwargs):
    if int(num_layers) == 50:
        layers = [3, 4, 6, 3]
    elif int(num_layers) == 101:
        layers = [3, 4, 23, 3]
    elif int(num_layers) == 152:
        layers = [3, 8, 36, 3]
    else:
        print('invalid num_layers')

    model = RefineNet(Bottleneck, layers, num_parallel, num_classes=num_classes, **kwargs)
    return model


def model_init(model, num_layers, num_parallel, imagenet=False, pretrained=True):
    if imagenet:
        key = str(num_layers) + '_imagenet'
        url = models_urls[key]
        state_dict = maybe_download(key, url)
        model_dict = expand_model_dict(model.state_dict(), state_dict, num_parallel)
        model.load_state_dict(model_dict, strict=True)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = str(num_layers) + '_' + dataset.lower()
            key = 'rf' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict
