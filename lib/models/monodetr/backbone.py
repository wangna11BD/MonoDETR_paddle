# import paddle
"""
Backbone modules.
"""
from collections import OrderedDict
# import torchvision
# from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from utils.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


import math
from numbers import Integral

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Uniform
from paddle.regularizer import L2Decay

from paddle3d.models import layers
from paddle3d.models.layers import reset_parameters
from paddle3d.utils import checkpoint
from paddle3d.models.layers import FrozenBatchNorm2d
from paddle3d.utils import checkpoint

class FrozenBatchNorm2d(paddle.nn.Layer):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from  with added eps before rqsrt,
    without which any other models than .models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-05):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', paddle.ones(shape=[n]))
        self.register_buffer('bias', paddle.zeros(shape=[n]))
        self.register_buffer('running_mean', paddle.zeros(shape=[n]))
        self.register_buffer('running_var', paddle.ones(shape=[n]))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict,
            prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs)

    def forward(self, x):
        w = self.weight.reshape([1, -1, 1, 1])
        b = self.bias.reshape([1, -1, 1, 1])
        rv = self.running_var.reshape([1, -1, 1, 1])
        rm = self.running_mean.reshape([1, -1, 1, 1])
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(paddle.nn.Layer):

    def __init__(self, backbone: paddle.nn.Layer, train_backbone: bool,
        return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (not train_backbone or 'layer2' not in name and 'layer3' not in
                name and 'layer4' not in name):
                parameter.stop_gradient=True
        if return_interm_layers:
            return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': '0'}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=
            return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = paddle.zeros(shape=[x.shape[0], x.shape[2], x.shape[3]]).cast('bool')
            out[name] = NestedTensor(x, m)
        return out


# class Backbone(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""

#     def __init__(self, name: str, train_backbone: bool,
#         return_interm_layers: bool, dilation: bool):
#         norm_layer = FrozenBatchNorm2d
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             pretrained=is_main_process(), norm_layer=norm_layer)
#         assert name not in ('resnet18', 'resnet34'
#             ), 'number of channels are hard coded'
#         super().__init__(backbone, train_backbone, return_interm_layers)
#         if dilation:
#             self.strides[-1] = self.strides[-1] // 2



class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 is_vd_mode=False,
                 norm_layer=None,
                 act=None,
                 data_format='NCHW'):
        super(ConvBNLayer, self).__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                "the kernel_size should be 3.")

        self.is_vd_mode = is_vd_mode
        self.act = act
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
            data_format=data_format)

        if norm_layer is None:
            self._batch_norm = nn.BatchNorm2D(out_channels, data_format=data_format)
        elif norm_layer=='frozenbn':
            self._batch_norm = FrozenBatchNorm2d(out_channels)
        else:
            raise ValueError("resnet not supported {}".format(norm_layer))
        if self.act:
            self._act = nn.ReLU()

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act:
            y = self._act(y)

        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 first_conv=False,
                 dilation=1,
                 is_vd_mode=False,
                 norm_layer=None,
                 data_format='NCHW'):
        super(BottleneckBlock, self).__init__()

        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            act='relu',
            data_format=data_format)

        if first_conv and dilation != 1:
            dilation //= 2

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            norm_layer=norm_layer,
            act='relu',
            dilation=dilation,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            norm_layer=norm_layer,
            act=None,
            data_format=data_format)

        if if_first or stride == 1:
            is_vd_mode = False

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=stride,
                is_vd_mode=is_vd_mode,
                norm_layer=norm_layer,
                data_format=data_format)

        self.shortcut = shortcut
        # NOTE: Use the wrap layer for quantization training
        self.relu = nn.ReLU()

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(short, conv2)
        y = self.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation=1,
                 shortcut=True,
                 if_first=False,
                 is_vd_mode=False,
                 norm_layer=None,
                 data_format='NCHW'):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            act='relu',
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            norm_layer=norm_layer,
            act=None,
            data_format=data_format)

        if if_first or stride == 1:
            is_vd_mode = False
        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=is_vd_mode,
                norm_layer=norm_layer,
                data_format=data_format)

        self.shortcut = shortcut
        self.dilation = dilation
        self.data_format = data_format
        self.relu = nn.ReLU()

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(short, conv1)
        y = self.relu(y)

        return y


class ResNet(nn.Layer):
    def __init__(self,
                 layers=50,
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 return_idx=[3],
                 pretrained=None,
                 variant='b',
                 norm_layer=None,
                 do_preprocess=True,
                 data_format='NCHW'):
        """
        Residual Network, see https://arxiv.org/abs/1512.03385

        Args:
            variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
            layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
            output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
            multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
            pretrained (str, optional): The path of pretrained model.
        """
        super(ResNet, self).__init__()
        self.variant = variant
        self.data_format = data_format
        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        self.do_preprocess = do_preprocess
        self.norm_mean = paddle.to_tensor([0.485, 0.456, 0.406])
        self.norm_std = paddle.to_tensor([0.229, 0.224, 0.225])
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.return_idx = return_idx

        if variant in ['c', 'd']:
            conv_defs = [
                [3, 32, 3, 2],
                [32, 32, 3, 1],
                [32, 64, 3, 1],
            ]
        else:
            conv_defs = [[3, 64, 7, 2]]
        self.conv1 = nn.Sequential()
        for (i, conv_def) in enumerate(conv_defs):
            c_in, c_out, k, s = conv_def
            self.conv1.add_sublayer(
                str(i),
                ConvBNLayer(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=k,
                    stride=s,
                    norm_layer=norm_layer,
                    act='relu',
                    data_format=data_format))
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=data_format)

        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    ###############################################################################

                    bottleneck_block = self.add_sublayer(
                        'layer_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0
                            and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            first_conv=i == 0,
                            is_vd_mode=variant in ['c', 'd'],
                            dilation=dilation_rate,
                            norm_layer=norm_layer,
                            data_format=data_format))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = dilation_dict[block] \
                        if dilation_dict and block in dilation_dict else 1
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    basic_block = self.add_sublayer(
                        'layer_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 \
                                and dilation_rate == 1 else 1,
                            dilation=dilation_rate,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            is_vd_mode=variant in ['c', 'd'],
                            norm_layer=norm_layer,
                            data_format=data_format))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, image):
        if self.do_preprocess:
            image = self.preprocess(image)
        y = self.conv1(image)
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for idx, stage in enumerate(self.stage_list):
            for block in stage:
                y = block(y)
            if idx in self.return_idx:
                feat_list.append(y)

        return feat_list

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images [paddle.Tensor(N, 3, H, W)]: Input images
        Return
            x [paddle.Tensor(N, 3, H, W)]: Preprocessed images
        """
        x = images
        # Create a mask for padded pixels
        mask = paddle.isnan(x)

        # Match ResNet pretrained preprocessing
        x = self.normalize(x, mean=self.norm_mean, std=self.norm_std)

        # Make padded pixels = 0
        a = paddle.zeros_like(x)
        x = paddle.where(mask, a, x)

        return x

    def normalize(self, image, mean, std):
        shape = paddle.shape(image)
        if mean.shape:
            mean = mean[..., :, None]
        if std.shape:
            std = std[..., :, None]
        out = (image.reshape([shape[0], shape[1], shape[2] * shape[3]]) -
               mean) / std
        return out.reshape(shape)

    def init_weight(self):
        if self.pretrained is not None:
            checkpoint.load_pretrained_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    reset_parameters(sublayer)


class FrozenResNet50(nn.Layer):
    def __init__(self, layers=50, 
                 output_stride=None,
                 multi_grid=(1, 1, 1),
                 return_idx=[3],
                 pretrained=None,
                 variant='b',
                 norm_layer=None,
                 do_preprocess=True,
                 data_format='NCHW'):
        super(FrozenResNet50, self).__init__()
        self.strides = [8, 16, 32]
        self.num_channels = [512, 1024, 2048]
        self.model = ResNet(layers=layers, 
                            output_stride=output_stride,
                            multi_grid=multi_grid,
                            return_idx=return_idx,
                            norm_layer=norm_layer,
                            do_preprocess=do_preprocess)
        for name, parameter in self.model.named_parameters():
            if (('layer_1' not in name) and ('layer_2' not in name) and ('layer_3' not in name)):
                parameter.stop_gradient=True
    
    def forward(self, images):
        xs = self.model(images)
        out = []
        for x in xs:
            m = paddle.zeros(shape=[x.shape[0], x.shape[2], x.shape[3]]).cast('bool')
            out.append(NestedTensor(x, m))
        return out

class Joiner(paddle.nn.Sequential):

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images):
        xs = self[0](images)
        # out: List[NestedTensor] = []
        pos = []
        # for name, x in sorted(xs.items()):
        #     out.append(x)
        for x in xs:
            pos.append(self[1](x).cast(x.tensors.dtype))
        return xs, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    backbone = FrozenResNet50(norm_layer='frozenbn', do_preprocess=False, return_idx=[1,2,3])
    model = Joiner(backbone, position_embedding)
    return model
