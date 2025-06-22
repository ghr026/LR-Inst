# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .neck_moudle import resize
from .neck_moudle import CAB, SAB, FASMLayer
from .neck_moudle import SSFM, DSSFM

@MODELS.register_module()
class CSPNeXtPAFPN(BaseModule):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        planes=48,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        align_corners: bool = False,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.compression_1 = ConvModule(
            planes * 4,
            planes * 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_3 = ConvModule(
            planes * 8,
            planes * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.fusion1 = SSFM(planes * 2)
        self.fusion2 = SSFM(planes * 8)

        self.fusion5 = DSSFM(planes * 2)
        self.fusion6 = DSSFM(planes * 2)
        self.down_1 = DepthwiseSeparableConvModule(
            planes * 2,
            planes * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.down_2 = DepthwiseSeparableConvModule(
            planes * 4,
            planes * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.down_4 = DepthwiseSeparableConvModule(
            planes * 2,
            planes * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.down_3 = nn.Sequential(
            DepthwiseSeparableConvModule(
            planes * 2,
            planes * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                planes * 2,
                planes * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        self.sab_2 = SAB()
        self.cab_2 = CAB(planes * 4)
        self.layer1 = nn.Sequential(
            FASMLayer(planes * 2, planes * 2, n=1, stride=1, kernel_sizes=[1,3,5]),
            FASMLayer(planes * 2, planes * 2, n=1, stride=1, kernel_sizes=[1, 3, 5]),
        )
        self.layer2 = nn.Sequential(
            FASMLayer(planes * 4, planes * 2, n=1, stride=1, kernel_sizes=[1,3,5]),
        )
        self.layer3 = nn.Sequential(
            FASMLayer(planes * 4, planes * 2, n=1, stride=1, kernel_sizes=[1,3,5]),
            FASMLayer(planes * 2, planes * 2, n=1, stride=1, kernel_sizes=[1, 3, 5]),
        )



        self.dec_layers = ConvModule(
            planes * 12,
            planes * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.dec_layers2 = ConvModule(
            planes * 8,
            planes * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.dec_layers3 = ConvModule(
            planes * 8,
            planes * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)
        out_size = (math.ceil(inputs[0].shape[-2]), math.ceil(inputs[0].shape[-1]))
        out_size_1 = (math.ceil(inputs[0].shape[-2] / 2), math.ceil(inputs[0].shape[-1] / 2))
        x_p = inputs[0]
        x_i = inputs[1]
        x_d = inputs[-1]


        comp_i = resize(
            x_i,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        comp_i = self.compression_1(self.relu(comp_i))

        x_p = self.fusion1(x_p, comp_i)

        comp_i2 = self.down_2(x_i)
        x_d = self.fusion2(x_d, comp_i2)
        comp_p = self.down_1(self.relu(x_p))


        x_d1 = resize(
            x_d,
            size=out_size_1,
            mode='bilinear',
            align_corners=self.align_corners)
        x_d1 = self.dec_layers2(x_d1)
        x_i = torch.cat([comp_p, x_i, x_d1],dim=1)
        x_i = self.dec_layers(x_i)


        x_i = self.sab_2(self.cab_2(x_i) * x_i) * (self.cab_2(x_i) * x_i)

        x_i = self.layer2[0](self.relu(x_i))

        comp_i = resize(
            x_i,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        x_p = self.layer1[0](self.relu(x_p))

        x_d = self.layer3[0](self.relu(self.dec_layers3(x_d)))
        comp_d = resize(
            x_d,
            size=out_size,
            mode='bilinear',
            align_corners=self.align_corners)
        comp_p = self.down_3(x_p)
        x_p = self.fusion5(x_p, comp_i, comp_d)

        x_p = self.layer1[1](x_p)

        x_d = self.fusion6(x_d, self.down_4(self.relu(x_i)), comp_p)
        x_d = self.layer3[1](x_d)

        outs = [x_p, x_i, x_d]

        return tuple(outs)
