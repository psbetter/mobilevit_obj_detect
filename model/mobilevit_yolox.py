import argparse

import torch
from torch import nn

from model.mobilevitv1 import mobilevit_v1
from model.yolox_common import BaseConv, CSPLayer, DWConv, YOLOXHead
from utils.utils import get_config


class MobileVit_YoloX(nn.Module):
    def __init__(self, cfg_data, cfg_train, phi="s", act="silu"):
        super(MobileVit_YoloX, self).__init__()
        if cfg_train.backbone == "mobilevitv1":
            # 80,80,96；40,40,128；20,20,640
            self.backbone = mobilevit_v1(pretrained=cfg_train.pretrained, weight_path=cfg_train.weights_path)
            in_filters = [96, 128, 640]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilevitv1.'.format(cfg_train.backbone))

        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        in_channels = [256, 512, 1024]

        self.lateral_conv0 = BaseConv(int(in_filters[2]), int(in_channels[1] * width), 1, 1, act=act)

        self.C3_p4 = CSPLayer(
            int(in_filters[1] + in_channels[1]*width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)

        self.C3_p3 = CSPLayer(
            int(in_filters[0]+in_channels[0]*width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)

        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)

        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.head = YOLOXHead(len(cfg_data.names), width, depthwise=depthwise)

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)

        # -------------------------------------------#
        #   20, 20, 640 -> 20, 20, 256
        # -------------------------------------------#
        P5 = self.lateral_conv0(feat3)
        # -------------------------------------------#
        #  20, 20, 256 -> 40, 40, 256
        # -------------------------------------------#
        P5_upsample = self.upsample(P5)
        # -------------------------------------------#
        #  40, 40, 256 + 40, 40, 128 -> 40, 40, 384
        # -------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        # -------------------------------------------#
        #   40, 40, 384 -> 40, 40, 256
        # -------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 128
        # -------------------------------------------#
        P4 = self.reduce_conv1(P5_upsample)
        # -------------------------------------------#
        #   40, 40, 128 -> 80, 80, 128
        # -------------------------------------------#
        P4_upsample = self.upsample(P4)
        # -------------------------------------------#
        #   80, 80, 128 + 80, 80, 96 -> 80, 80, 224
        # -------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        # -------------------------------------------#
        #   80, 80, 224 -> 80, 80, 128
        # -------------------------------------------#
        P3_out = self.C3_p3(P4_upsample)

        # -------------------------------------------#
        #   80, 80, 128 -> 40, 40, 128
        # -------------------------------------------#
        P3_downsample = self.bu_conv2(P3_out)
        # -------------------------------------------#
        #   40, 40, 128 + 40, 40, 128 -> 40, 40, 256
        # -------------------------------------------#
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        # -------------------------------------------#
        #   40, 40, 256 -> 40, 40, 256
        # -------------------------------------------#
        P4_out = self.C3_n3(P3_downsample)

        # -------------------------------------------#
        #   40, 40, 256 -> 20, 20, 256
        # -------------------------------------------#
        P4_downsample = self.bu_conv1(P4_out)
        # -------------------------------------------#
        #   20, 20, 256 + 20, 20, 256 -> 20, 20, 512
        # -------------------------------------------#
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        # -------------------------------------------#
        #   20, 20, 512 -> 20, 20, 512
        # -------------------------------------------#
        P5_out = self.C3_n4(P4_downsample)

        out0, out1, out2 = self.head([P3_out, P4_out, P5_out])

        return out0, out1, out2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification of CMT model")
    parser.add_argument('--train', default='../config/train.yaml', type=str, help='config of train process')
    parser.add_argument('--datasets', default='../config/coco.yaml', type=str, help='config of datasets')
    args = parser.parse_args()

    cfg_train = get_config(args.train)
    cfg_data = get_config(args.datasets)

    cfg_train.weights_path = "../config/model_best.pth.tar"

    img = torch.randn(1, 3, 640, 640)

    model = MobileVit_YoloX(cfg_data, cfg_train)
    out = model(img)
    print(out)
