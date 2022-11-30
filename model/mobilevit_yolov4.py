import torch
from torch import nn

from model.mobilevitv1 import mobilevit_v1, mobilevit_xxs
from model.yolov4_common import make_three_conv, SpatialPyramidPooling, Upsample, conv2d, yolo_head, conv_dw, make_five_conv


class MobileVit_YoloV4(nn.Module):
    def __init__(self, num_classes, cfg_train):
        super(MobileVit_YoloV4, self).__init__()
        if cfg_train.backbone == "mobilevitv1":
            # 80,80,96；40,40,128；20,20,640
            self.backbone = mobilevit_v1(pretrained=cfg_train.pretrained, weight_path=cfg_train.weights_path)
            in_filters = [96, 128, 640]
        elif cfg_train.backbone == "mobilevit_xxs":
            # 80,80,96；40,40,128；20,20,640
            self.backbone = mobilevit_xxs(pretrained=cfg_train.pretrained, weight_path=cfg_train.weights_path)
            in_filters = [48, 64, 320]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilevitv1.'.format(cfg_train.backbone))

        self.conv1 = make_three_conv([512, 1024], in_filters[2])
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(in_filters[1], 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.num_classes = num_classes
        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3 = yolo_head([256, len(cfg_train.anchors_mask[0]) * (5 + self.num_classes)], 128)

        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2 = yolo_head([512, len(cfg_train.anchors_mask[1]) * (5 + self.num_classes)], 256)

        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1 = yolo_head([1024, len(cfg_train.anchors_mask[2]) * (5 + self.num_classes)], 512)

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # 16,16,640 -> 16,16,512 -> 16,16,1024 -> 16,16,512 -> 16,16,2048
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # 16,16,2048 -> 16,16,512 -> 16,16,1024 -> 16,16,512
        P5 = self.conv2(P5)

        # 16,16,512 -> 16,16,256 -> 32,32,256
        P5_upsample = self.upsample1(P5)
        # 32,32,512 -> 32,32,256
        P4 = self.conv_for_P4(x1)
        # 32,32,256 + 32,32,256 -> 32,32,512
        P4 = torch.cat([P4, P5_upsample], axis=1)
        # 32,32,512 -> 32,32,256 -> 32,32,512 -> 32,32,256 -> 32,32,512 -> 32,32,256
        P4 = self.make_five_conv1(P4)

        # 32,32,256 -> 32,32,128 -> 64,64,128
        P4_upsample = self.upsample2(P4)
        # 64,64,256 -> 64,64,128
        P3 = self.conv_for_P3(x2)
        # 64,64,128 + 64,64,128 -> 64,64,256
        P3 = torch.cat([P3, P4_upsample], axis=1)
        # 64,64,256 -> 64,64,128 -> 64,64,256 -> 64,64,128 -> 64,64,256 -> 64,64,128
        P3 = self.make_five_conv2(P3)

        # 64,64,128 -> 32,32,256
        P3_downsample = self.down_sample1(P3)
        # 32,32,256 + 32,32,256 -> 32,32,512
        P4 = torch.cat([P3_downsample, P4], axis=1)
        # 32,32,512 -> 32,32,256 -> 32,32,512 -> 32,32,256 -> 32,32,512 -> 32,32,256
        P4 = self.make_five_conv3(P4)

        # 32,32,256 -> 16,16,512
        P4_downsample = self.down_sample2(P4)
        # 16,16,512 + 16,16,512 -> 16,16,1024
        P5 = torch.cat([P4_downsample, P5], axis=1)
        # 16,16,1024 -> 16,16,512 -> 16,16,1024 -> 16,16,512 -> 16,16,1024 -> 16,16,512
        P5 = self.make_five_conv4(P5)

        # ---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,64,64)
        # ---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,32,32)
        # ---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,16,16)
        # ---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
