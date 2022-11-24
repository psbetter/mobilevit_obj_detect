# MobileVit for object detection

## Overview

This is a PyTorch implementation of [MobileVit](https://arxiv.org/abs/2110.02178) for object detection. 

Idea of model construction: We take MobileVit as the backbone network, and add PAFPN feature enhancement structure to it, 
then add Yolov4 or YoloX decoding header. 

More implementation details will be released soon.

## Usage
1. envs
```bash=
pip install -r requirements.txt
```

2. Training
```bash=
python train.py
```

3. Testing
```bash=
python demo.py
```

The pretrained weights of MobileVit is download from [MobileViT](https://github.com/wilile26811249/MobileViT)

## Reference
https://github.com/wilile26811249/MobileViT

https://github.com/bubbliiiing/mobilenet-yolov4-pytorch

https://github.com/bubbliiiing/yolox-pytorch