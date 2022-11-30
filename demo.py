import argparse
import time
import cv2
import numpy as np
from PIL import Image
from model.inference_net import inference_net
from utils.utils import get_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification of CMT model")
    parser.add_argument('--config', default='config/config.yaml', type=str, help='config of train process')
    parser.add_argument('--classes_path', default='config/class_names.txt', type=str, help='file of class names')
    parser.add_argument('--anchors_path', default='config/anchors.txt', type=str, help='config of anchor')
    parser.add_argument('--weights_path', default='config/ep030-loss0.136-val_loss0.133.pth', type=str, help='path of weights')
    parser.add_argument('--cuda', default=True, type=bool, help='using cuda or not')
    parser.add_argument('--mode', default="fps", type=str, help='detect video or image')
    parser.add_argument("--crop", default=False, type=bool, help="get crop result or not")
    parser.add_argument('--count', default=False, type=bool, help='count objects')
    parser.add_argument('--video_path', default="0", type=str, help='detect file or camera')
    parser.add_argument('--video_save_path', default='', type=str, help='result save path')
    parser.add_argument('--video_fps', default=25.0, type=float, help='fps of video')
    parser.add_argument('--test_interval', default=100, type=int, help='test interval')
    parser.add_argument('--fps_image_path', default="E:/PyCharmWorkSpace/datasets/mask/test/649.jpg", type=str, help='fps image path')
    parser.add_argument('--dir_origin_path', default="", type=str, help='dir of images')
    parser.add_argument('--dir_save_path', default="", type=str, help='save path of dir detect')
    parser.add_argument('--letterbox_image', default=False, type=bool, help='letterbox image')
    parser.add_argument('--confidence', default=0.5, type=float, help='confidence thresh')
    parser.add_argument('--nms_iou', default=0.3, type=float, help='iou thresh')
    args = parser.parse_args()

    cfg_train = get_config(args.config)

    cfg_train.pretrained = False

    model = inference_net(args, cfg_train)

    if args.mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = model.detect_image(image, crop=args.crop, count=args.count)
                r_image.show()

    elif args.mode == "video":
        if args.video_path.isdigit():
            args.video_path = int(args.video_path)
        capture = cv2.VideoCapture(args.video_path)
        if args.video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(args.video_save_path, fourcc, args.video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(model.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if args.video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if args.video_save_path != "":
            print("Save processed video to the path :" + args.video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif args.mode == "fps":
        img = Image.open(args.fps_image_path)
        tact_time = model.get_FPS(img, args.test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif args.mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(args.dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(args.dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = model.detect_image(image)
                if not os.path.exists(args.dir_save_path):
                    os.makedirs(args.dir_save_path)
                r_image.save(os.path.join(args.dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict'.")
