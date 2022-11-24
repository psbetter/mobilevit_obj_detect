import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.mobilevit_yolov4 import MobileVit_YoloV4
from model.mobilevit_yolox import MobileVit_YoloX
from model.yolov4_common import YOLOV4Loss
from model.yolox_common import YOLOXLoss
from utils.dataloader import yolo_dataset_collate, YoloDataset
from utils.utils import weights_init, get_lr_scheduler, set_optimizer_lr, get_lr, get_config, get_optimizer


def train(model_train, yolo_loss, optimizer, epoch, epoch_step, train_loader, Epoch, cuda, local_rank=0):
    loss = 0
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(train_loader):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()

        outputs = model_train(images)
        # yolov4 loss
        # loss_value_all = 0
        # for l in range(len(outputs)):
        #     loss_item = yolo_loss(l, outputs[l], targets)
        #     loss_value_all += loss_item
        # loss_value = loss_value_all
        # yolox loss
        loss_value = yolo_loss(outputs, targets)

        # ----------------------#
        #   反向传播
        # ----------------------#
        loss_value.backward()
        optimizer.step()

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    pbar.close()
    return loss


def eval(model_train, yolo_loss, optimizer, epoch, epoch_step_val, val_loader, Epoch, cuda, local_rank=0):
    val_loss = 0
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(val_loader):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

            optimizer.zero_grad()

            outputs = model_train(images)

            # yolov4 loss
            # loss_value_all = 0
            # for l in range(len(outputs)):
            #     loss_item = yolo_loss(l, outputs[l], targets)
            #     loss_value_all += loss_item
            # loss_value = loss_value_all

            # yolox loss
            loss_value = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    pbar.close()

    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification of CMT model")
    parser.add_argument('--train', default='config/train.yaml', type=str, help='config of train process')
    parser.add_argument('--datasets', default='config/coco.yaml', type=str, help='config of datasets')
    args = parser.parse_args()

    cfg_train = get_config(args.train)
    cfg_data = get_config(args.datasets)

    # model = MobileVit_YoloV4(cfg_data, cfg_train)
    model = MobileVit_YoloX(cfg_data, cfg_train)
    if not cfg_train.pretrained:
        weights_init(model)
    model_train = model.cuda()

    # yolo_loss = YOLOV4Loss(cfg_data, cfg_train)
    # yolox loss
    yolo_loss = YOLOXLoss(cfg_data, cfg_train)

    UnFreeze_flag = False
    if cfg_train.freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = cfg_train.freeze_batch_size if cfg_train.freeze_train else cfg_train.unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if cfg_train.optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-4 if cfg_train.optimizer_type in ['adam', 'adamw'] else 5e-4
    init_lr_fit = min(max(batch_size / nbs * cfg_train.init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * cfg_train.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # select optimizer
    optimizer = get_optimizer(model, init_lr_fit, cfg_train)
    lr_scheduler_func = get_lr_scheduler(cfg_train.lr_decay_type, init_lr_fit, min_lr_fit, cfg_train.unfreeze_epoch)

    # load data
    with open(cfg_data.train, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(cfg_data.val, encoding='utf-8') as f:
        val_lines = f.readlines()
    train_dataset = YoloDataset(train_lines, cfg_train, cfg_data, train=True, special_aug_ratio=cfg_train.special_aug_ratio)
    val_dataset = YoloDataset(val_lines, cfg_train, cfg_data, train=False, special_aug_ratio=0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=cfg_train.num_workers,
                              pin_memory=True,
                              drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=cfg_train.num_workers,
                            pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
    num_train = len(train_lines)
    num_val = len(val_lines)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    writer = SummaryWriter('./logs/mobilevit_yolov4')

    for epoch in range(cfg_train.init_epoch, cfg_train.unfreeze_epoch):
        if epoch >= cfg_train.freeze_epoch and not UnFreeze_flag and cfg_train.freeze_train:
            batch_size = cfg_train.unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 64
            lr_limit_max = 1e-3 if cfg_train.optimizer_type in ['adam', 'adamw'] else 5e-2
            lr_limit_min = 3e-4 if cfg_train.optimizer_type in ['adam', 'adamw'] else 5e-4
            init_lr_fit = min(max(batch_size / nbs * cfg_train.init_lr, lr_limit_min), lr_limit_max)
            min_lr_fit = min(max(batch_size / nbs * cfg_train.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(cfg_train.lr_decay_type, init_lr_fit, min_lr_fit,
                                                 cfg_train.unfreeze_epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                                      num_workers=cfg_train.num_workers,
                                      pin_memory=True,
                                      drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)
            val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=cfg_train.num_workers,
                                    pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=None)

            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = train(model_train, yolo_loss, optimizer, epoch, epoch_step, train_loader, cfg_train.unfreeze_epoch,
                     cfg_train.cuda)

        val_loss = eval(model_train, yolo_loss, optimizer, epoch, epoch_step_val, val_loader, cfg_train.unfreeze_epoch,
                        cfg_train.cuda)

        writer.add_scalar('train_loss', loss / epoch_step, global_step=epoch)
        writer.add_scalar('eval_loss', val_loss / epoch_step_val, global_step=epoch)
        print('[INFO] Epoch:' + str(epoch + 1) + '/' + str(cfg_train.unfreeze_epoch))
        print('[INFO] Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # save checkpoints
        if (epoch + 1) % cfg_train.save_period == 0 or epoch + 1 == cfg_train.unfreeze_epoch:
            torch.save(model.state_dict(), os.path.join(cfg_train.save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        torch.save(model.state_dict(), os.path.join(cfg_train.save_dir, "last_epoch_weights.pth"))
