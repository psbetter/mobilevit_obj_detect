import argparse
import datetime
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.mobilevit_yolov4 import MobileVit_YoloV4
from model.yolov4_common import YOLOV4Loss
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import yolo_dataset_collate, YoloDataset
from utils.utils import weights_init, get_lr_scheduler, set_optimizer_lr, get_lr, get_config, get_optimizer, ModelEMA, \
    get_classes, get_anchors

def train(model_train, yolo_loss, optimizer, epoch, epoch_step, train_loader, Epoch, cuda, scaler, ema=None, local_rank=0):
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

        optimizer.zero_grad()
        if scaler is None:
            outputs = model_train(images)
            # yolov4 loss
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

            loss_value.backward()
            optimizer.step()
        else:
            # mixed precision training
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item
                loss_value = loss_value_all
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    pbar.close()
    return loss


def eval(model_train, yolo_loss, optimizer, epoch, epoch_step_val, val_loader, Epoch, cuda, ema=None, local_rank=0):
    val_loss = 0
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    if ema:
        model_train = ema.ema
    else:
        model_train = model_train.eval()

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
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    pbar.close()

    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classification of CMT model")
    parser.add_argument('--config', default='config/config.yaml', type=str, help='config of train process')
    parser.add_argument('--classes_path', default='config/class_names.txt', type=str, help='file of class names')
    parser.add_argument('--anchors_path', default='config/anchors.txt', type=str, help='config of anchor')
    args = parser.parse_args()

    cfg_train = get_config(args.config)
    class_names, num_classes = get_classes(args.classes_path)
    anchors, num_anchors = get_anchors(args.anchors_path)

    model = MobileVit_YoloV4(num_classes, cfg_train)
    if not cfg_train.pretrained:
        weights_init(model)
    model_train = model.cuda()

    yolo_loss = YOLOV4Loss(args, cfg_train)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(cfg_train.save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=cfg_train.input_shape)

    # mixed precision training
    if cfg_train.mixed_precision:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

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
    with open(cfg_train.train, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(cfg_train.val, encoding='utf-8') as f:
        val_lines = f.readlines()
    train_dataset = YoloDataset(train_lines, cfg_train, num_classes, train=True, special_aug_ratio=cfg_train.special_aug_ratio)
    val_dataset = YoloDataset(val_lines, cfg_train, num_classes, train=False, special_aug_ratio=0)
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

    eval_callback = EvalCallback(model, cfg_train.input_shape, anchors, cfg_train.anchor_mask, class_names, num_classes,
                                 val_lines, log_dir, cfg_train.cuda, eval_flag=cfg_train.eval_flag, period=cfg_train.eval_period)

    if cfg_train.ema:
        ema = ModelEMA(model_train)
        ema.updates = epoch_step * cfg_train.init_epoch
    else:
        ema = None

    for epoch in range(cfg_train.init_epoch, cfg_train.unfreeze_epoch):
        if epoch >= cfg_train.freeze_epoch and not UnFreeze_flag and cfg_train.freeze_train:
            batch_size = cfg_train.unfreeze_batch_size

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

            if ema:
                ema.updates = epoch_step * epoch

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = train(model_train, yolo_loss, optimizer, epoch, epoch_step, train_loader, cfg_train.unfreeze_epoch,
                     cfg_train.cuda, scaler, ema)

        val_loss = eval(model_train, yolo_loss, optimizer, epoch, epoch_step_val, val_loader, cfg_train.unfreeze_epoch,
                        cfg_train.cuda, ema)

        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train.eval())
        print('[INFO] Epoch:' + str(epoch + 1) + '/' + str(cfg_train.unfreeze_epoch))
        print('[INFO] Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        # save checkpoints
        if (epoch + 1) % cfg_train.save_period == 0 or epoch + 1 == cfg_train.unfreeze_epoch:
            torch.save(save_state_dict, os.path.join(cfg_train.save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            torch.save(save_state_dict, os.path.join(cfg_train.save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(cfg_train.save_dir, "last_epoch_weights.pth"))
