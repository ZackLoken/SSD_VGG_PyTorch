import math
import sys
import time
import contextlib
import torch
import torchvision.models.detection.mask_rcnn
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq, val_data_loader=None, profiler=None):
    model.train()
    train_metric_logger = utils.MetricLogger(delimiter="  ")
    train_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_header = 'Epoch: [{}] Training'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # Use profiler if provided, otherwise use nullcontext
    profiler_context = profiler if profiler else contextlib.nullcontext()

    scaler = torch.amp.GradScaler()

    with profiler_context:
        for images, targets in train_metric_logger.log_every(train_data_loader, print_freq, train_header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Use autocast for mixed precision
            with torch.amp.autocast(device_type="cuda"):
                train_loss_dict = model(images, targets)
                train_losses = sum(loss for loss in train_loss_dict.values())

            # Backward pass
            scaler.scale(train_losses).backward()

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Reduce losses over all GPUs for logging purposes
            train_loss_dict_reduced = utils.reduce_dict(train_loss_dict)
            train_losses_reduced = sum(loss for loss in train_loss_dict_reduced.values())
            train_loss_value = train_losses_reduced.item()

            if not math.isfinite(train_loss_value):
                print("Loss is {}, stopping training".format(train_loss_value))
                print(train_loss_dict_reduced)
                sys.exit(1)

            if lr_scheduler is not None:
                lr_scheduler.step()

            train_metric_logger.update(loss=train_losses_reduced, **train_loss_dict_reduced)
            train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # validation loop
        if val_data_loader is not None:
            val_metric_logger = utils.MetricLogger(delimiter="  ")
            val_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            val_header = 'Epoch: [{}] Validation'.format(epoch)

            with torch.no_grad():
                for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 2, val_header):
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.amp.autocast(device_type="cuda"):
                        val_loss_dict = model(images, targets)
                        val_loss_dict_reduced = utils.reduce_dict(val_loss_dict)
                        val_losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())
                        val_loss_value = val_losses_reduced.item()

                    if not math.isfinite(val_loss_value):
                        print("Loss is {}, stopping validation".format(val_loss_value))
                        print(val_loss_dict_reduced)
                        sys.exit(1)

                    val_metric_logger.update(loss=val_losses_reduced, **val_loss_dict_reduced)
                    val_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return train_metric_logger, val_metric_logger if val_data_loader is not None else train_metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, val_data_loader, device, train_data_loader=None, profiler=None):
    """
    Evaluate model on validation dataset and return validation accuracy to metric logger. If training dataset is provided,
    also return training accuracy to metric logger.
    :param model: torch.nn.Module
    :param val_data_loader: torch.utils.data.DataLoader
    :param device: torch.device
    :param train_data_loader: torch.utils.data.DataLoader (default=None)
    :return: train_coco_evaluator: CocoEvaluator, 
             val_coco_evaluator: CocoEvaluator (if train_data_loader is not None else val_coco_evaluator)
    """
    
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    iou_types = _get_iou_types(model)

    val_metric_logger = utils.MetricLogger(delimiter="  ")
    val_coco = get_coco_api_from_dataset(val_data_loader.dataset)
    val_coco_evaluator = CocoEvaluator(val_coco, iou_types)

    profiler_context = profiler if profiler else contextlib.nullcontext()

    with profiler_context:    
        if train_data_loader is not None:
            train_metric_logger = utils.MetricLogger(delimiter="  ")
            train_coco = get_coco_api_from_dataset(train_data_loader.dataset)
            train_coco_evaluator = CocoEvaluator(train_coco, iou_types)

            # Calculate training accuracy
            for images, targets in train_metric_logger.log_every(train_data_loader, len(train_data_loader) // 3, 'Training Accuracy: '):
                images = list(img.to(device) for img in images)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_time = time.time()
                with torch.amp.autocast():
                    outputs = model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time

                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                evaluator_time = time.time()
                train_coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                train_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            # Gather the stats from all processes
            train_metric_logger.synchronize_between_processes()
            print("Training Performance:", train_metric_logger)
            train_coco_evaluator.synchronize_between_processes()

            # Accumulate predictions from all images
            train_coco_evaluator.accumulate()
            train_coco_evaluator.summarize()

        # Calculate validation accuracy
        for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 2, 'Validation Accuracy: '):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.amp.autocast():
                outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            val_coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            val_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # Gather the stats from all processes
        val_metric_logger.synchronize_between_processes()
        print("Validation Performance:", val_metric_logger)
        val_coco_evaluator.synchronize_between_processes()
        val_coco_evaluator.accumulate()
        val_coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    return train_coco_evaluator, val_coco_evaluator if train_data_loader is not None else val_coco_evaluator