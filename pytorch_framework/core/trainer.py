CUDA_LAUNCH_BLOCKING = 1

import numpy as np

import torch
import os
from helpers.utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

from ray import tune
from losses.Yolov1 import YoloLoss
from helpers.logger import Logger
from helpers.viewer import Viewer
from core.train_config import TrainConfig


class Trainer:

    def __init__(self, config: TrainConfig):
        self.config = config
        self.losses = YoloLoss()

    def to_gpu(self, item):
        return item.type(torch.cuda.FloatTensor).to(self.device)

    def loss_calc(self, target, model):
        gpu_img = self.to_gpu(target['image'])
        gpu_mask = self.to_gpu(target['mask'])

        prediction = model(gpu_img)

        losses = self.losses.calc(prediction, gpu_mask)
        for key in losses.keys():
            self.config.logger.add_scalar('Losses_{}/{}'.format('train', key), losses[key].item())

        return sum(losses.values())

    def train(self, epoch):
        c = self.config
        c.model.train()
        if epoch > 0 and c.scheduler is not None:
            c.scheduler.step()

        loss_accum = 0
        step_count = len(c.loaders['train'])
        for i_step, target in enumerate(c.loaders['train']):
            loss_value = self.loss_calc(target, c.model)

            loss_value.backward()
            c.optimizer.step()
            c.optimizer.zero_grad()

            loss_accum += loss_value.item()
            c.logger.add_scalar('Loss_sum_{}/batch'.format('train'), loss_value.item())

            ave_loss = loss_accum / step_count
            c.logger.add_scalar('Loss_sum_train/epoch', ave_loss)
