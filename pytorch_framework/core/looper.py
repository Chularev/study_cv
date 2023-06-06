

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
class Looper:

    def __init__(self, datasets):
        self.datasets = datasets
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = Logger('TensorBoard')
        self.losses = YoloLoss()

        '''
        self.metrics = {
            'train': MyMetric(self.device),
            'val': MyMetric(self.device)
        }
        '''

        self.phase = 'train'
        self.viewer = Viewer()
        self.out = ''

    def to_gpu(self, item):
        return item.type(torch.cuda.FloatTensor).to(self.device)



    def start(self, model, loaders, optimizer, num_epochs, scheduler=None):

        torch.cuda.empty_cache()

        model = model.to(self.device)

        '''
        report_metrics = {
            'loss': {
                'train': [],
                'val': []
            }
        }
        '''

        for epoch in range(num_epochs):
            for phase in ['train']:  # , 'val']:
                model.train(phase == 'train')  # Set model to training mode

                loss_accum = 0
                step_count = len(loaders[phase])
                for i_step, target in enumerate(loaders[phase]):
                    loss_value = self.loss_calc(target, model)

                    loss_accum += loss_value.item()
                    self.logger.add_scalar('Loss_sum_{}/batch'.format(phase), loss_value.item())
                    # report_metrics['loss'][phase].append(loss_value.item())
                    print('Epoch {}/{}. Phase {} Step {}/{} Loss {}'.format(epoch, num_epochs - 1, phase,
                                                                            i_step, step_count,
                                                                            loss_value.item()) + self.out)

                model.train(False)
                torch.set_grad_enabled(False)
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)

                    for index in range(100, 120):
                        target = self.datasets[phase][index]
                        predict = model(self.to_gpu(target['image'].unsqueeze(0)))
                        predict = predict[0].to('cpu')

                        img_grid = self.viewer.create_output(target, predict)
                        self.logger.add_grid_images('Output ' + str(index), img_grid)

                ave_loss = loss_accum / step_count
                self.logger.add_scalar('Loss_sum_train/epoch', ave_loss)

    # train_metrics = self.metrics['train'].compute()
    #  val_metrics = self.metrics['val'].compute()


'''
        tune.report(
            train_loss=sum(report_metrics['loss']['train']) / len(report_metrics['loss']['train']),
            val_loss=sum(report_metrics['loss']['val']) / len(report_metrics['loss']['val']),

       #     train_iou=train_metrics['iou'].item(),
       #     val_iou=val_metrics['iou'].item(),

        #    train_accuracy=train_metrics['accuracy'].item(),
        #    val_accuracy=val_metrics['accuracy'].item()
        )
'''