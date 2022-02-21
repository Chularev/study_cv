CUDA_LAUNCH_BLOCKING = 1

import numpy as np

import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
from metrics import MyMetric
from torchvision import transforms
from resource_monitor import ResourceMonitor
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from losses import MyLoss


class PyTorchHelper:

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = SummaryWriter('TensorBoard')
        self.losses = MyLoss()
        self.metrics = {
            'train': MyMetric(self.device),
            'val': MyMetric(self.device)
        }
        self.phase = 'train'

    def to_gpu(self, item):
        return item.type(torch.cuda.FloatTensor).to(self.device)

    def loss_calc(self, img, target, model):
        gpu_img = self.to_gpu(img)
        gpu_img_has_person = target['img_has_person'].type(torch.cuda.IntTensor).to(self.device)
        gpu_box = self.to_gpu(target['box'])

        prediction = model(gpu_img)

        with torch.no_grad():
            metrics = self.metrics[self.phase].step(prediction, gpu_img_has_person, gpu_box)
            print('iou {}, acc {}'.format(metrics['iou'], metrics['accuracy']))

        losses = self.losses.calc(prediction, gpu_img_has_person, gpu_box)
        return losses

    '''
    @torch.inference_mode()
    def evaluate(self, model, data_loader):
        return M.iou(model, data_loader)
    '''

    def train_model(self, model, loaders, optimizer, num_epochs, scheduler=None):

        torch.cuda.empty_cache()
        resource_monitor = ResourceMonitor()

        model = model.to(self.device)

        report_metrics = {
            'loss': {
                'train': [],
                'val': []
            }
        }


        tb_step = -1
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:

                self.phase = phase
                model.train(phase == 'train')  # Set model to training mode

                loss_accum = 0
                step_count = len(loaders[phase])
                for i_step, (img, target) in enumerate(loaders[phase]):
                    with torch.set_grad_enabled(phase == 'train'):
                        loss_value = self.loss_calc(img, target, model)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss_value.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                    loss_accum += loss_value.item()
                    self.logger.add_scalar('Loss_{}/batch'.format(phase), loss_value.item(), tb_step)
                    report_metrics['loss'][phase].append(loss_value.item())
                    tb_step += 1

                    print('Epoch {}/{}. Phase {} Step {}/{} Loss {}'.format(epoch, num_epochs - 1, phase,
                                                                            i_step, step_count, loss_value.item()))

                ave_loss = loss_accum / step_count
                self.logger.add_scalar('Loss_train/epoch', ave_loss, epoch)

            '''
            with torch.inference_mode():
                for phase in ('train', 'val'):
                    m_map = self.evaluate(model, loaders[phase])
            '''

        tune.report(
            train_loss=sum(report_metrics['loss']['train']) / len(report_metrics['loss']['train']),
            val_loss=sum(report_metrics['loss']['val']) / len(report_metrics['loss']['val'])
        )
