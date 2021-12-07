CUDA_LAUNCH_BLOCKING = 1

import numpy as np

import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
from metrics import Metrics
from torchvision import transforms
from resource_monitor import ResourceMonitor


class PyTorchHelper:

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def split(self, validation_split):

        data_size = self.data.data.shape[0]
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices

    def loss_calc(self, img, target, model):
        loss_function_xy = torch.nn.SmoothL1Loss()
        loss_function_bce = torch.nn.BCEWithLogitsLoss()

        gpu_img = img.type(torch.cuda.FloatTensor)
        gpu_img = gpu_img.to(self.device)
        prediction = model(gpu_img)

        gpu_img_has_person = target['img_has_person'].type(torch.cuda.FloatTensor)
        gpu_img_has_person = gpu_img_has_person.to(self.device)

        gpu_box = target['box'].type(torch.cuda.FloatTensor)
        gpu_box = gpu_box.to(self.device)

        loss_value = loss_function_bce(prediction[:, 0], gpu_img_has_person)

        indexes_with_label = (gpu_img_has_person == 1).nonzero(as_tuple=True)
        if len(indexes_with_label) > 0:
            return loss_value + loss_function_xy(prediction[:, 1:][indexes_with_label], gpu_box[indexes_with_label])
        return loss_value

    @torch.inference_mode()
    def evaluate(self, model, data_loader):
        return Metrics.iou(model, data_loader)

    def train_model(self, model, train_loader, val_loader, optimizer, num_epochs, scheduler=None):

        torch.cuda.empty_cache()
        resource_monitor = ResourceMonitor()

        model.type(torch.cuda.FloatTensor)
        model.to(self.device)

        loss_history = {
            'train': [],
            'val': []
        }

        metric_history = {
            'train': [],
            'val': []
        }

        loaders = {
            'train': train_loader,
            'val': val_loader

        }

        print('=' * 30)
        print("Start train:")
        resource_monitor.print_statistics('MB')
        print('=' * 30)

        for epoch in range(num_epochs):
            model.train()  # Enter train mode

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            loss_accum = 0
            step_count = len(train_loader)
            for i_step, (img, target) in enumerate(train_loader):
                loss_value = self.loss_calc(img, target, model)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                print('Step {}/{} Loss {}'.format(i_step, step_count, loss_value))
                loss_accum += loss_value

            if scheduler is not None:
                scheduler.step()

            ave_loss = loss_accum / step_count

            loss_history['train'].append(float(ave_loss))

            print('-' * 30)
            print("Average loss train: %f" % ave_loss)

            loss_accum = 0
            model.eval()
            with torch.no_grad():
                for i_step, (img, target) in enumerate(val_loader):
                    loss_value = self.loss_calc(img, target, model)
                    loss_accum += loss_value

                ave_loss = loss_accum / len(val_loader)

                loss_history['val'].append(float(ave_loss))
                print("Average loss test: %f" % (ave_loss))

            print('=' * 30)
            with torch.inference_mode():
                for phase in ('train', 'val'):
                    m_map = self.evaluate(model, loaders[phase])
                    metric_history[phase].append(m_map)
                    print("{0} map: {1}".format(phase, m_map))

            print('=' * 30)
            resource_monitor.print_statistics('MB')
            print('=' * 30)
        model = model.to(torch.device('cpu'))
        return model, loss_history['train'], loss_history['val'], metric_history['train'], metric_history['val']
