CUDA_LAUNCH_BLOCKING=1

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

    def __init__(self,  batch_size, data):
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
            loss_value = loss_value + loss_function_xy(prediction[:, 1:][indexes_with_label], gpu_box[indexes_with_label])
        return loss_value

    def train_model(self, model, train_loader, val_loader,  optimizer, num_epochs, scheduler=None):

        torch.cuda.empty_cache()
        resourceMonitor = ResourceMonitor()

        model.type(torch.cuda.FloatTensor)
        model.to(self.device)

        train_loss_history = []
        val_loss_history = []

        train_metric_history = []
        val_metric_history = []

        print('=' * 30)
        print("Start train:")
        resourceMonitor.print_statistics('MB')
        print('=' * 30)

        for epoch in range(num_epochs):
            model.train()  # Enter train mode

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            loss_accum = 0
            step_count = len(train_loader)
            for i_step, (img, target) in enumerate(train_loader):

                loss_value = self.loss_calc(img,target,model)

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                print('Step {}/{} Loss {}'.format(i_step, step_count, loss_value))
                loss_accum += loss_value

            if scheduler is not None:
                scheduler.step()

            ave_loss = loss_accum / i_step

            train_loss_history.append(float(ave_loss))

            model.eval()

            print('=' * 30)
            print("Average loss train: %f" % (ave_loss))
            map = Metrics.iou(model,train_loader)
            train_metric_history.append(map)
            print("Train map: %f" % (map))

            loss_accum = 0
            for i_step, (img, target) in enumerate(val_loader):
                with torch.no_grad():
                    loss_value = self.loss_calc(img,target,model)
                    loss_accum += loss_value

            ave_loss = loss_accum / i_step

            val_loss_history.append(float(ave_loss))
            print("Average loss test: %f" % (ave_loss))
            map = Metrics.iou(model, val_loader)
            val_metric_history.append(map)
            print("Test map: %f" % (map))
            print('=' * 30)
            resourceMonitor.print_statistics('MB')
            print('=' * 30)

        model = model.to(torch.device('cpu'))
        return model, train_loss_history, val_loss_history, train_metric_history, val_metric_history

