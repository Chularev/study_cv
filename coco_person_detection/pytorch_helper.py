import numpy as np

import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms

class PyTorchHelper:

    def __init__(self,  batch_size, data):
        self.output = 'output'
        self.batch_size = batch_size
        self.data = data
        self.device = torch.device("cuda:0")  # Let's make sure GPU is available!

    def split(self, validation_split):

        data_size = self.data.data.shape[0]
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices

    def load_model(self, model_name, model):

        checkpoint = torch.load(self.output + '/' + model_name)
        model.load_state_dict(checkpoint['model_state_dict'])

        loss_history = checkpoint['loss_history']
        train_history = checkpoint['train_history']
        val_history = checkpoint['val_history']

        for i in range(len(val_history)):
            print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (loss_history[i], train_history[i], val_history[i]))

        return loss_history, train_history, val_history

    def save_model(self, model_name, model, loss_history, train_history, val_history):

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'train_history' : train_history,
            'val_history': val_history
        }, self.output + '/' + model_name)

        return 0

    def train_model(self, model_name, model, train_loader, val_loader, loss, optimizer, num_epochs, scheduler=None):

        #if os.path.isfile(self.output + '/' + model_name):
         #   return self.load_model(model_name, model)

        device = torch.device("cuda:0")
        model.type(torch.cuda.FloatTensor)
        model.to(device)

        loss_history = []
        train_history = []
        val_history = []
        loss_function_xy = torch.nn.MSELoss()
        loss_function_bce = torch.nn.BCELoss()

        for epoch in range(num_epochs):
            model.train()  # Enter train mode

            loss_accum = 0
            for i_step, (img, target) in enumerate(train_loader):

                img = img.to(torch.float)
                target['img_has_person'] = target['img_has_person'].to(torch.float)
                target['box'] = target['box'].to(torch.float)

                img = img.to(device)
                target['img_has_person'] = target['img_has_person'].to(device)
                target['box'] = target['box'].to(device)

                prediction = model(img)

                print('prediction')
                print(prediction)
                print('target')
                print(target)
                print('my_prediction_BCE')
                print(prediction[:,0])

                print('my_prediction_BBOX')
                print(prediction[:,1:])

                loss_value = loss_function_bce(prediction[:,0], target['img_has_person'])
                if target['img_has_person'] == 1:
                    loss_value + loss_function_xy(prediction[:,1:], target['box'])
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                loss_accum += loss_value

            if scheduler is not None:
                scheduler.step()

            ave_loss = loss_accum / i_step

            loss_history.append(float(ave_loss))

            print("Average loss: %f" % (ave_loss))

        #self.save_model(model_name, model, loss_history, train_history, val_history)
        return loss_history, train_history, val_history

    def compute_accuracy(self, model, loader):
        """
        Computes accuracy on the dataset wrapped in a loader

        Returns: accuracy as a float value between 0 and 1
        """
        model.eval()  # Evaluation mode
        # TODO: Copy implementation from previous assignment
        # Don't forget to move the data to device before running it through the model!

        total_samples = 0
        correct_samples = 0
        for i_step, (x, y) in enumerate(loader):
            x_gpu = x.to(self.device)
            y_gpu = y.to(self.device)

            prediction = model(x_gpu)

            _, indices = torch.max(prediction, 1)

            total_samples += y.shape[0]
            correct_samples += torch.sum(indices == y_gpu)

        return float(correct_samples) / total_samples
