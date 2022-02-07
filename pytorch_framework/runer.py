from itertools import product
from pytorch_helper import PyTorchHelper
from extended_model import ExtendedModel

import torch
import torch.nn as nn
import torch.optim as optim

def find_hyperparameters(config, data_train, data_test):
    # define training and validation data_handlers loaders
    data_loader = torch.utils.data.DataLoader(
        data_train, batch_size=64, shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(
        data_test, batch_size=64, shuffle=False)

    learning_rates = [1e-1]
    anneal_coeff = 0.5
    anneal_epochs = [5]
    regs = config['regs']
    optimizers = config['optimizers']

    batch_size = 64
    epoch_num = 5

    run_record = {}

    helper = PyTorchHelper(8,  None)

    lenet_model = None
    val_loss = 8
    loss_history = None
    for lr, reg, anneal_epoch, optimizer in product(learning_rates, regs, anneal_epochs, optimizers):

        lenet_model = ExtendedModel(config['model'](), config['need_train'], config['model_name'])
        if not lenet_model.need_train:
            if lenet_model.load_model():
                return lenet_model

        optimizer = optimizer(lenet_model.torch_model.parameters(), lr=lr, weight_decay=reg)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=anneal_epoch, gamma=anneal_coeff)

        model, train_loss_history, val_loss_history, train_metric_history, val_metric_history = helper.train_model \
            (lenet_model.torch_model, data_loader, data_loader_test, optimizer, epoch_num, scheduler)
        lenet_model.add_history(train_loss_history, val_loss_history, train_metric_history, val_metric_history)

    return lenet_model
