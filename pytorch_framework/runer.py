from itertools import product
from pytorch_helper import PyTorchHelper
from extended_model import ExtendedModel

import torch
import torch.nn as nn
import torch.optim as optim

from models import Net
import ray
from ray import tune
from data_handlers.data_preparer import get_datasets


def find_hyperparameters(config, datasets):
    # define training and validation data_handlers loaders
    data_loader = torch.utils.data.DataLoader(
        datasets['train'], batch_size=64, shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(
        datasets['test'], batch_size=64, shuffle=False)

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

if __name__ == "__main__":
    config = {
        'need_train': True,
        'regs': [0.001],
        'optimizers': [optim.Adam],
        'model': Net,
        'model_name': 'best_lenet'
    }

    datasets = get_datasets()
    analysis = tune.run(
        tune.with_parameters(find_hyperparameters, datasets=datasets),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        name="experiment_name",
        local_dir="/mnt/heap/My folder/checpoint",
        num_samples=1,
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1})

