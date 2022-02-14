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
from typing import Tuple, Dict

def get_loaders(datasets) -> Dict[str,  torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(
            datasets['train'], batch_size=64, shuffle=True),
        'val': torch.utils.data.DataLoader(
            datasets['val'], batch_size=64, shuffle=False)
    }


def find_hyperparameters(config, datasets):
    # define training and validation data_handlers loaders

    loaders = get_loaders(datasets)
    helper = PyTorchHelper(8,  None)

    lenet_model = ExtendedModel(config['model'](), config['need_train'], config['model_name'])
    if not lenet_model.need_train:
        if lenet_model.load_model():
            return lenet_model

    optimizer = config['optimizer'](lenet_model.torch_model.parameters(), lr=config['learning_rate'], weight_decay=config['reg'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['anneal_epoch'], gamma=config['anneal_coeff'])

    model, train_loss_history, val_loss_history, train_metric_history, val_metric_history = helper.train_model \
        (lenet_model.torch_model, loaders, optimizer, config['epoch_num'] , scheduler)
    lenet_model.add_history(train_loss_history, val_loss_history, train_metric_history, val_metric_history)

    return lenet_model

if __name__ == "__main__":
    config = {
        'need_train': True,
        'reg': 0.01,
        'optimizer': optim.Adam,
        'model': Net,
        'model_name': 'best_lenet',
        'learning_rate': 1e-1,
        'anneal_epoch': 5,
        'anneal_coeff': 0.5,
        'epoch_num': tune.grid_search([2,3])
    }

    datasets = get_datasets()
    analysis = tune.run(
        tune.with_parameters(find_hyperparameters, datasets=datasets),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        name="experiment_name",
        local_dir="/mnt/heap/My folder/checkpoint",
        num_samples=1,
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1})

