from trainer import Trainer

import torch
import torch.optim as optim
import os

from models import Net
from ray.tune import CLIReporter
from ray import tune
from data_handlers.data_preparer import get_datasets
from typing import Dict

def get_loaders(datasets) -> Dict[str,  torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(
            datasets['train'], batch_size=64, shuffle=True),
        'val': torch.utils.data.DataLoader(
            datasets['val'], batch_size=64, shuffle=False)
    }


def find_hyperparameters(config, datasets, checkpoint_dir=None):
    # define training and validation data_handlers loaders

    loaders = get_loaders(datasets)
    trainer = Trainer()

    model = config['model']()

    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], weight_decay=config['reg'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['anneal_epoch'], gamma=config['anneal_coeff'])

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainer.train(model, loaders, optimizer, config['epoch_num'], scheduler)

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
        'epoch_num': 10
    }

    datasets = get_datasets()

    reporter = CLIReporter(max_report_frequency=30)
    analysis = tune.run(
        tune.with_parameters(find_hyperparameters, datasets=datasets),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        name="experiment_name",
        local_dir="/mnt/heap/My folder/tune_reports",
        num_samples=1,
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1},
        progress_reporter=reporter)

