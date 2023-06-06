from core.train_loop import start_train_loop

import torch
import torch.optim as optim
import os

from models.Yolov1 import Yolov1
from ray.tune import CLIReporter
from ray import tune
from data_handlers.data_preparer import get_datasets

config = {
    'need_train': True,
    'reg': 0.0001,
    'optimizer': optim.Adam,
    'model': Yolov1,
    'model_name': 'best_net',
    'learning_rate': 1e-3,
    'scheduler_epoch': 100,
    'scheduler_coefficient': 0.1,
    'epoch_num': 200
}



if __name__ == "__main__":

    datasets = get_datasets()

    reporter = CLIReporter(max_report_frequency=30)
    analysis = tune.run(
        tune.with_parameters(start_train_loop, datasets=datasets),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        name="YoloV1",
        local_dir="/home/alex/workspace/experiments/",
        num_samples=1,
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1},
        progress_reporter=reporter
    )

