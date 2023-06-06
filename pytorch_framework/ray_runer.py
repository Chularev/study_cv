from core.train_loop import start_train_loop
from core.train_parameters import TrainParameters
import torch.optim as optim
from models.Yolov1 import Yolov1
from ray.tune import CLIReporter
from ray import tune
from dataset.dataset_helperr import DatasetHelper

def get_parameters():
    p = TrainParameters()

    p.need_train = True
    p.reg = 0.0001
    p.optimizer = optim.Adam
    p.model = Yolov1
    p.model_name = 'best_net'
    p.learning_rate = 1e-3
    p.scheduler_epoch = 100
    p.scheduler_coefficient = 0.1
    p.epoch_num = 1
    return p



if __name__ == "__main__":

    datasets = DatasetHelper.get_datasets()

    reporter = CLIReporter(max_report_frequency=30)
    analysis = tune.run(
        tune.with_parameters(start_train_loop, datasets=datasets),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        name="YoloV1",
        local_dir="/home/alex/workspace/experiments/",
        num_samples=1,
        config={'params': get_parameters()},
        resources_per_trial={"cpu": 3, "gpu": 1},
        progress_reporter=reporter
    )

