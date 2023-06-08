
from typing import Dict
from helpers.logger import Logger
import torch
import torch.optim as optim

from core.train_context import _TrainContext
from core.train_parameters import TrainParameters
from core.train_loop import Looper

def get_loaders(datasets) -> Dict[str, torch.utils.data.DataLoader]:
    return {
        'train': torch.utils.data.DataLoader(
            datasets['train'], batch_size=16, shuffle=True
        ),
        'val': torch.utils.data.DataLoader(
            datasets['train'], batch_size=16, shuffle=True
        )
    }


'''
    This function used to work autocode helper
'''
def covert_to_TrainParameters(parameters) -> TrainParameters:
    return parameters['params']


def start_train(parameters, datasets, checkpoint_dir=None):
    # define training and validation dataset loaders

    p = covert_to_TrainParameters(parameters)
    loaders = get_loaders(datasets)
    context = _TrainContext()

    context.train_loader = loaders['train']
    context.val_loader = loaders['val']

    context.model = p.model(split_size=7, num_boxes=2, num_classes=20)

    context.optimizer = p.optimizer(
        context.model.parameters(), lr=p.learning_rate, weight_decay=p.reg
    )

    context.scheduler = optim.lr_scheduler.StepLR(
        context.optimizer, step_size=p.scheduler_epoch, gamma=p.scheduler_coefficient
    )

    # Checkpoint
    context.load_strategy = p.load_strategy
    context.save_strategy = p.save_strategy
    context.metric_type = p.metric_type
    context.metric_value_stop = p.metric_value_stop
    context.checkpoint_frequency = p.checkpoint_frequency



    context.epoch_num = p.epoch_num

    context.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context.logger = Logger('TensorBoard')

    looper = Looper(context)
    looper.train_loop()