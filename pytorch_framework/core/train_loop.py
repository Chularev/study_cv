import torch
import torch.optim as optim
import os

from core.train_context import _TrainContext
from core.trainer import Trainer
from core.validator import Validator
from typing import Dict
from helpers.logger import Logger
from core.train_parameters import TrainParameters


class Looper:
    def __init__(self, context: _TrainContext):
        self.c = context
        self.trainer = Trainer(context)
        self.validator = Validator(context)

    def train_loop(self):
        torch.cuda.empty_cache()

        self.c.model = self.c.model.to(self.c.device)

        for epoch in range(self.c.epoch_num):

            metric = self.validator.validate(epoch)
            self.c.logger.add_scalar('Validation/epoch/metric', metric)

            loss = self.trainer.train(epoch)
            self.c.logger.add_scalar('Train/epoch/loss', loss)
            '''
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

                for index in range(100, 120):
                    target = self.datasets[phase][index]
                    predict = model(self.to_gpu(target['image'].unsqueeze(0)))
                    predict = predict[0].to('cpu')

                    img_grid = self.viewer.create_output(target, predict)
                    self.logger.add_grid_images('Output ' + str(index), img_grid)

                ave_loss = loss_accum / step_count
                self.logger.add_scalar('Loss_sum_train/epoch', ave_loss)
                '''


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


def start_train_loop(parameters, datasets, checkpoint_dir=None):
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
    context.checkpoint_frequency = p.checkpoint_frequency

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        context.model.load_state_dict(model_state)
        context.optimizer.load_state_dict(optimizer_state)

    context.epoch_num = p.epoch_num

    context.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context.logger = Logger('TensorBoard')

    looper = Looper(context)
    looper.train_loop()
