import torch
import torch.optim as optim
import os

from core.train_config import _TrainConfig
from core.trainer import Trainer
from core.validator import Validator
from typing import Dict
from helpers.logger import Logger
from core.train_parameters import TrainParameters
class Looper:
    def __init__(self, config: _TrainConfig):
        self.c = config
        self.trainer = Trainer(config)
        self.validator = Validator(config)

    def train_loop(self):

        torch.cuda.empty_cache()

        self.c.model = self.c.model.to(self.c.device)

        for epoch in range(self.c.epoch_num):
            self.validator.validate(epoch)
            self.trainer.train(epoch)
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

def get_loaders(datasets) -> Dict[str,  torch.utils.data.DataLoader]:
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
    # define training and validation data_handlers loaders

    p = covert_to_TrainParameters(parameters)
    loaders = get_loaders(datasets)
    train_config = _TrainConfig()

    train_config.train_loader = loaders['train']
    train_config.val_loader = loaders['val']

    train_config.model = p.model(split_size=7, num_boxes=2, num_classes=20)

    train_config.optimizer = p.optimizer(
        train_config.model.parameters(), lr=p.learning_rate, weight_decay=p.reg
    )

    train_config.scheduler = optim.lr_scheduler.StepLR(
        train_config.optimizer, step_size=p.scheduler_epoch, gamma=p.scheduler_coefficient
    )

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        train_config.model.load_state_dict(model_state)
        train_config.optimizer.load_state_dict(optimizer_state)

    train_config.epoch_num = p.epoch_num

    train_config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_config.logger = Logger('TensorBoard')

    looper = Looper(train_config)
    looper.train_loop()