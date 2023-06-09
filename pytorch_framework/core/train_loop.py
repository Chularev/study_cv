import torch
import torch.optim as optim
import os

from core.train_context import _TrainContext
from core.trainer import Trainer
from core.validator import Validator
from core.checkpointer import Checkpointer


class Looper:
    def __init__(self, context: _TrainContext):
        self.c = context
        self.trainer = Trainer(context)
        self.validator = Validator(context)
        self.checkpointer = Checkpointer(context)

    def train_loop(self):
        torch.cuda.empty_cache()

        self.checkpointer.load()

        self.c.model = self.c.model.to(self.c.device)

        for epoch in range(self.c.epoch_num):

            metric = self.validator.validate(epoch)
            self.c.logger.add_scalar('Validation/epoch/metric', metric)

            if epoch % self.c.checkpoint_frequency == 0:
                self.checkpointer.save(metric)

            if self.checkpointer.is_finish(metric):
                break

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
