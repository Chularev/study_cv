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

        self.c.model = self.c.model.to(self.c.device)
        self.checkpointer.load()

        for epoch in range(self.c.epoch_num):

            if epoch % self.c.checkpoint_frequency == 0:
                metric = self.validator.validate(epoch)
                self.c.logger.add_scalar('Validation/epoch/metric', metric)

                if self.checkpointer.is_finish(metric):
                    break

                self.checkpointer.save(metric)

            loss = self.trainer.train(epoch)
            self.c.logger.add_scalar('Train/epoch/loss', loss)