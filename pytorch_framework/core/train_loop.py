import torch
import torch.optim as optim
import os

from core.train_context import _TrainContext
from core.trainer import Trainer
from core.validator_metric import ValidatorMetric
from core.validator_loss import ValidatorLoss
from core.checkpointer import Checkpointer


class Looper:
    def __init__(self, context: _TrainContext):
        self.c = context
        self.trainer = Trainer(context)
        self.validator_metric = ValidatorMetric(context)
        self.validator_loss = ValidatorLoss(context)
        self.checkpointer = Checkpointer(context)

    def train_loop(self):
        torch.cuda.empty_cache()

        self.c.model = self.c.model.to(self.c.device)
        self.checkpointer.load()

        if self.checkpointer.is_finish(self.checkpointer.best_metric):
            return

        for epoch in range(self.c.epoch_num):

            if epoch % self.c.checkpoint_frequency == 0 and epoch > 0:
                metric = self.validator_metric.validate(epoch)
                self.c.logger.add_scalar('Validation/epoch/metric', metric)

                if self.checkpointer.is_finish(metric):
                    break

                self.checkpointer.save(metric)



            loss = self.trainer.train(epoch)
            self.c.logger.add_scalar('Train/epoch/loss_train', loss)

            loss = self.validator_loss.validate(epoch)
            self.c.logger.add_scalar('Train/epoch/loss_train', loss)
