import torch

from core.train_context import _TrainContext
from core.trainer import Trainer
from core.validator_metric import ValidatorMetric
from core.validator_loss import ValidatorLoss
from core.checkpointers.base_checkpointer import BaseCheckpointer


class Looper:
    def __init__(self, context: _TrainContext):
        self.c = context
        self.trainer = Trainer(context)
        self.validator_metric = ValidatorMetric(context)
        self.validator_loss = ValidatorLoss(context)

    def train_loop(self):
        torch.cuda.empty_cache()

        self.c.model = self.c.model.to(self.c.device)
        self.c.metric_checkpointer.load()

        if self.c.metric_checkpointer.is_finish(self.c.metric_checkpointer.best_metric):
            return

        for epoch in range(self.c.epoch_num):

            if epoch % self.c.metric_checkpointer.c.checkpoint_frequency == 0 and epoch > 0:
                metric = self.validator_metric.step(epoch)
                self.c.logger.add_scalar('Validation/epoch/metric', metric)

                if self.c.metric_checkpointer.is_finish(metric):
                    break

                self.c.metric_checkpointer.save(metric)

            loss = self.trainer.step(epoch)
            self.c.logger.add_scalar('Train/epoch/loss_train', loss)

            loss = self.validator_loss.step(epoch)
            self.c.logger.add_scalar('Train/epoch/loss_train', loss)
