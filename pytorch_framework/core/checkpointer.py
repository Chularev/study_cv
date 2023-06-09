import os
import torch

from core.train_context import _TrainContext
from core.train_param_enums import LoadStrategy, SaveStrategy, MetricType
from helpers.constants import CHECKPOINT_FILE


class Checkpointer:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.best_metric = 0
        
    def load(self):

        if not os.path.exists(CHECKPOINT_FILE):
            return False

        if self.c.load_strategy == LoadStrategy.NONE:
            return False

       # checkpoint = os.path.join(CHECKPOINT_FILE)
        model_state, optimizer_state, metric = torch.load(CHECKPOINT_FILE)

        self.best_metric = metric

        if self.c.load_strategy == LoadStrategy.MODEL_OPTIMIZER:
            self.c.optimizer.load_state_dict(optimizer_state)

        self.c.model.load_state_dict(model_state)

        return True

    def _check_metric(self, template, current):
        if self.c.metric_type == MetricType.METRIC:
            if template <= current:
                return True

        # metric is loss
        if template >= current:
            return True

        return False

    def save(self, metric):
        if self.c.save_strategy == SaveStrategy.NONE:
            return False

        if self.c.save_strategy == SaveStrategy.BEST_MODEL_OPTIMIZER or self.c.save_strategy == SaveStrategy.BEST_MODEL:
            if not self._check_metric(self.best_metric, metric):
                return False

        self.best_metric = metric

        checkpoint = {
            'model': self.c.model,
            'metric': self.best_metric
        }

        if self.c.save_strategy == SaveStrategy.BEST_MODEL_OPTIMIZER or self.c.save_strategy == SaveStrategy.MODEL_OPTIMIZER:
            checkpoint['optimizer'] = self.c.optimizer

        torch.save(checkpoint, CHECKPOINT_FILE)
        return True

    def _save_on_disk(self, metric):
        self.best_metric = metric

        checkpoint = {
            "state_dict": self.c.model.state_dict(),
            "optimizer": self.c.optimizer.state_dict(),
            "metric": self.best_metric
        }
        torch.save(checkpoint, CHECKPOINT_FILE)

    def is_finish(self, metric):

        result = self._check_metric(self.c.metric_value_stop, metric)
        if result:
            self._save_on_disk(metric)

        return result

if __name__ == "__main__":
    context = _TrainContext()
    context.load_strategy = LoadStrategy.MODEL

    checkopinter = Checkpointer(context)
    print(checkopinter.load())