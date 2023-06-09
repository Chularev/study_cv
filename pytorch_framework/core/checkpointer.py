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

        if not os.path.exists(CHECKPOINT_DIR):
            return False

        if self.c.load_strategy == LoadStrategy.NONE:
            return False

        checkpoint = os.path.join(CHECKPOINT_FILE)
        model_state, optimizer_state, metric = torch.load(checkpoint)

        self.best_metric = metric

        if self.c.load_strategy == LoadStrategy.MODEL_OPTIMIZER:
            self.c.optimizer.load_state_dict(optimizer_state)

        self.c.model.load_state_dict(model_state)

        return True

    def _save_on_disk(self, metric):
        self.best_metric = metric
        return 0
    def save(self):
        return 0

    def is_finish(self, metric):
        if self.c.metric_type == metric:
            if self.c.metric_value_stop <= metric:
                self._save_on_disk(metric)
                return True

        if self.c.metric_value_stop >= metric:
            self._save_on_disk(metric)
            return True

        return False

if __name__ == "__main__":
    context = _TrainContext()
    context.load_strategy = LoadStrategy.MODEL

    CHECKPOINT_DIR = '/home/alex/workspace/experiments/best_model/'
    checkopinter = Checkpointer(context)
    print(checkopinter.load())