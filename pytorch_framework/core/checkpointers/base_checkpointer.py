import os
import torch

from core.checkpointers.checkpoint_context import _CheckpointContext
from core.train_context import _TrainContext
from core.train_param_enums import LoadStrategy, SaveStrategy, MetricType
from helpers.constants import CHECKPOINT_FOLDER


class BaseCheckpointer:

    def __init__(self, context: _CheckpointContext, t_context: _TrainContext):
        self.c = context
        self.tc = t_context
        self.best_metric = 0
        
    def load(self):

        if not os.path.exists(self.c.file):
            print("Model is not exist")
            return False

        if self.c.load_strategy == LoadStrategy.NONE:
            print('Strategy of model load is NONE')
            return False

       # checkpoint = os.path.join(CHECKPOINT_FILE)
        checkpoint = torch.load(self.c.file)

        self.best_metric = checkpoint['metric']

        if self.c.load_strategy == LoadStrategy.MODEL_OPTIMIZER:
            self.tc.optimizer.load_state_dict(checkpoint['optimizer_state'])

        self.tc.model.load_state_dict(checkpoint['model_state'])

        print("Model was loaded. Current metric is ", self.best_metric)
        return True

    def _check_metric(self, template, current):
        if self.c.metric_type == MetricType.METRIC:
            return template < current

        # metric is loss
        return template > current

    def _save_checkpoint(self, checkpoint):
        self._create_recursive_dir()
        self._remove_file()
        torch.save(checkpoint, self.c.file)

        print('Model saved current metric is ', self.best_metric)

    def _create_recursive_dir(self):
        if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)

    def _remove_file(self):
        if os.path.exists(self.c.file):
            os.remove(self.c.file)

    def save(self, metric):
        if self.c.save_strategy == SaveStrategy.NONE:
            print('Model was not save because save strategy is None')
            return False

        if self.c.save_strategy == SaveStrategy.BEST_MODEL_OPTIMIZER or self.c.save_strategy == SaveStrategy.BEST_MODEL:
            if not self._check_metric(self.best_metric, metric):
                print("Model was not save because current metric is ", metric, ' but the best metric is ', self.best_metric)
                return False

        self.best_metric = metric

        checkpoint = {
            'model_state': self.tc.model.state_dict(),
            'metric': self.best_metric
        }

        if self.c.save_strategy == SaveStrategy.BEST_MODEL_OPTIMIZER or self.c.save_strategy == SaveStrategy.MODEL_OPTIMIZER:
            checkpoint['optimizer_state'] = self.tc.optimizer.state_dict()

        self._save_checkpoint(checkpoint)

        return True

    def _save_on_disk(self, metric):
        self.best_metric = metric

        checkpoint = {
            "model_state": self.tc.model.state_dict(),
            "optimizer_state": self.tc.optimizer.state_dict(),
            "metric": self.best_metric
        }
        self._save_checkpoint(checkpoint)

    def is_finish(self, metric):

        result = self._check_metric(self.c.metric_value_stop, metric)
        if result:
            self._save_on_disk(metric)
            print('Model saved current metric is ', self.best_metric, ' train finished')

        return result

if __name__ == "__main__":
    n = {'hgjkl;': 0, 'll;': 5, '7': 8}
    context,j,k = n

    print(context)
    print(k)