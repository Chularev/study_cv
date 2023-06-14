
from core.checkpointers.base_checkpointer import BaseCheckpointer
from core.checkpointers.checkpoint_context import _CheckpointContext
from core.train_context import _TrainContext
from core.validator_metric import ValidatorMetric


class MetricCheckpointer(BaseCheckpointer):

    def __int__(self, c_context: _CheckpointContext, t_context: _TrainContext):
        super().__init__(c_context, t_context)
        self.validator_metric = ValidatorMetric(t_context)

    def step(self, epoch):
        if epoch % self.c.checkpoint_frequency == 0 and epoch > 0:
            metric = self.validator_metric.step(epoch)
            self.tc.logger.add_scalar('Validation/epoch/metric', metric)

            if self.is_finish(metric):
                return True

            self.save(metric)