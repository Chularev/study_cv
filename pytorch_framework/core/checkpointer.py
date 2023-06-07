from core.train_context import _TrainContext


class Checkpointer:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.best_metric = 0

    def load(self):
        return 0

    def save(self):
        return 0