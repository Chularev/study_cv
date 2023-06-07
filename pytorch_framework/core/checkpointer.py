from core.train_context import _TrainContext


class Checkpointer:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.best_metric = 0
    def load(self):
        '''
           if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        context.model.load_state_dict(model_state)
        context.optimizer.load_state_dict(optimizer_state)
        :return:
        '''
        return 0

    def save(self):
        return 0

    def is_finish(self):
        return False