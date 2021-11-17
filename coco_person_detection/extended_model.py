import torch
import os


class ExtendedModel:
    def __init__(self, torch_model, need_train, model_name):

        self.output = 'output'

        self.torch_model = torch_model
        self.need_train = need_train
        self.model_name = model_name

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_metric_history = []
        self.val_metric_history = []


    def add_history(self, train_loss_history, val_loss_history, train_metric_history, val_metric_history):
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        self.train_metric_history = train_metric_history
        self.val_metric_history = val_metric_history

    def load_model(self):

        if not os.path.isfile(self.output + '/' + self.model_name):
            return False

        checkpoint = torch.load(self.output + '/' + self.model_name)
        self.torch_model.load_state_dict(checkpoint['model_state_dict'])

        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']

        train_metric_history = checkpoint['train_metric_history']
        val_metric_history = checkpoint['val_metric_history']

        for i in range(len(train_loss_history)):
            print("epoch: %d, Train loss: %f, Val loss: %f, Train metric: %f, Val metric: %f" % (i, train_loss_history[i], val_loss_history[i], train_metric_history[i], val_metric_history[i]))

        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        self.train_metric_history = train_metric_history
        self.val_metric_history = val_metric_history

        return True

    def save_model(self):

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        torch.save({
            'model_state_dict': self.torch_model.state_dict(),

            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,

            'train_metric_history': self.train_metric_history,
            'val_metric_history': self.val_metric_history
        }, self.output + '/' + self.model_name)

    def load_best_model(self):
        self.model_name = 'the_best'
        self.load_model()

    def save_best_model(self):
        self.model_name = 'the_best'
        self.save_model()
