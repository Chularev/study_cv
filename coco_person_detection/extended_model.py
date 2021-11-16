import torch
import os


class ExtendedModel:
    def __init__(self, torch_model, need_train, model_name):

        self.output = 'output'

        self.torch_model = torch_model
        self.need_train = need_train
        self.model_name = model_name

        self.loss_history = None
        self.train_history = None
        self.val_history = None

    def add_history(self, train_history, val_history):
        #self.loss_history = loss_history
        self.train_history = train_history
        self.val_history = val_history

    def load_model(self):

        if not os.path.isfile(self.output + '/' + self.model_name):
            return False

        checkpoint = torch.load(self.output + '/' + self.model_name)
        self.torch_model.load_state_dict(checkpoint['model_state_dict'])

        loss_history = checkpoint['loss_history']
        train_history = checkpoint['train_history']
        val_history = checkpoint['val_history']

        for i in range(len(val_history)):
            print("Average loss: %f, Train accuracy: %f, Val accuracy: %f"
                  % (loss_history[i], train_history[i], val_history[i]))

        self.loss_history = loss_history
        self.train_history = train_history
        self.val_history = val_history
        return True

    def save_model(self):

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        torch.save({
            'model_state_dict': self.torch_model.state_dict(),
            'loss_history': self.loss_history,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, self.output + '/' + self.model_name)

    def load_best_model(self):
        self.model_name = 'the_best'
        self.load_model()

    def save_best_model(self):
        self.model_name = 'the_best'
        self.save_model()
