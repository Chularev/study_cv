
class ExtendedModel:
    def __init__(self, torch_model, need_train):
        self.torch_model = torch_model
        self.need_train = need_train

        self.loss_history = None
        self.train_history = None
        self.val_history = None

    def add_history(self, loss_history, train_history, val_history):
        self.loss_history = loss_history
        self.train_history = train_history
        self.val_history = val_history

    def load_model(self, model_name, model):

        checkpoint = torch.load(self.output + '/' + model_name)
        model.load_state_dict(checkpoint['model_state_dict'])

        loss_history = checkpoint['loss_history']
        train_history = checkpoint['train_history']
        val_history = checkpoint['val_history']

        for i in range(len(val_history)):
            print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (
            loss_history[i], train_history[i], val_history[i]))

        return loss_history, train_history, val_history

    def save_model(self, model_name, model, loss_history, train_history, val_history):

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'train_history': train_history,
            'val_history': val_history
        }, self.output + '/' + model_name)

        return 0