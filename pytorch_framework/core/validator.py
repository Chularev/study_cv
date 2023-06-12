import torch
from losses.Yolov1 import YoloLoss
from core.train_context import _TrainContext
from helpers.bar import Bar


class Validator:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.metrics = self.c.metric(self.c.device)

    def metric_calc(self, target, model, train_idx):
        img, bboxes = target

        img = img.to(self.c.device, torch.float)
        bboxes = bboxes.to(self.c.device, torch.float)

        prediction = model(img)

        metrics = self.metrics(prediction, bboxes, img.shape[0], train_idx)
        for key in metrics.keys():
            item = metrics[key].item()
            self.bar.set(key,item)
            self.c.logger.add_scalar('Validation/batch/metric/{}'.format(key), item)

        return sum(metrics.values())

    def bar_init(self, epoch):
        self.bar = Bar(self.c.val_loader)
        self.bar.set('phase', 'validation')
        self.bar.set('epoch', epoch)

    @torch.no_grad()
    def validate(self, epoch):

        self.bar_init(epoch)

        self.c.model.eval()

        metric_accum = 0
        for i_step, target in enumerate(self.bar):
            metric = self.metric_calc(target, self.c.model, i_step)

            metric_accum += metric.item()
            self.c.logger.add_scalar('Validation/batch/metric/sum', metric.item())

            ave_metric = metric_accum / (i_step + 1)
            self.c.logger.add_scalar('Validation/batch/metric/average', ave_metric)

            self.bar.set('ave_metric', ave_metric)
            self.bar.update()

        metric = self.metrics.compute()
        return metric['map'].item()
