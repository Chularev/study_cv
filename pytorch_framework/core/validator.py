import torch
from losses.Yolov1 import YoloLoss
from core.train_context import _TrainContext
from helpers.bar import Bar


class Validator:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.metrics = YoloLoss()
        self.bar = Bar(self.c.val_loader)
        self.bar.set('phase', 'validation')

    def metric_calc(self, target, model):
        img, bboxes = target

        img = img.to(self.c.device, torch.float)
        bboxes = bboxes.to(self.c.device, torch.float)

        prediction = model(img)

        metrics = self.metrics(prediction, bboxes)
        for key in metrics.keys():
            item = metrics[key].item()
            self.bar.set(key,item)
            self.c.logger.add_scalar('Validation/batch/metric/{}'.format(key), item)

        return sum(metrics.values())

    @torch.no_grad()
    def validate(self, epoch):

        self.bar.set('epoch', epoch)

        self.c.model.eval()

        metric_accum = 0
        for i_step, target in enumerate(self.bar):
            metric = self.metric_calc(target, self.c.model)

            metric_accum += metric.item()
            self.c.logger.add_scalar('Validation/batch/metric/sum', metric.item())

            ave_metric = metric_accum / (i_step + 1)
            self.c.logger.add_scalar('Validation/batch/metric/average', ave_metric)

            self.bar.set('ave_metric', ave_metric)
            self.bar.update()

        return metric_accum / len(self.bar)
