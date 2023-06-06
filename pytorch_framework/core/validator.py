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
            self.c.logger.add_scalar('Metrics_{}/{}'.format('train', key), item)

        return sum(metrics.values())

    def validate(self, epoch):

        self.bar.set('epoch', epoch)
        self.c.model.eval()

        torch.set_grad_enabled(False)

        metric_accum = 0
        step_count = len(self.bar)
        for i_step, target in enumerate(self.bar):
            loss_value = self.metric_calc(target, self.c.model)

            metric_accum += loss_value.item()
            self.c.logger.add_scalar('Metric_sum_{}/batch'.format('validation'), loss_value.item())
            # report_metrics['loss'][phase].append(loss_value.item())
            #print('Epoch {}/{}. Phase {} Step {}/{} Metric {}'.format(epoch, epoch - 1, 'validation',
             #                                                         i_step, step_count,
              #                                                        loss_value.item()))
            self.bar.update()
