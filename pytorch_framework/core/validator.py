import torch
from losses.Yolov1 import YoloLoss
from core.train_config import TrainConfig

class Validator:

    def __init__(self, config: TrainConfig):
        self.config = config
        self.metrics = YoloLoss()

    def metric_calc(self, target, model):
        img, bboxes = target

        img = img.to(self.config.device, torch.cuda.FloatTensor)
        bboxes = bboxes.to(self.config.device, torch.cuda.FloatTensor)

        prediction = model(img)

        metrics = self.metrics(prediction, bboxes)
        for key in metrics.keys():
            self.config.logger.add_scalar('Metrics_{}/{}'.format('train', key), metrics[key].item())

        return sum(metrics.values())

    def validate(self, epoch):
        C = self.config
        C.model.eval()

        torch.set_grad_enabled(False)

        metric_accum = 0
        step_count = len(C.val_loader)
        for i_step, target in enumerate(C.val_loader):
            loss_value = self.metric_calc(target, C.model)

            metric_accum += loss_value.item()
            C.logger.add_scalar('Metric_sum_{}/batch'.format('validation'), loss_value.item())
            # report_metrics['loss'][phase].append(loss_value.item())
            print('Epoch {}/{}. Phase {} Step {}/{} Metric {}'.format(epoch, epoch - 1, 'validation',
                                                                    i_step, step_count,
                                                                    loss_value.item()))