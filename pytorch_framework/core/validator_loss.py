import torch
from core.train_context import _TrainContext
from losses.Yolov1 import YoloLoss
from helpers.bar import Bar


class ValidatorLoss:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.losses = YoloLoss()

        self.bar = Bar(self.c.val_loader)
        self.bar.set('phase', 'validator_loss')

    def loss_calc(self, target, model):
        img, bboxes = target

        img = img.to(self.c.device, torch.float)
        bboxes = bboxes.to(self.c.device, torch.float)

        prediction = model(img)

        losses = self.losses(prediction, bboxes)
        for key in losses.keys():
            item = losses[key].item()
            self.bar.set(key, item)
            self.c.logger.add_scalar('Train/batch/loss/{}'.format(key), item)

        return sum(losses.values())

    @torch.no_grad()
    def validate(self, epoch):

        self.bar.set('epoch', epoch)
        self.bar.start()

        self.c.model.eval()

        loss_accum = 0
        for i_step, target in enumerate(self.bar):
            loss_value = self.loss_calc(target, self.c.model)

            loss_accum += loss_value.item()
            self.c.logger.add_scalar('Train/batch/loss/sum', loss_value.item())

            ave_loss = loss_accum / (i_step + 1)
            self.c.logger.add_scalar('Train/batch/loss/average', ave_loss)

            self.bar.set('ave_loss', ave_loss)
            self.bar.update()

        self.bar.stop()

        return loss_accum / len(self.c.val_loader)
