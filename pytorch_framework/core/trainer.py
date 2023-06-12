import torch
from losses.Yolov1 import YoloLoss
from core.train_context import _TrainContext
from helpers.bar import Bar


class Trainer:

    def __init__(self, context: _TrainContext):
        self.c = context
        self.losses = YoloLoss()

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

    def bar_init(self, epoch):
        self.bar = Bar(self.c.val_loader)
        self.bar.set('phase', 'train')
        self.bar.set('epoch', epoch)

    @torch.enable_grad()
    def train(self, epoch):

        self.bar_init(epoch)

        self.c.model.train()

        if epoch > 0 and self.c.scheduler is not None:
            self.c.scheduler.step()

        loss_accum = 0
        for i_step, target in enumerate(self.bar):
            loss_value = self.loss_calc(target, self.c.model)

            loss_value.backward()
            self.c.optimizer.step()
            self.c.optimizer.zero_grad()

            loss_accum += loss_value.item()
            self.c.logger.add_scalar('Train/batch/loss/sum', loss_value.item())

            ave_loss = loss_accum / (i_step + 1)
            self.c.logger.add_scalar('Train/batch/loss/average', ave_loss)

            self.bar.set('ave_loss', ave_loss)
            self.bar.update()

        return loss_accum / len(self.bar)
