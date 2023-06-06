import torch
from losses.Yolov1 import YoloLoss
from core.train_context import _TrainContext


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
            self.c.logger.add_scalar('Losses_{}/{}'.format('train', key), losses[key].item())

        return sum(losses.values())

    def train(self, epoch):
        self.c.model.train()

        torch.set_grad_enabled(True)

        if epoch > 0 and self.c.scheduler is not None:
            self.c.scheduler.step()

        loss_accum = 0
        step_count = len(self.c.train_loader)
        for i_step, target in enumerate(self.c.train_loader):
            loss_value = self.loss_calc(target, self.c.model)

            loss_value.backward()
            self.c.optimizer.step()
            self.c.optimizer.zero_grad()

            loss_accum += loss_value.item()
            self.c.logger.add_scalar('Loss_sum_{}/batch'.format('train'), loss_value.item())

            ave_loss = loss_accum / step_count
            self.c.logger.add_scalar('Loss_sum_train/epoch', ave_loss)
