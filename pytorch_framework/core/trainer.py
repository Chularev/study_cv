import torch
from losses.Yolov1 import YoloLoss
from core.train_config import _TrainConfig


class Trainer:

    def __init__(self, config: _TrainConfig):
        self.config = config
        self.losses = YoloLoss()

    def loss_calc(self, target, model):
        img, bboxes = target

        img = img.to(self.config.device, torch.float)
        bboxes = bboxes.to(self.config.device, torch.float)

        prediction = model(img)

        losses = self.losses(prediction, bboxes)
        for key in losses.keys():
            self.config.logger.add_scalar('Losses_{}/{}'.format('train', key), losses[key].item())

        return sum(losses.values())

    def train(self, epoch):
        c = self.config
        c.model.train()

        torch.set_grad_enabled(True)

        if epoch > 0 and c.scheduler is not None:
            c.scheduler.step()

        loss_accum = 0
        step_count = len(c.train_loader)
        for i_step, target in enumerate(c.train_loader):
            loss_value = self.loss_calc(target, c.model)

            loss_value.backward()
            c.optimizer.step()
            c.optimizer.zero_grad()

            loss_accum += loss_value.item()
            c.logger.add_scalar('Loss_sum_{}/batch'.format('train'), loss_value.item())

            ave_loss = loss_accum / step_count
            c.logger.add_scalar('Loss_sum_train/epoch', ave_loss)
