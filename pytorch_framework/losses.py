from torch import Module
from torch import Tensor
import torch


class MyLoss(Module):
    def __init__(self) -> None:
        super(MyLoss, self).__init__()

        self.class_loss = torch.nn.BCEWithLogitsLoss()
        self.bbox_loss = torch.nn.SmoothL1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return 90;