import torch
from pycparser.c_ast import ID

from metrics_iou import Iou
from torchmetrics.classification import Accuracy


class MyMetric:
    def __init__(self):
        self.accuracy = Accuracy()
        self.iou = Iou()

    def step(self, predict, g_truth):
        return 0

    def compute(self):
        return {
            'accuracy': self.accuracy.compute(),
            'iou': self.iou.compute()
        }