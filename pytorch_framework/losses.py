from torch import Tensor
import torch


class MyLoss:
    def __init__(self) -> None:
        super(MyLoss, self).__init__()

        self.class_loss = torch.nn.BCELoss()
        self.bbox_loss = torch.nn.SmoothL1Loss()

    def calc(self, prediction: Tensor, gpu_img_has_person: Tensor, gpu_box: Tensor) -> Tensor:

        loss_value = self.class_loss(prediction['class'], gpu_img_has_person.type(torch.cuda.FloatTensor))

        indexes_with_label = (gpu_img_has_person == 1).nonzero(as_tuple=True)
        if len(indexes_with_label) > 0:
            return loss_value + self.bbox_loss(prediction['bbox'][indexes_with_label], gpu_box[indexes_with_label])
        return loss_value