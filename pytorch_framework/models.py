from torchvision.models import resnet18
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=5)

    def forward(self, x):
        x = self.model(x)
        return {
            'class': x[:, 0],
            'bbox': x[:, 1:]
        }
