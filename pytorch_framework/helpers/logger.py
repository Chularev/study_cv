from torch.utils.tensorboard import SummaryWriter
from helpers.constants import LOG_DIR
import json


class Logger(object):
    def __init__(self, log_dir):
        log_dir = LOG_DIR + log_dir
        self.logger = SummaryWriter(log_dir=log_dir)
        self.values = {}

    def get_step(self, name):
        if name not in self.values.keys():
            self.values[name] = -1

        self.values[name] += 1
        return self.values[name]

    def add_scalar(self, name, value):
        step = self.get_step(name)
        self.logger.add_scalar(name, value, step)

    def add_scalars(self, name, value):
        step = self.get_step(name)
        self.logger.add_scalars(name, value, step)

    def add_image(self, name, img):
        step = self.get_step(name)
        self.logger.add_image(name, img, step)

    def add_grid_images(self, name, imgs):
        step = self.get_step(name)
        self.logger.add_images(name, imgs, step)

    def load(self, path):
        with open(path, 'r') as f:
            self.values = json.load(f)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.values, f)
