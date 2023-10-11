import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

from PIL import Image

from analysis.my_profile import my_profile
from helpers.viewer import Viewer


class Utilis_plot_image:
    def __init__(self):
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.colors = []
        for i in range(len(self.classes)):
            self.colors.append(self.random_color())

        self.viewer = Viewer()

    def random_color(self):
        color = list(np.random.choice(range(255), size=3))
        return (int(color[0]), int(color[1]), int(color[2]))

    def plot_image(self, image, boxes):

        height, width, _ = image.shape

        for box in boxes:
            c_class = int(box[0])
            box = box[2:]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            x1, y1 = int((box[0] - box[2] / 2) * width), int((box[1] - box[3] / 2) * height)
            w, h = box[2] * width, box[3] * height,
            x2, y2 = int(x1 + w), int(y1 + h)
            cv2.rectangle(image, (x1,y1), (x2,y2), color=self.colors[c_class], thickness=1)

        return self.viewer.convert_from_cv2_to_image(image)