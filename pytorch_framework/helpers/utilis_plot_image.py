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
        return list(np.random.choice(range(256), size=3) / 256)


    def plot_image(self, image, boxes):
        r1 = self.plot_image_new(image, boxes)
        r2 = self.plot_image_old(image, boxes)
        assert (r1 == r2)
        return r1

    @my_profile
    def plot_image_new(self, image, boxes):

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

    @my_profile
    def plot_image_old(self, image, boxes):
        """Plots predicted bounding boxes on the image"""
        image = self.viewer.convert_from_cv2_to_image(image)
        im = np.array(image)
        height, width, _ = im.shape

        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)

        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height

        # Create a Rectangle potch

        for box in boxes:
            c_class = int(box[0])
            box = box[2:]
            assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
            upper_left_x = box[0] - box[2] / 2
            upper_left_y = box[1] - box[3] / 2
            rect = patches.Rectangle(
                (upper_left_x * width, upper_left_y * height),
                box[2] * width,
                box[3] * height,
                linewidth=1,
                edgecolor=self.colors[c_class],
                facecolor="none",
            )

            ax.add_artist(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() - 16

            ax.annotate(self.classes[c_class], (cx, cy), color='r', weight='bold',
                        fontsize=16, ha='center', va='center')

            # Add the patch to the Axes
            # ax.add_patch(rect)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()

        return Image.open(buf)