from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from metrics_iou import Iou
import io
from torchvision.transforms import ToTensor


class Viewer:
    def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(self, img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def add_title(self, cv_img, title):
        img = self.convert_from_cv2_to_image(cv_img)
        plt.figure()

        plt.title(title)
        plt.imshow(img)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()

        img = Image.open(buf)
        return self.convert_from_image_to_cv2(img)

    def mask_image(self, img, mask):
        newImg = img.copy()
        newImg[:, :, 0] = img[:, :, 0] * mask[:, :]
        newImg[:, :, 1] = img[:, :, 1] * mask[:, :]
        newImg[:, :, 2] = img[:, :, 2] * mask[:, :]
        return newImg


