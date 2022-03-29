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
        mask = mask.numpy()
        newImg = img * mask
        return newImg

    def binary(self, img, mask):
        newImg = img.copy()
        newImg[:, :, 0] = 1 * mask[:, :]
        newImg[:, :, 1] = 1 * mask[:, :]
        newImg[:, :, 2] = 1 * mask[:, :]
        return newImg

    def create_output(self, target, prediction = None):
        result = []

        path = target['path']
        img_orig = cv2.imread(path)

        mask = target['mask']

        SIZE = (mask.shape[0], mask.shape[1])
        img_orig = cv2.resize(img_orig, SIZE)

        img_orig = self.convert_from_image_to_cv2(img_orig)
        image = self.add_title(img_orig, 'Original img')
        result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        mask = self.mask_image(img_orig, mask)
        mask = self.convert_from_image_to_cv2(mask)
        image = self.add_title(mask, 'Mask ')
        result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        if prediction != None:
            prediction = torch.round(prediction)
            mask = self.mask_image(img_orig, prediction)
            mask = self.convert_from_image_to_cv2(mask)
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(SIZE).convert('RGB'))
            image = self.add_title(mask, 'Mask prediction')
            result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        return torch.cat(result)


