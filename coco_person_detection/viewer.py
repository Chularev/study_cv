from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


class Viewer:
    def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(self, img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def print_img(self, img, target):
        label = "Yes" if target['img_has_person'] == 1 else 'No'
        if target['img_has_person'] == 1:
            img = self.add_box(img, target)
        plt.title(label)
        plt.imshow(img)

    def add_box(self, img, target):
        x_top_left = math.ceil(target['box'][0] * target['width'])
        y_top_left = math.ceil(target['box'][1] * target['height'])

        x_bottom_right = math.ceil(target['box'][2] * target['width'])
        y_bottom_right = math.ceil(target['box'][3] * target['height'])

        img = self.convert_from_image_to_cv2(img)
        img = cv2.rectangle(img, (x_top_left, y_top_left), (x_bottom_right,y_bottom_right),
                              (255, 0, 0), 2)
        return self.convert_from_cv2_to_image(img)


    def visualize_samples(dataset, indices, title=None, count=10):
        # visualize random 10 samples
        plt.figure(figsize=(count * 3, 3))
        display_indices = indices[:count]
        if title:
            plt.suptitle("%s %s/%s\n\n" % (title, len(display_indices), len(indices)))

        for i, index in enumerate(display_indices):
            x, y, _ = dataset[index]
            plt.subplot(1, count, i + 1)
            plt.title("Label: %s" % y)
            plt.imshow(x)
            plt.grid(False)
            plt.axis('off')