from PIL import Image
import torch
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
        plt.show()

    def add_box(self, img, target):
        x_top_left = math.ceil(target['box'][0] * target['img_width'])
        y_top_left = math.ceil(target['box'][1] * target['img_height'])

        x_bottom_right = math.ceil(target['box'][2] * target['img_width'])
        y_bottom_right = math.ceil(target['box'][3] * target['img_height'])

        img = self.convert_from_image_to_cv2(img)
        img = cv2.rectangle(img, (x_top_left, y_top_left), (x_bottom_right,y_bottom_right),
                              (255, 0, 0), 4)
        return self.convert_from_cv2_to_image(img)

    def print_prediction(self, img, target, prediction):
        self.print_img(img, target)
        flag = torch.round(torch.sigmoid(prediction[0])) == 1
        label = "Yes" if flag else 'No'
        if flag > 0.5:
            x_top_left = math.ceil(prediction[1] * target['img_width'])
            y_top_left = math.ceil(prediction[2] * target['img_height'])

            x_bottom_right = math.ceil(prediction[3] * target['img_width'])
            y_bottom_right = math.ceil(prediction[4] * target['img_height'])

            img = self.convert_from_image_to_cv2(img)
            img = cv2.rectangle(img, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                (0, 0, 255), 4)
            img = self.convert_from_cv2_to_image(img)
        plt.title(label)
        plt.imshow(img)
        plt.show()

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

    @staticmethod
    def visualize_metric_history(extended_model):
        plt.plot(extended_model.train_metric_history, label='train')
        plt.plot(extended_model.val_metric_history, label='val')
        plt.title('model metric')
        plt.ylabel('metric')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()

    @staticmethod
    def visualize_loss_history(extended_model):
        plt.plot(extended_model.train_loss_history, label='train')
        plt.plot(extended_model.val_loss_history, label='val')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()
