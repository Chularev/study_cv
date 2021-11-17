import torch
from metrics_iou import Iou


class Metrics:
    @staticmethod
    def iou(model, loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        iou = 0

        for i_step, (img, target) in enumerate(loader):
            x = img.to(device)
            predictions = model(x)
            for i in range(len(predictions)):
                prediction = predictions[i]
                if prediction[0] < 0.5:
                    if not target['img_has_person']:
                        iou += 1
                    continue

                if target['img_has_person']:
                    tmp = Iou.iou(prediction[1:], target['box'])
                    if tmp > 0.5:
                        iou += 1

        return iou / len(loader)

    @staticmethod
    def compute_accuracy(model, loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        Computes accuracy on the dataset wrapped in a loader

        Returns: accuracy as a float value between 0 and 1
        """
        model.eval()  # Evaluation mode
        # TODO: Copy implementation from previous assignment
        # Don't forget to move the data to device before running it through the model!

        total_samples = 0
        correct_samples = 0
        for i_step, (x, y) in enumerate(loader):
            x_gpu = x.to(device)
            y_gpu = y.to(device)

            prediction = model(x_gpu)

            _, indices = torch.max(prediction, 1)

            total_samples += y.shape[0]
            correct_samples += torch.sum(indices == y_gpu)

        return float(correct_samples) / total_samples


