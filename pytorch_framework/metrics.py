import torch
from metrics_iou import Iou


class Metrics:
    @staticmethod
    def iou(model, loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        iou = 0
        total_samples = 0

        for i_step, (img, targets) in enumerate(loader):
            total_samples += img.shape[0]

            gpu_img = img.type(torch.cuda.FloatTensor)
            gpu_img = gpu_img.to(device)
            predictions = model(gpu_img)

            gpu_img_has_person = targets['img_has_person'].type(torch.cuda.FloatTensor)
            gpu_img_has_person = gpu_img_has_person.to(device)

            gpu_box = targets['box'].type(torch.cuda.FloatTensor)
            gpu_box = gpu_box.to(device)

            for i in range(len(predictions)):
                prediction = predictions[i]
                if torch.round(torch.sigmoid(prediction[0])) == 0:
                    if not gpu_img_has_person[i]:
                        iou += 1
                    continue

                if gpu_img_has_person[i]:
                    tmp = Iou.iou(prediction[1:], gpu_box[i])
                    if tmp > 0.5:
                        iou += 1

        return iou / total_samples

    @staticmethod
    def compute_accuracy(model, loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """
        Computes accuracy on the dataset wrapped in a loader

        Returns: accuracy as a float value between 0 and 1
        """
        model.eval()  # Evaluation mode
        # TODO: Copy implementation from previous assignment
        # Don't forget to move the data_handlers to device before running it through the model!

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


