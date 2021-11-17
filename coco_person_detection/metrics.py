import torch


class Metrics:
    @staticmethod
    def iou(model, loader):
        return 0

    @staticmethod
    def compute_accuracy(model, loader):
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
            x_gpu = x.to(self.device)
            y_gpu = y.to(self.device)

            prediction = model(x_gpu)

            _, indices = torch.max(prediction, 1)

            total_samples += y.shape[0]
            correct_samples += torch.sum(indices == y_gpu)

        return float(correct_samples) / total_samples


