"""
Main file for training Yolo model on Pascal VOC dataset

"""
import os

import torch
from dataset.augments import Augments as Aug
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.Yolov1 import Yolov1
from dataset.vocdataset import VOCDataset
from  core.trainer import Trainer
from helpers.utils import (
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    load_checkpoint,
)
from losses.Yolov1 import YoloLoss
from core.train_parameters import TrainParameters
from core.train_starter import create_context_from_params
from core.train_param_enums import LoadStrategy, SaveStrategy
from helpers.constants import CHECKPOINT_FOLDER, IMG_DIR, LABEL_DIR
from dataset.dataset_helperr import DatasetHelper

LOAD_MODEL_FILE = CHECKPOINT_FOLDER + "overfit.pth.tar"

p = TrainParameters()
datasets = DatasetHelper.get_datasets()
c = create_context_from_params(p, datasets)

def validation_fn(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device=c.device,
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device,  torch.float)
        labels = labels.to(device,  torch.float)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    mean_avg_prec = mean_average_precision(
        all_pred_boxes, all_true_boxes, iou_threshold=0.5, box_format="midpoint"
    )
    model.train()

    return mean_avg_prec


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(1)



    trainer = Trainer(c)
    c.model.to(c.device)

    for epoch in range(p.epoch_num):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        mean_avg_prec = validation_fn(
            c.val_loader, c.model, iou_threshold=0.5, threshold=0.4
        )


        print(f"Train mAP: {mean_avg_prec}")

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        trainer.train(epoch)


if __name__ == "__main__":
    main()
