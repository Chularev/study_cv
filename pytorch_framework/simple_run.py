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
    device="cuda:0",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

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


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(c.device), y.to(c.device)
        out = model(x)
        loss = loss_fn(out, y)
        loss = loss['YoloV1']
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(1)



    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(c.device)
    optimizer = optim.Adam(
        model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
    )
    loss_fn = YoloLoss()

   # if p.load_strategy == LoadStrategy.MODEL_OPTIMIZER:
    #    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        "data/train.csv",
        transform=Aug.train(),
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=Aug.validation(), img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=p.t_loader_batch_size,
        num_workers=p.t_loader_num_workers,
        pin_memory=p.t_loader_pin_memory,
        shuffle=p.t_loader_shuffle,
        drop_last=p.t_loader_drop_last,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=p.v_loader_batch_size,
        num_workers=p.v_loader_num_workers,
        pin_memory=p.v_loader_pin_memory,
        shuffle=p.v_loader_shuffle,
        drop_last=p.v_loader_drop_last,
    )

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
            train_loader, model, iou_threshold=0.5, threshold=0.4
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

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
