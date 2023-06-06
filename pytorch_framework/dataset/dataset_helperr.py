from dataset.dataset import VOCDataset
from dataset.augmentations import get_transforms_for_train
from typing import Dict
from helpers.constants import get_project_root_dir



def get_datasets() -> Dict[str, VOCDataset]:

    transform = get_transforms_for_train()

    ROOT_DIR = get_project_root_dir()

    path = ROOT_DIR + "/data/train.csv"
    IMG_DIR = ROOT_DIR + "/data/data/images"
    LABEL_DIR = ROOT_DIR + "/data/data/labels"
    torch_dataset = VOCDataset(
        path,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )


    print('train dataset = ' + str(len(torch_dataset)))

    return {
        'train': torch_dataset
    }