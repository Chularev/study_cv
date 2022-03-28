import torch
import fiftyone.utils.coco as fouc
import cv2
from PIL import Image
import os


class RoadDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path,
            transforms=None,
            a_transforms=None
    ):
        self.transforms = transforms
        self.a_transforms = a_transforms

        self.img_paths = os.listdir(path)
        self.path = path

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_path = self.path + '/' + img_path

        img = cv2.imread(img_path)

        img_path_target = img_path.replace("image_2", "gt_image_2")
        img_path_target = img_path_target.replace("um_", "um_road_")
        img_path_target = img_path_target.replace("umm_", "umm_road_")
        img_path_target = img_path_target.replace("uu_", "uu_road_")

        img_target = cv2.imread(img_path_target)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.a_transforms:
            img = self.a_transforms(image=img)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_target

    def __len__(self):
        return len(self.img_paths)