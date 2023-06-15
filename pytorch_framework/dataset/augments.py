from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
from helpers.viewer import Viewer
from helpers.logger import Logger
from helpers.constants import IMG_AUG_PATH
import cv2
class Augments:

    @staticmethod
    def get_all_augmentations():
        return [
            A.Resize(448, 448),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.ToGray(p=0.5),
            A.Blur(always_apply=False, p=0.5, blur_limit=(3, 16)),
            A.PixelDropout(always_apply=False, p=1.0, dropout_prob=0.26, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None)
        ]

    @staticmethod
    def train():
        transform = Augments.get_all_augmentations()
        transform.append(ToTensorV2(p=1.0))
        return A.Compose(
            transform,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    @staticmethod
    def validation():
        return A.Compose([
            A.Resize(width=448, height=448),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    @staticmethod
    def predict():
        return A.Compose([
                    A.Resize(width=448, height=448),
                    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )



if __name__ == "__main__":
    aug = Augments()
    logger = Logger('Augments')

    image = cv2.imread(IMG_AUG_PATH)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = aug.get_all_augmentations()
    viewer = Viewer()
    for j in range(len(transform)):
        result = []
        image = viewer.add_title(img, 'Original img')
        result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))
        for i in range(len(transform) - 1):
            transform[i+1].p = 1
            t2 = transform[i+1]
            tgr = A.Compose([
                transform[0], t2
            ])
            transformed = tgr(image=img)
            image = transformed["image"]
            image = viewer.add_title(image, str(transform[i + 1].__class__.__name__))
            result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        result = torch.cat(result)
        print('Done inerr {} !'.format(j))
        logger.add_grid_images('my_test', result)

    print('Done All !')
