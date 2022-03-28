import albumentations as A
import torch
from data_handlers.data_preparer import get_datasets
from torchvision.transforms import ToTensor
import torchvision
from logger import Logger
import cv2

def get_augmentations():
    a_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3)
    ], bbox_params=A.BboxParams(format='albumentations'))
    return a_transform


if __name__ == "__main__":
    #datasets = get_datasets()
    transform = get_augmentations()
    logger = Logger('TensorBoard')

    #data = datasets['train']

    #image, target = data[0]

    #image = cv2.imread(target['img_path'])
    image = cv2.imread('/mnt/heap/imges/coco/validation/data/000000161861.jpg')

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    for j in range(4):
        result = []
        for i in range(10):
            transformed = transform(image=img, bboxes=[[0.7, 0.7, 0.8, 0.8, 'kjk']])
            image = transformed["image"]
            result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        result = torch.cat(result)
        print('Done inerr {} !'.format(j))
        logger.add_grid_images('my_test', result)

    print('Done All !')
