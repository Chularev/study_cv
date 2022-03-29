from data_handlers.dataset import RoadDataset
from typing import Tuple, Dict
import collections
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import torchvision.transforms as T
import albumentations as A
import cv2
'''
    def split(self, validation_split):

        data_size = self.data_handlers.data_handlers.shape[0]
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices
'''

def get_datasets() -> Dict[str, RoadDataset]:
    '''
    img_size = (254, 254)

    p = 0.5
    a_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3)
    ], bbox_params=A.BboxParams(format='albumentations'))
    '''

    path = '/mnt/heap/imges/road/training/image_2'
    torch_dataset = RoadDataset(path)

    print('train dataset = ' + str(len(torch_dataset)))

    return {
        'train': torch_dataset
    }