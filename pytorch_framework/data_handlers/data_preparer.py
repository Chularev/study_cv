from data_handlers.dataset import RoadDataset
from typing import Tuple, Dict
from albumentations.pytorch import ToTensorV2
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

    transform = A.Compose([
        A.Resize(width=128, height=128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


    path = '/mnt/heap/imges/road/training/image_2'
    torch_dataset = RoadDataset(path, transform)

    print('train dataset = ' + str(len(torch_dataset)))

    return {
        'train': torch_dataset
    }