from data_handlers.dataset import VOCDataset
from data_handlers.augmentations import get_transforms_for_predict
from typing import Dict
'''
    def split(self, validation_split):

        data_size = self.data_handlers.data_handlers.shape[0]
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices
'''

def get_datasets() -> Dict[str, VOCDataset]:

    transform = get_transforms_for_predict()

    path = "data/train.csv"
    IMG_DIR = "data/data/images"
    LABEL_DIR = "data/data/labels"
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