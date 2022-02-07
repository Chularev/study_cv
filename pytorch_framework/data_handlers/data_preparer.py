from data_handlers.dataset import FiftyOneTorchDataset
from typing import Tuple, Dict
import collections
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import torchvision.transforms as T

'''
    def split(self, validation_split):

        data_size = self.data_handlers.data_handlers.shape[0]
        split = int(np.floor(validation_split * data_size))
        indices = list(range(data_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices
'''

def get_datasets() -> Dict[str, FiftyOneTorchDataset]:
    fo_dataset = foz.load_zoo_dataset("coco-2017", split="validation",
                                      dataset_dir='/mnt/heap')
    fo_dataset.compute_metadata()

    person_list = ['person', "car", "truck", "bus", 'boat']
    person_view = fo_dataset.filter_labels("ground_truth",
                                           F("label").is_in(person_list)).match(
        F("ground_truth.detections").length() == 1)
    print('person_view len = ' + str(len(person_view)))

    # split the dataset in train and test set
    train_view = person_view.take(len(person_view) * 0.75, seed=51)
    test_view = person_view.exclude([s.id for s in train_view])

    img_size = (256, 256)
    train_transforms = T.Compose([T.Resize(img_size),
                                  T.ToTensor()])
    test_transforms = T.Compose([T.Resize(img_size),
                                 T.ToTensor()])

    torch_dataset = FiftyOneTorchDataset(train_view, train_transforms,
                                         classes=person_list)
    torch_dataset_test = FiftyOneTorchDataset(test_view, test_transforms,
                                              classes=person_list)
    print('train dataset = ' + str(len(torch_dataset)))
    print('test dataset = ' + str(len(torch_dataset_test)))

    return {
        'train' : torch_dataset,
        'test': torch_dataset_test
    }