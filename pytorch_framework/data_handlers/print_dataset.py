from data_handlers.data_preparer import get_datasets
import torch
import cv2
from viewer import Viewer
from logger import Logger

if __name__ == "__main__":
    datasets = get_datasets()
    train = datasets['train']

    viewer = Viewer()
    logger = Logger('TensorboardImages')

    for i in range(10):
        result = []
        target = train[i]

        path = target['path']
        img_orig = cv2.imread(path)

        mask = target['mask']

        img_orig = viewer.convert_from_image_to_cv2(img_orig)
        image = viewer.add_title(img_orig, 'Original img')
        result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        mask = viewer.mask_image(img_orig, mask)
        mask = viewer.convert_from_image_to_cv2(mask)
        image = viewer.add_title(mask, 'Mask ')
        result.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        result = torch.cat(result)
        print('Done inerr {} !'.format(i))
        logger.add_grid_images('my_test', result)

    print('Done All !')
