import torch
import fiftyone.utils.coco as fouc
from PIL import Image


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            fiftyone_dataset,
            transforms=None,
            gt_field="ground_truth",
            classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        detections = sample[self.gt_field].detections
        img_has_person = 0
        for det in detections:
            if det.label != 'person':
                continue

            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_detection(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox

            boxes.append([x / metadata['width'],
                          y / metadata['height'],
                          (x + w) / metadata['width'],
                          (y + h) / metadata['height']])

            img_has_person = 1

        target = {}
        target['img_width'] = metadata['width']
        target['img_height'] = metadata['height']

        box = [0,0,0,0]
        if img_has_person:
            if len(boxes) > 1:
                box = sorted(boxes)[0]
            else:
                box = boxes[0]
        target["box"] = torch.as_tensor(box, dtype=torch.float32)

        target["img_has_person"] = img_has_person
        target["img_path"] = img_path

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes