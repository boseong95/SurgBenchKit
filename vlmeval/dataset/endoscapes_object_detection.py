from torch.utils.data import Dataset
import os
import numpy as np
import json
from collections import defaultdict


class EndoscapesObjectDetection(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(self.data_dir, split)
        self.split = split
        self.category = config.get('category', 'all')  # 'all', 'tool', or 'anatomy'
        self.max_samples = config.get('max_samples', None)

        # Load COCO annotations
        ann_path = os.path.join(self.image_dir, 'annotation_coco.json')
        with open(ann_path) as f:
            self.coco_data = json.load(f)

        # Category mappings
        self.category_ids_to_name = {c['id']: c['name'] for c in self.coco_data['categories']}
        self.category_name_to_id = {v: k for k, v in self.category_ids_to_name.items()}

        # Image id to filename mapping
        self.file_names_to_id = {}
        self.id_to_image = {}
        for img in self.coco_data['images']:
            self.file_names_to_id[img['file_name']] = img['id']
            self.id_to_image[img['id']] = img

        self.labels = self.load_labels()
        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def load_labels(self):
        # Group annotations by image
        img_annotations = defaultdict(lambda: defaultdict(list))
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            cat_name = self.category_ids_to_name[ann['category_id']]
            img_annotations[img_id][cat_name].append(ann['bbox'])  # [x, y, w, h]

        labels = []
        for img in self.coco_data['images']:
            img_id = img['id']
            frame_path = os.path.join(self.image_dir, img['file_name'])
            label = dict(img_annotations.get(img_id, {}))
            label['im_size_wh'] = [img['width'], img['height']]
            labels.append((frame_path, label))

        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_path, frame_label = self.labels[idx]
        frame = {'path': frame_path}
        return frame, frame_label
