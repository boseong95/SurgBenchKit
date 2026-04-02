# CholecT50 dataset loader for phase recognition and tool presence detection
# Based on CholecT50 JSON annotation format
# Reference: https://arxiv.org/abs/2204.05235

from torch.utils.data import Dataset
import os
import json
import numpy as np


class CholecT50PhaseRecognition(Dataset):
    """CholecT50 phase recognition — 7 phases, single-label per frame."""

    # Official split from https://arxiv.org/pdf/2204.05235
    SPLITS = {
        'train': ['VID01', 'VID02', 'VID04', 'VID05', 'VID13', 'VID15', 'VID18', 'VID22', 'VID23',
                  'VID25', 'VID26', 'VID27', 'VID31', 'VID35', 'VID36', 'VID40', 'VID43', 'VID47',
                  'VID48', 'VID49', 'VID52', 'VID56', 'VID57', 'VID60', 'VID62', 'VID65', 'VID66',
                  'VID68', 'VID70', 'VID75', 'VID79', 'VID92', 'VID96', 'VID103', 'VID110', 'VID111'],
        'val': ['VID08', 'VID29', 'VID50', 'VID78'],
        'test': ['VID06', 'VID10', 'VID14', 'VID32', 'VID42', 'VID51', 'VID73', 'VID74', 'VID80'],
    }

    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(self.data_dir, 'videos')
        self.label_dir = os.path.join(self.data_dir, 'labels')
        self.folders = self.SPLITS[split]
        self.split = split
        self.max_samples = config.get('max_samples', None)
        self.labels = self.load_labels()
        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def load_labels(self):
        fps_rate = 25  # sample every 25th frame (1 fps from 25fps video)
        labels = []
        for video_name in self.folders:
            label_file = os.path.join(self.label_dir, f'{video_name}.json')
            if not os.path.exists(label_file):
                continue
            with open(label_file) as f:
                data = json.load(f)
            annotations = data['annotations']
            for frame_idx_str, anns in annotations.items():
                frame_idx = int(frame_idx_str)
                if frame_idx % fps_rate != 0:
                    continue
                # Phase is the last element of the first annotation
                phase = anns[0][-1]
                frame_path = os.path.join(self.image_dir, video_name, f'{frame_idx:06d}.png')
                labels.append((frame_path, phase))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_path, frame_label = self.labels[idx]
        frame = {'path': frame_path}
        return (frame, frame_label)


class CholecT50ToolRecognition(Dataset):
    """CholecT50 tool presence detection — 6 tools, multi-label per frame."""

    SPLITS = CholecT50PhaseRecognition.SPLITS
    TOOLS = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator']

    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(self.data_dir, 'videos')
        self.label_dir = os.path.join(self.data_dir, 'labels')
        self.folders = self.SPLITS[split]
        self.split = split
        self.max_samples = config.get('max_samples', None)
        self.labels = self.load_labels()
        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def load_labels(self):
        fps_rate = 25
        labels = []
        for video_name in self.folders:
            label_file = os.path.join(self.label_dir, f'{video_name}.json')
            if not os.path.exists(label_file):
                continue
            with open(label_file) as f:
                data = json.load(f)
            tool_categories = data['categories']['instrument']
            annotations = data['annotations']
            for frame_idx_str, anns in annotations.items():
                frame_idx = int(frame_idx_str)
                if frame_idx % fps_rate != 0:
                    continue
                # Collect active tools from all triplets in this frame
                tool_vector = np.zeros(len(self.TOOLS), dtype=np.float32)
                for ann in anns:
                    instrument_id = ann[1]  # instrument ID from annotation
                    if 0 <= instrument_id < len(self.TOOLS):
                        tool_vector[instrument_id] = 1
                frame_path = os.path.join(self.image_dir, video_name, f'{frame_idx:06d}.png')
                labels.append((frame_path, tool_vector))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_path, frame_label = self.labels[idx]
        frame = {'path': frame_path}
        return (frame, frame_label)
