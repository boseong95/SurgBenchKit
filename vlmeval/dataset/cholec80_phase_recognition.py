from torch.utils.data import Dataset
import os
import numpy as np


class Cholec80PhaseRecognition(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.image_dir = os.path.join(config['data_config']['data_dir'], 'frames_25fps', split)
        self.data_dir = config['data_config']['data_dir']

        self.map = {phase: idx for idx, phase in enumerate(config['label_names'])}
        self.split = split
        self.few_shot = True if config['shots'] != 'zero' else False
        self.max_samples = config.get('max_samples', None)
        self.labels = self.load_labels()
        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]


    def load_labels(self):
        labels = []
        fps_rate = 25
        if self.split == 'test':
            fps_rate = fps_rate * 5
            if self.few_shot:
                fps_rate = fps_rate * 15
        for video_name in os.listdir(self.image_dir):
            video_labels_path = os.path.join(self.data_dir, 'phase_annotations', f'{video_name}-phase.txt')
            with open(video_labels_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if line[0] != 'Frame':
                        if int(line[0]) % fps_rate != 0:
                            continue  # only sample 1/5 frame per sec
                        frame_label = self.map[line[1]]
                        frame_name = f'{line[0]}.jpg'
                        frame_path = os.path.join(self.image_dir, video_name, frame_name)
                        labels.append((frame_path, frame_label))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_name, frame_label = self.labels[idx]

        frame = {'path': frame_name}
        return (
            frame,
            frame_label
        )