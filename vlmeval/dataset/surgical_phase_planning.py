# Generic surgical phase planning + progress prediction dataset
# Works for CholecT50, Cholec80, HeiChole

from torch.utils.data import Dataset
import os
import json
import numpy as np


PHASE_NAMES = {
    0: "Preparation", 1: "Calot Triangle Dissection", 2: "Clipping & Cutting",
    3: "Gallbladder Dissection", 4: "Gallbladder Packaging",
    5: "Cleaning & Coagulation", 6: "Gallbladder Retraction",
}


class SurgicalPhasePlanning(Dataset):
    """Phase planning dataset — predict current_phase + next_phase from frames."""

    def __init__(self, config, split):
        self.data_dir = config['data_config']['data_dir']
        self.vqa_path = config['data_config']['vqa_path']
        self.frame_template = config['data_config'].get('frame_template', None)
        self.n_context = config.get('n_context', 16)
        self.stride = config.get('stride', 1)
        self.max_samples = config.get('max_samples', None)
        self.split = split
        self.labels = self._load()
        if self.max_samples and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def _load(self):
        with open(self.vqa_path) as f:
            data = json.load(f)

        labels = []
        for s in data['samples']:
            anchor = s['anchor_frame']
            frame_ids = [max(0, anchor - i * self.stride) for i in range(self.n_context - 1, -1, -1)]

            if self.frame_template:
                frame_paths = [
                    self.frame_template.format(data_dir=self.data_dir, video_id=s['video_id'], frame=fid)
                    for fid in frame_ids
                ]
            else:
                frame_paths = [
                    os.path.join(self.data_dir, 'videos', s['video_id'], f'{fid:06d}.png')
                    for fid in frame_ids
                ]

            meta = {
                'id': s['id'],
                'video_id': s['video_id'],
                'anchor_frame': anchor,
                'phase': s['phase'],
                'phase_id': s['phase_id'],
                'phase_start_frame': s['phase_start_frame'],
                'phase_end_frame': s['phase_end_frame'],
                'next_phase': s.get('next_phase', s['phase']),
                'status': s.get('answer', 'in-progress'),
                'progress_pct': round((anchor - s['phase_start_frame']) / max(s['phase_end_frame'] - s['phase_start_frame'], 1) * 100),
            }

            label = {
                'current_phase': s['phase'],
                'next_phase': meta['next_phase'],
            }

            labels.append((frame_paths, label, meta))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_paths, label, meta = self.labels[idx]
        if len(frame_paths) == 1:
            frame = {'path': frame_paths[0], 'meta': meta}
        else:
            frame = {'paths': frame_paths, 'path': frame_paths[-1], 'meta': meta}
        return (frame, label)

    def get_few_shot_examples(self, sample_meta, n_examples=3):
        target_phase = sample_meta['phase_id']
        target_vid = sample_meta['video_id']
        candidates = [
            (p, l, m) for p, l, m in self.labels
            if m['video_id'] != target_vid and m['phase_id'] == target_phase
        ]
        if len(candidates) <= n_examples:
            return candidates
        candidates.sort(key=lambda x: x[2]['progress_pct'])
        indices = np.linspace(0, len(candidates) - 1, n_examples, dtype=int)
        return [candidates[i] for i in indices]


class SurgicalPhaseProgress(Dataset):
    """Phase progress prediction — classify as early/middle/late."""

    def __init__(self, config, split):
        self.data_dir = config['data_config']['data_dir']
        self.vqa_path = config['data_config']['vqa_path']
        self.frame_template = config['data_config'].get('frame_template', None)
        self.n_context = config.get('n_context', 16)
        self.stride = config.get('stride', 1)
        self.max_samples = config.get('max_samples', None)
        self.split = split
        self.labels = self._load()
        if self.max_samples and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def _load(self):
        with open(self.vqa_path) as f:
            data = json.load(f)

        labels = []
        for s in data['samples']:
            anchor = s['anchor_frame']
            frame_ids = [max(0, anchor - i * self.stride) for i in range(self.n_context - 1, -1, -1)]

            if self.frame_template:
                frame_paths = [
                    self.frame_template.format(data_dir=self.data_dir, video_id=s['video_id'], frame=fid)
                    for fid in frame_ids
                ]
            else:
                frame_paths = [
                    os.path.join(self.data_dir, 'videos', s['video_id'], f'{fid:06d}.png')
                    for fid in frame_ids
                ]

            meta = {
                'id': s['id'],
                'video_id': s['video_id'],
                'anchor_frame': anchor,
                'phase': s['phase'],
                'phase_id': s['phase_id'],
                'phase_start_frame': s['phase_start_frame'],
                'phase_end_frame': s['phase_end_frame'],
                'progress_class': s['progress_class'],
                'progress_pct': s['progress_pct'],
            }

            labels.append((frame_paths, s['progress_class'], meta))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_paths, label, meta = self.labels[idx]
        if len(frame_paths) == 1:
            frame = {'path': frame_paths[0], 'meta': meta}
        else:
            frame = {'paths': frame_paths, 'path': frame_paths[-1], 'meta': meta}
        return (frame, label)

    def get_few_shot_examples(self, sample_meta, n_examples=3):
        target_phase = sample_meta['phase_id']
        target_vid = sample_meta['video_id']
        candidates = [
            (p, l, m) for p, l, m in self.labels
            if m['video_id'] != target_vid and m['phase_id'] == target_phase
        ]
        if len(candidates) <= n_examples:
            return candidates
        candidates.sort(key=lambda x: x[2]['progress_pct'])
        indices = np.linspace(0, len(candidates) - 1, n_examples, dtype=int)
        return [candidates[i] for i in indices]
