# CholecT50 Status Reasoning dataset
# Supports multi-frame input with configurable n_context and stride
# Answer formats: A1 (binary), A2 (progress %), A3 (remaining frames)

from torch.utils.data import Dataset
import os
import json
import numpy as np


class CholecT50StatusReasoning(Dataset):
    """CholecT50 phase status reasoning with multi-frame input.

    Each sample provides an anchor frame + phase boundaries.
    Frame selection (n_context, stride) is applied at load time.
    """

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
        self.vqa_path = config['data_config'].get('vqa_path',
            '/home/ubuntu/datasets/vlm/CholecT50-VQA/q3/q3.json')
        self.split = split
        self.folders = self.SPLITS[split]

        # Frame selection config (can be overridden per experiment)
        self.n_context = config.get('n_context', 5)
        self.stride = config.get('stride', 1)
        self.answer_format = config.get('answer_format', 'A1')  # A1, A2, A3, A4

        # Load Q4 data for A4 triplet GT
        self.q4_path = config['data_config'].get('q4_path',
            '/home/ubuntu/datasets/vlm/CholecT50-VQA/q4/q4.json')
        self.q4_by_anchor = self._load_q4()

        self.max_samples = config.get('max_samples', None)
        self.labels = self.load_labels()
        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def _load_q4(self):
        """Load Q4 next-action data, indexed by (video_id, anchor_frame)."""
        if not os.path.exists(self.q4_path):
            return {}
        with open(self.q4_path) as f:
            data = json.load(f)
        by_anchor = {}
        for s in data['samples']:
            by_anchor[(s['video_id'], s['anchor_frame'])] = s['answer']  # list of triplet dicts
        return by_anchor

    def _get_current_triplets(self, vid, anchor):
        """Get current triplets from CholecT50 annotations."""
        label_path = os.path.join(self.data_dir, 'labels', f'{vid}.json')
        if not os.path.exists(label_path):
            return []
        with open(label_path) as f:
            data = json.load(f)
        ann = data['annotations'].get(str(anchor), [])
        cats = data['categories']
        triplets = []
        for t in ann:
            tid = t[0]
            if tid < 0:
                continue
            tname = cats['triplet'].get(str(tid), '')
            if 'null_verb' in tname or 'null_target' in tname:
                continue
            parts = tname.split(',')
            triplets.append({'instrument': parts[0], 'verb': parts[1], 'target': parts[2]})
        return triplets

    def load_labels(self):
        with open(self.vqa_path) as f:
            data = json.load(f)

        labels = []
        for sample in data['samples']:
            vid = sample['video_id']
            if vid not in self.folders:
                continue

            anchor = sample['anchor_frame']
            phase_start = sample['phase_start_frame']
            phase_end = sample['phase_end_frame']
            phase = sample['phase']
            phase_id = sample['phase_id']

            # Compute frame paths based on n_context and stride
            frame_ids = []
            for i in range(self.n_context - 1, -1, -1):
                fid = anchor - i * self.stride
                fid = max(0, fid)
                frame_ids.append(fid)

            frame_paths = [
                os.path.join(self.image_dir, vid, f'{fid:06d}.png')
                for fid in frame_ids
            ]

            # Compute label based on answer format
            phase_duration = phase_end - phase_start
            if phase_duration == 0:
                progress = 100
                remaining = 0
            else:
                progress = round((anchor - phase_start) / phase_duration * 100)
                remaining = round((phase_end - anchor) / self.stride)

            status = sample['answer']  # "in-progress" or "finished"

            # Triplet data for A4
            PHASE_NAMES = {
                0: "Preparation", 1: "Calot Triangle Dissection",
                2: "Clipping & Cutting", 3: "Gallbladder Dissection",
                4: "Gallbladder Packaging", 5: "Cleaning & Coagulation",
                6: "Gallbladder Retraction",
            }
            current_triplet = self._get_current_triplets(vid, anchor)
            next_triplet = self.q4_by_anchor.get((vid, anchor), [])
            next_phase = PHASE_NAMES.get(phase_id + 1, phase) if status == 'finished' else phase

            if self.answer_format == 'A1':
                label = status
            elif self.answer_format == 'A2':
                label = progress
            elif self.answer_format == 'A3':
                label = remaining
            elif self.answer_format == 'A4':
                label = {
                    'current': {'phase': phase, 'triplet': current_triplet},
                    'next': {'phase': next_phase, 'triplet': next_triplet},
                }
            else:
                label = status

            labels.append((
                frame_paths,
                label,
                {
                    'id': sample['id'],
                    'video_id': vid,
                    'anchor_frame': anchor,
                    'phase': phase,
                    'phase_id': phase_id,
                    'phase_start_frame': phase_start,
                    'phase_end_frame': phase_end,
                    'status': status,
                    'progress': progress,
                    'remaining': remaining,
                    'current_triplet': current_triplet,
                    'next_triplet': next_triplet,
                    'next_phase': next_phase,
                }
            ))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_paths, label, meta = self.labels[idx]
        # For single frame, return dict with 'path' (compatible with existing pipeline)
        # For multi-frame, return dict with 'paths' list
        if len(frame_paths) == 1:
            frame = {'path': frame_paths[0], 'meta': meta}
        else:
            frame = {'paths': frame_paths, 'path': frame_paths[-1], 'meta': meta}
        return (frame, label)

    def get_few_shot_examples(self, sample_meta, n_examples=3):
        """Get few-shot examples from other videos, same phase, different progress."""
        target_phase = sample_meta['phase_id']
        target_vid = sample_meta['video_id']

        candidates = [
            (paths, label, meta) for paths, label, meta in self.labels
            if meta['video_id'] != target_vid and meta['phase_id'] == target_phase
        ]

        if len(candidates) <= n_examples:
            return candidates

        # Pick examples spread across progress range
        candidates.sort(key=lambda x: x[2]['progress'])
        indices = np.linspace(0, len(candidates) - 1, n_examples, dtype=int)
        return [candidates[i] for i in indices]
