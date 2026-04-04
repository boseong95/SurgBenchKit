# CholecT50 Phase + Triplet Planning dataset
# Predicts: current phase, current triplet, next phase, next triplet
# Uses multi-frame input with configurable n_context and stride

from torch.utils.data import Dataset
import os
import json
import numpy as np


class CholecT50PhaseTripletPlanning(Dataset):
    """CholecT50 phase and triplet planning with multi-frame input.

    Each sample provides an anchor frame with ground truth for:
    - phase: current surgical phase
    - current_triplet: (instrument, verb, target) actions happening now
    - next_phase: the phase that follows
    - next_triplet: the next actions the surgeon should perform
    """

    SPLITS = {
        'train': ['VID01', 'VID02', 'VID04', 'VID05', 'VID13', 'VID15', 'VID18', 'VID22', 'VID23',
                  'VID25', 'VID26', 'VID27', 'VID31', 'VID35', 'VID36', 'VID40', 'VID43', 'VID47',
                  'VID48', 'VID49', 'VID52', 'VID56', 'VID57', 'VID60', 'VID62', 'VID65', 'VID66',
                  'VID68', 'VID70', 'VID75', 'VID79', 'VID92', 'VID96', 'VID103', 'VID110', 'VID111'],
        'val': ['VID08', 'VID29', 'VID50', 'VID78'],
        'test': ['VID06', 'VID10', 'VID14', 'VID32', 'VID42', 'VID51', 'VID73', 'VID74', 'VID80'],
    }

    PHASE_NAMES = {
        0: "Preparation", 1: "Calot Triangle Dissection",
        2: "Clipping & Cutting", 3: "Gallbladder Dissection",
        4: "Gallbladder Packaging", 5: "Cleaning & Coagulation",
        6: "Gallbladder Retraction",
    }

    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(self.data_dir, 'videos')
        self.vqa_path = config['data_config'].get('vqa_path',
            '/home/ubuntu/datasets/vlm/CholecT50-VQA/phase_triplet_planning/phase_triplet_planning.json')
        self.split = split
        self.folders = self.SPLITS[split]

        self.n_context = config.get('n_context', 10)
        self.stride = config.get('stride', 5)

        # Load Q4 data for next-triplet GT (used when generating from q3 base)
        self.q4_path = config['data_config'].get('q4_path',
            '/home/ubuntu/datasets/vlm/CholecT50-VQA/q4/q4.json')
        self.q4_by_anchor = self._load_q4()

        self.max_samples = config.get('max_samples', None)
        self.labels = self.load_labels()
        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def _load_q4(self):
        if not os.path.exists(self.q4_path):
            return {}
        with open(self.q4_path) as f:
            data = json.load(f)
        by_anchor = {}
        for s in data['samples']:
            by_anchor[(s['video_id'], s['anchor_frame'])] = s['answer']
        return by_anchor

    def _get_current_triplets(self, vid, anchor):
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

    def _find_next_phase(self, vid, phase_id, phase_end):
        label_path = os.path.join(self.data_dir, 'labels', f'{vid}.json')
        if not os.path.exists(label_path):
            return self.PHASE_NAMES.get(phase_id, "")
        with open(label_path) as f:
            data = json.load(f)
        ann = data['annotations']
        for fid in sorted(ann.keys(), key=int):
            if int(fid) > phase_end:
                next_p = ann[fid][0][-1]
                if next_p != phase_id:
                    return self.PHASE_NAMES.get(next_p, f"Phase {next_p}")
        return "End of Procedure"

    def load_labels(self):
        with open(self.vqa_path) as f:
            data = json.load(f)

        # Check if this is the curated JSON (has answer.current_triplet)
        is_curated = (data['samples'] and
                      isinstance(data['samples'][0].get('answer'), dict) and
                      'current_triplet' in data['samples'][0]['answer'])

        labels = []
        for sample in data['samples']:
            vid = sample['video_id']
            if vid not in self.folders:
                continue

            anchor = sample['anchor_frame']
            phase_start = sample.get('phase_start_frame', 0)
            phase_end = sample.get('phase_end_frame', 0)
            phase = sample.get('phase', '')
            if not phase and isinstance(sample.get('answer'), dict):
                phase = sample['answer'].get('phase', '')
            phase_id = sample.get('phase_id', 0)

            # Compute frame paths
            if 'frame_indices' in sample:
                frame_ids = sample['frame_indices']
            else:
                frame_ids = []
                for i in range(self.n_context - 1, -1, -1):
                    fid = anchor - i * self.stride
                    fid = max(0, fid)
                    frame_ids.append(fid)

            frame_paths = [
                os.path.join(self.image_dir, vid, f'{fid:06d}.png')
                for fid in frame_ids
            ]

            if is_curated:
                answer = sample['answer']
                current_triplet = answer['current_triplet']
                next_phase = answer['next_phase']
                next_triplet = answer['next_triplet']
            else:
                # Build from q3 + q4 data
                current_triplet = self._get_current_triplets(vid, anchor)
                next_triplet = self.q4_by_anchor.get((vid, anchor), [])
                status = sample.get('answer', 'in-progress')
                if isinstance(status, str) and status == 'finished':
                    next_phase = self._find_next_phase(vid, phase_id, phase_end)
                else:
                    next_phase = phase

            label = {
                'phase': phase,
                'current_triplet': current_triplet,
                'next_phase': next_phase,
                'next_triplet': next_triplet,
            }

            meta = {
                'id': sample.get('id', f'{vid}_{anchor}'),
                'video_id': vid,
                'anchor_frame': anchor,
                'phase': phase,
                'phase_id': phase_id,
                'phase_start_frame': phase_start,
                'phase_end_frame': phase_end,
                'current_triplet': current_triplet,
                'next_phase': next_phase,
                'next_triplet': next_triplet,
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
