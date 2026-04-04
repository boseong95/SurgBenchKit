# VTRB-Suturing Recognition (few-shot bbox detection)
# Sends 3 annotated example images + 1 test image to the model

from torch.utils.data import Dataset
import os
import json
import numpy as np
import cv2
import av


class VTRBSuturingRecognition(Dataset):
    """VTRB-Suturing few-shot bounding box detection.

    Each sample provides:
    - 3 few-shot example images with bbox annotations (in-context learning)
    - 1 test image to detect objects in

    Classes: grippers, target_tissue, wound_gap, bites, needle
    """

    SPLITS = {
        'train': ['20260328_hyunjun', '20260328_seunghoon', '20260329_taeyoon_v2', '20260327_minho'],
        'val': ['20260327_hyunjun'],
        'test': ['20260329_taeyoon', '20260330_kx'],
    }

    CLASSES = ["grippers", "target_tissue", "wound_gap", "bites", "needle"]

    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.vqa_path = config['data_config'].get('vqa_path',
            '/home/ubuntu/datasets/vlm/VTRB-Suturing-VQA/recognition/recognition.json')
        self.bbox_path = config['data_config'].get('bbox_path',
            '/home/ubuntu/datasets/vlm/VTRB-Suturing-VQA/bbox_annotations.json')
        self.frames_dir = config['data_config'].get('frames_dir',
            '/home/ubuntu/datasets/vlm/VTRB-Suturing-VQA/frames')
        self.split = split
        self.folders = self.SPLITS[split]

        self.max_samples = config.get('max_samples', None)

        # Load few-shot bbox annotations
        self.few_shot_examples = self._load_few_shot()
        self.labels = self.load_labels()

        if self.max_samples is not None and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def _load_few_shot(self):
        """Load few-shot bbox annotations and build example list."""
        if not os.path.exists(self.bbox_path):
            return []
        with open(self.bbox_path) as f:
            annotations = json.load(f)

        examples = []
        for frame_key, boxes in sorted(annotations.items()):
            if not boxes:
                continue
            image_path = os.path.join(self.frames_dir, frame_key)
            if not os.path.exists(image_path):
                continue

            by_class = {}
            for box in boxes:
                cls = box["class"]
                if cls not in by_class:
                    by_class[cls] = []
                by_class[cls].append({
                    "bbox": [box["x_pct"], box["y_pct"],
                             round(box["x_pct"] + box["w_pct"], 1),
                             round(box["y_pct"] + box["h_pct"], 1)],
                })
            examples.append({
                "image_path": image_path,
                "objects": by_class,
            })
        return examples

    def load_labels(self):
        with open(self.vqa_path) as f:
            data = json.load(f)

        labels = []
        for sample in data['samples']:
            folder = sample.get('folder', '')
            if folder not in self.folders:
                continue

            image_path = sample['image_path']
            # Ensure frame exists (extract if needed)
            if not os.path.exists(image_path):
                self._extract_frame(sample)

            meta = {
                'id': sample['id'],
                'folder': folder,
                'episode_idx': sample.get('episode_idx', 0),
                'anchor_frame': sample.get('anchor_frame', 0),
                'subtask': sample.get('subtask', ''),
            }

            label = {}  # No GT for recognition

            labels.append((image_path, label, meta))
        return labels

    def _extract_frame(self, sample):
        """Extract a single frame from video if not cached."""
        folder = sample['folder']
        ep_idx = sample['episode_idx']
        anchor = sample['anchor_frame']
        image_path = sample['image_path']

        vid_path = os.path.join(
            self.data_dir, 'annotated', folder,
            'videos', 'chunk-000', 'laparoscope_camera',
            f'episode_{ep_idx:06d}.mp4'
        )
        if not os.path.exists(vid_path):
            return

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        try:
            container = av.open(vid_path)
            for i, frame in enumerate(container.decode(video=0)):
                if i == anchor:
                    img = frame.to_ndarray(format='bgr24')
                    cv2.imwrite(image_path, img)
                    break
            container.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path, label, meta = self.labels[idx]

        # Build message paths: few-shot images + test image
        # The inference function will interleave these with prompt text
        paths = []
        for ex in self.few_shot_examples:
            paths.append(ex['image_path'])
        paths.append(image_path)

        frame = {
            'paths': paths,
            'path': image_path,
            'meta': {
                **meta,
                'few_shot_examples': self.few_shot_examples,
                'test_image_path': image_path,
            },
        }
        return (frame, label)


class VTRBSuturingPhasePredictEasy(Dataset):
    """VTRB Suturing phase prediction — 5-choice MCQ from single frame."""

    def __init__(self, config, split):
        self.data_dir = config['data_config']['data_dir']
        self.vqa_path = config['data_config']['vqa_path']
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
            frame_path = os.path.join(self.data_dir, s['frame_path'])
            meta = {
                'id': s['id'],
                'folder': s['folder'],
                'session': s['session'],
                'episode_index': s['episode_index'],
                'frame_index': s['frame_index'],
                'choices': s['choices'],
                'answer': s['answer'],
                'answer_text': s['answer_text'],
            }
            labels.append((frame_path, s['answer'], meta))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_path, label, meta = self.labels[idx]
        frame = {'path': frame_path, 'meta': meta}
        return (frame, label)


class VTRBSuturingPhasePlanning(Dataset):
    """VTRB Suturing phase planning — predict current_phase + next_phase from multi-frame.

    Extracts frames from video on-the-fly and caches them.
    """

    def __init__(self, config, split):
        self.data_dir = config['data_config']['data_dir']
        self.vtrb_root = config['data_config'].get('vtrb_root',
            '/home/ubuntu/datasets/vlm/VTRB-Suturing')
        self.vqa_path = config['data_config']['vqa_path']
        self.n_context = config.get('n_context', 10)
        self.stride = config.get('stride', 5)
        self.max_samples = config.get('max_samples', None)
        self.cache_dir = os.path.join(self.data_dir, 'frame_cache_planning')
        self.split = split
        self.labels = self._load()
        if self.max_samples and self.max_samples < len(self.labels):
            indices = np.linspace(0, len(self.labels) - 1, self.max_samples, dtype=int)
            self.labels = [self.labels[i] for i in indices]

    def _extract_frames(self, video_path, frame_indices, cache_prefix):
        """Extract frames from video, caching to disk."""
        paths = []
        to_extract = {}
        for fid in frame_indices:
            cache_path = os.path.join(self.cache_dir, cache_prefix, f'{fid:06d}.png')
            paths.append(cache_path)
            if not os.path.exists(cache_path):
                to_extract[fid] = cache_path

        if to_extract:
            os.makedirs(os.path.join(self.cache_dir, cache_prefix), exist_ok=True)
            full_vid = os.path.join(self.vtrb_root, video_path)
            if os.path.exists(full_vid):
                try:
                    container = av.open(full_vid)
                    max_idx = max(to_extract.keys())
                    for i, frame in enumerate(container.decode(video=0)):
                        if i in to_extract:
                            img = frame.to_ndarray(format='bgr24')
                            cv2.imwrite(to_extract[i], img)
                        if i > max_idx:
                            break
                    container.close()
                except Exception:
                    pass
        return paths

    def _load(self):
        with open(self.vqa_path) as f:
            data = json.load(f)

        labels = []
        for s in data['samples']:
            anchor = s['anchor_frame']
            frame_indices = [max(0, anchor - i * self.stride) for i in range(self.n_context - 1, -1, -1)]
            cache_prefix = f"{s['session']}_ep{s['episode_index']:03d}"

            frame_paths = self._extract_frames(s['video_path'], frame_indices, cache_prefix)

            meta = {
                'id': s['id'],
                'session': s['session'],
                'episode_index': s['episode_index'],
                'anchor_frame': anchor,
                'phase': s['phase'],
                'phase_idx': s['phase_idx'],
                'phase_start_frame': s['phase_start_frame'],
                'phase_end_frame': s['phase_end_frame'],
                'next_phase': s['next_phase'],
                'choices': s['choices'],
                'phase_choice': s.get('phase_choice', ''),
            }

            labels.append((frame_paths, s['next_phase'], meta))
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
