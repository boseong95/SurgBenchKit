"""
Extract frames from surgical video datasets for VLM evaluation.

Supported datasets and their frame extraction logic:
┌─────────────┬──────────┬───────────────────┬────────────────────────────────────────┐
│ Dataset      │ Source   │ Frame indices      │ Output format                          │
├─────────────┼──────────┼───────────────────┼────────────────────────────────────────┤
│ HeiChole     │ 25fps    │ every 25th (1fps)  │ {video}/frame_{idx:05d}.png            │
│ Cholec80     │ 25fps    │ every 25th (1fps)  │ frames_25fps/{split}/{video}/{idx}.jpg │
│ Endoscapes   │ N/A      │ pre-extracted      │ already in {split}/{vid}_{frame}.jpg   │
└─────────────┴──────────┴───────────────────┴────────────────────────────────────────┘

Sampling rates from the paper (Table S2, arxiv:2504.02799):
  - Cholec80:  zero-shot SR=5, few-shot SR=15  (applied by dataset class, NOT here)
  - HeiChole:  zero-shot SR=25, few-shot SR=375 (applied by dataset class, NOT here)
  - Endoscapes: zero-shot SR=1, few-shot SR=2   (applied by dataset class, NOT here)

This script extracts at the *base* annotation rate (every 25th frame = 1fps).
The dataset classes in vlmeval/dataset/ handle additional subsampling at test time.

For Cholec80, tool annotations exist every 25 frames (0, 25, 50, ...),
and phase annotations exist every frame (0, 1, 2, ...).
We extract every 25th frame to match tool annotation granularity.
The phase dataset class further subsamples by 5x (test fps_rate=125).
The tool dataset class also subsamples by 5x (test fps_rate=125).
Both yield ~10,000 test frames total across 40 test videos, consistent with the paper.

Cholec80 split (standard):
  - Train: videos 01-40
  - Test:  videos 41-80
  (No official val split; the dataset class accepts train/val/test directory names)

Usage:
  python extract_test_frames_parallel.py --dataset heichole
  python extract_test_frames_parallel.py --dataset cholec80
  python extract_test_frames_parallel.py --dataset cholec80 --split train
"""
import argparse
import cv2
import os
from multiprocessing import Pool


# ── Dataset configurations ──────────────────────────────────────────────────

DATASETS = {
    "heichole": {
        "video_dir": "/home/ubuntu/datasets/vlm/HeiCo/Videos/Full",
        "output_dir": "/home/ubuntu/datasets/vlm/HeiCo/extracted_frames",
        "video_ids": {
            "test": [1, 4, 13, 16, 22],
            "train": [11, 12, 24],  # per paper's few-shot examples
        },
        "video_name_fmt": "Hei-Chole{id}",
        "video_file_fmt": "{name}.mp4",
        "frame_file_fmt": "frame_{idx:05d}.png",  # named by original frame index
        "source_fps": 25,
        "extract_every": 25,  # 1 fps from 25 fps source
    },
    "cholec80": {
        "video_dir": "/home/ubuntu/datasets/vlm/Cholec80/videos",
        "output_dir": "/home/ubuntu/datasets/vlm/Cholec80/frames_25fps",
        "video_ids": {
            "test": list(range(41, 81)),   # videos 41-80
            "train": list(range(1, 41)),   # videos 01-40
        },
        "video_name_fmt": "video{id:02d}",
        "video_file_fmt": "{name}.mp4",
        "frame_file_fmt": "{idx}.jpg",  # matches annotation frame numbers: 0, 25, 50, ...
        "source_fps": 25,
        "extract_every": 25,  # 1 fps from 25 fps source; matches tool annotation granularity
    },
}


# ── Extraction worker ───────────────────────────────────────────────────────

def extract_video(args):
    video_path, out_dir, extract_every, frame_file_fmt, video_label = args

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"[ERROR] Cannot open {video_path}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = total // extract_every + 1
    print(f"{video_label}: {total} total frames -> ~{expected} to extract", flush=True)

    saved, skipped = 0, 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % extract_every == 0:
            fname = os.path.join(out_dir, frame_file_fmt.format(idx=frame_idx))
            if not os.path.exists(fname):
                cv2.imwrite(fname, frame)
                saved += 1
            else:
                skipped += 1
        frame_idx += 1
        if frame_idx % 10000 == 0:
            print(f"  [{video_label}] {frame_idx}/{total} scanned, "
                  f"{saved} new + {skipped} existing", flush=True)

    cap.release()
    return (f"{video_label}: done — {saved} new + {skipped} existing "
            f"= {saved + skipped} total frames in {out_dir}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from surgical video datasets for VLM evaluation."
    )
    parser.add_argument(
        "--dataset", required=True, choices=DATASETS.keys(),
        help="Which dataset to extract frames from."
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "test"],
        help="Which split to extract (default: test)."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: one per video)."
    )
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    video_ids = cfg["video_ids"][args.split]

    # Build task list
    tasks = []
    for vid in video_ids:
        video_name = cfg["video_name_fmt"].format(id=vid)
        video_path = os.path.join(cfg["video_dir"], cfg["video_file_fmt"].format(name=video_name))
        out_dir = os.path.join(cfg["output_dir"], args.split, video_name)
        tasks.append((
            video_path,
            out_dir,
            cfg["extract_every"],
            cfg["frame_file_fmt"],
            video_name,
        ))

    n_workers = args.workers or len(tasks)
    print(f"Extracting {args.dataset} ({args.split}): "
          f"{len(tasks)} videos with {n_workers} workers")
    print(f"  Source: {cfg['video_dir']}")
    print(f"  Output: {cfg['output_dir']}/{args.split}/")
    print(f"  Rate: every {cfg['extract_every']}th frame "
          f"({cfg['source_fps'] / cfg['extract_every']:.1f} fps)")
    print()

    with Pool(n_workers) as pool:
        results = pool.map(extract_video, tasks)

    print()
    for r in results:
        print(r)
    print(f"\nDone. Frames saved to: {cfg['output_dir']}/{args.split}/")


if __name__ == "__main__":
    main()
