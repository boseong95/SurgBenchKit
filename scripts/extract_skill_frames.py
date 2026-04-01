"""
Extract frames from skill assessment videos at 0.2 fps (1 frame every 5 seconds).
Saves frames as frame-{idx}.jpg in a directory named {video}_images/.

Usage:
    python scripts/extract_skill_frames.py                # extract all
    python scripts/extract_skill_frames.py --dataset heichole
    python scripts/extract_skill_frames.py --dataset jigsaws
"""

import argparse
import os
import cv2
from pathlib import Path


FPS_RATE = 0.2  # 1 frame every 5 seconds


def extract_frames(video_path, output_dir, target_fps=FPS_RATE):
    """Extract frames from a video at the target fps rate."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0 or total_frames <= 0:
        print(f"  ERROR: Invalid video {video_path} (fps={src_fps}, frames={total_frames})")
        cap.release()
        return 0

    # Sample every N frames to achieve target_fps
    frame_interval = max(1, int(src_fps / target_fps))

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame-{count}.jpg")
            cv2.imwrite(out_path, frame)
            count += 1
        frame_idx += 1

    cap.release()
    return count


def extract_heichole():
    """Extract frames from HeiChole skill videos (calot + dissection)."""
    skill_dir = "/home/ubuntu/datasets/vlm/HeiCo/Videos/Skill"
    output_base = "/home/ubuntu/datasets/vlm/HeiCo/skill_frames"

    videos = sorted([f for f in os.listdir(skill_dir) if f.endswith(".mp4")])
    print(f"HeiChole: {len(videos)} skill videos")

    for video_file in videos:
        video_path = os.path.join(skill_dir, video_file)
        output_dir = os.path.join(output_base, video_file.replace(".mp4", "_images"))
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"  SKIP {video_file} ({len(os.listdir(output_dir))} frames exist)")
            continue
        n = extract_frames(video_path, output_dir)
        print(f"  {video_file} → {n} frames")


def extract_jigsaws():
    """Extract frames from JIGSAWS videos."""
    base_dir = "/home/ubuntu/datasets/vlm/JIGSAWS"
    output_base = "/home/ubuntu/datasets/vlm/JIGSAWS"

    for category in ["Knot_Tying", "Needle_Passing", "Suturing"]:
        video_dir = os.path.join(base_dir, category, "video")
        if not os.path.exists(video_dir):
            print(f"  SKIP {category}: no video dir")
            continue

        # Only capture1 (main camera view)
        videos = sorted([f for f in os.listdir(video_dir) if "capture1" in f])
        print(f"JIGSAWS {category}: {len(videos)} videos")

        for video_file in videos:
            video_path = os.path.join(video_dir, video_file)
            stem = video_file.replace(".avi", "").replace(".mp4", "")
            output_dir = os.path.join(output_base, category, f"{stem}_images")
            if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
                print(f"  SKIP {video_file} ({len(os.listdir(output_dir))} frames exist)")
                continue
            n = extract_frames(video_path, output_dir)
            print(f"  {video_file} → {n} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["heichole", "jigsaws", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("heichole", "all"):
        extract_heichole()
    if args.dataset in ("jigsaws", "all"):
        extract_jigsaws()

    print("Done!")


if __name__ == "__main__":
    main()
