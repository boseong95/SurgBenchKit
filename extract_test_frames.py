"""
Extract test set frames for HeiChole evaluation.
Saves only every 25th frame (1 fps from 25 fps video) named by original frame index,
matching what HeiCholeDataloader expects: frame_{original_idx:05d}.png
"""
import cv2
import os

VIDEO_DIR = "/Users/boseong/dataset/HeiChole/Videos/Full"
OUTPUT_DIR = "/Users/boseong/dataset/HeiChole/extracted_frames"
TEST_IDS = [1, 4, 13, 16, 22]
FPS_RATE = 25  # select every 25th frame = 1 fps from 25 fps source

for video_id in TEST_IDS:
    video_name = f"Hei-Chole{video_id}"
    video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    out_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = total // FPS_RATE + 1
    print(f"\n{video_name}: {total} total frames → ~{expected} to extract")

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FPS_RATE == 0:
            fname = os.path.join(out_dir, f"frame_{frame_idx:05d}.png")
            if not os.path.exists(fname):  # skip already extracted
                cv2.imwrite(fname, frame)
                saved += 1
            else:
                saved += 1  # count existing
        frame_idx += 1
        if frame_idx % 5000 == 0:
            print(f"  {frame_idx}/{total} frames scanned, {saved} saved...", flush=True)

    cap.release()
    print(f"  Done: {saved} frames saved to {out_dir}")

print("\nExtraction complete.")
