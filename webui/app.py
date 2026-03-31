import re
import os
import json
import random
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
PROMPTS_PY = REPO_ROOT / "vlmeval" / "prompts.py"
DATASET_ROOT = Path("/Users/boseong/dataset")

# ── Label definitions (from task YAML configs) ───────────────────────────────
HEICHOLE_LABELS = {
    "phase": {
        "type": "single",
        "names": ["Preparation", "Calot Triangle Dissection", "Clipping & Cutting",
                  "Gallbladder Dissection", "Gallbladder Packaging",
                  "Cleaning & Coagulation", "Gallbladder Retraction"],
    },
    "tool": {
        "type": "multi",
        "names": ["Grasper", "Bipolar Forceps", "Hook", "Scissors",
                  "Clipper", "Irrigator", "Specimen Bag"],
        # Each instrument spans 3 CSV cols (left/right/in-field); take col-wise OR
        "col_groups": [(1,4), (4,7), (7,10), (10,13), (13,16), (16,19), (19,22)],
    },
    "action": {
        "type": "multi",
        "names": ["Grasp", "Hold", "Cut", "Clip"],
        "col_groups": [(1,2), (2,3), (3,4), (4,5)],
    },
}

TASK_DATASET_ANNOTATION = {
    # Frame-based
    "heichole_phase_recognition":           ("HeiChole",    "phase"),
    "heichole_tool_recognition":            ("HeiChole",    "tool"),
    "heichole_action_recognition":          ("HeiChole",    "action"),
    "endoscapes_object_detection":          ("endoscapes",  "bbox"),
    # Video-based
    "heichole_skill_assessment":            ("HeiChole",    "skill"),
    "jigsaws_skill_assessment":             ("JIGSAWS",     "skill"),
    "jigsaws_gesture_classification":       ("JIGSAWS",     "gesture"),
    "autolaparo_maneuver_classification":   ("autolaparo",  "maneuver"),
}

# Classification of tasks: "frame" or "video"
TASK_GRANULARITY = {
    "phase_recognition":        "frame",
    "tool_recognition":         "frame",
    "action_recognition":       "frame",
    "object_detection":         "frame",
    "cvs_assessment":           "frame",
    "triplet_recognition":      "frame",
    "error_detection":          "frame",
    "error_recognition":        "frame",
    "anatomy_presence":         "frame",
    "skill_assessment":         "video",
    "gesture_classification":   "video",
    "maneuver_classification":  "video",
}

# ── Video task label definitions ─────────────────────────────────────────────
HEICHOLE_SKILL_CRITERIA = [
    "Tissue Handling", "Time & Motion", "Instrument Handling",
    "Flow of Operation", "Overall Performance",
]
JIGSAWS_SKILL_CRITERIA = [
    "Respect for Tissue", "Suture Needle Handling", "Time & Motion",
    "Flow of Operation", "Overall Performance", "Quality of Final Product",
]
JIGSAWS_CATEGORIES = ["Knot_Tying", "Needle_Passing", "Suturing"]
JIGSAWS_EXPERIENCE = {"N": "Novice", "I": "Intermediate", "E": "Expert"}

AUTOLAPARO_MANEUVER_LABELS = ["Static", "Up", "Down", "Left", "Right", "Zoom-in", "Zoom-out"]
AUTOLAPARO_MANEUVER_COLORS = {
    "Static": "#6b7280", "Up": "#3b82f6", "Down": "#f59e0b",
    "Left": "#10b981", "Right": "#ef4444", "Zoom-in": "#8b5cf6", "Zoom-out": "#ec4899",
}
GESTURE_COLORS = {
    "G1":"#6b7280","G2":"#3b82f6","G3":"#f59e0b","G4":"#10b981","G5":"#ef4444",
    "G6":"#8b5cf6","G8":"#ec4899","G9":"#14b8a6","G10":"#f97316","G11":"#84cc16",
    "G12":"#06b6d4","G13":"#a855f7","G14":"#e11d48","G15":"#0ea5e9",
}
SCORE_COLORS = ["#ef4444","#f97316","#f59e0b","#84cc16","#22c55e"]  # 1→red … 5→green

ENDOSCAPES_LABELS = ["background", "cystic_plate", "calot_triangle",
                     "cystic_artery", "cystic_duct", "gallbladder", "tool"]
ENDOSCAPES_COLORS = {
    "cystic_plate":    "#f59e0b",
    "calot_triangle":  "#10b981",
    "cystic_artery":   "#ef4444",
    "cystic_duct":     "#3b82f6",
    "gallbladder":     "#8b5cf6",
    "tool":            "#ec4899",
    "background":      "#9ca3af",
}
ENDOSCAPES_IMG_W, ENDOSCAPES_IMG_H = 854, 480

ANNOTATION_SUFFIX = {
    "phase":  "_Annotation_Phase.csv",
    "tool":   "_Annotation_Instrument.csv",
    "action": "_Annotation_Action.csv",
}

ANN_SUBDIR = {
    "phase":  "Phase",
    "tool":   "Instrument",
    "action": "Action",
}

# ── Parse prompts.py ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def parse_task_dataset_map() -> dict[str, list[str]]:
    """Return {task_type: [dataset, ...]} by parsing PROMPTS keys in prompts.py."""
    text = PROMPTS_PY.read_text()
    # Match PROMPTS[('some_task_name', 'ModelName')]
    keys = re.findall(r"PROMPTS\[\('([^']+)',\s*'[^']+'\)\]", text)
    task_map: dict[str, set[str]] = {}
    for key in keys:
        # key format: {dataset}_{task_type}  e.g. heichole_phase_recognition
        # dataset prefix = everything before the first known task suffix
        parts = key.split("_")
        # Identify split point: task suffixes are known keywords
        TASK_KEYWORDS = {"phase", "tool", "action", "skill", "error",
                         "gesture", "triplet", "cvs", "maneuver", "anatomy"}
        split_idx = next(
            (i for i, p in enumerate(parts) if p in TASK_KEYWORDS), 1
        )
        dataset = "_".join(parts[:split_idx])
        task_type = "_".join(parts[split_idx:])
        # Strip few-shot suffixes for grouping purposes
        task_type = re.sub(r"_(oneshot|threeshot|fiveshot)$", "", task_type)
        if task_type:
            task_map.setdefault(task_type, set()).add(dataset)
    return {k: sorted(v) for k, v in sorted(task_map.items())}


# ── Dataset helpers ───────────────────────────────────────────────────────────
def heichole_videos() -> list[str]:
    frame_dir = DATASET_ROOT / "HeiChole" / "extracted_frames"
    if not frame_dir.exists():
        return []
    return sorted(p.name for p in frame_dir.iterdir() if p.is_dir())


# ── Endoscapes helpers ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_endoscapes_coco() -> tuple[dict[str, list[dict]], dict[str, tuple[int, int]]]:
    """Parse test/annotation_coco.json → ({stem: [bbox_dict,...]}, {stem: (W,H)})."""
    coco_path = DATASET_ROOT / "endoscapes" / "test" / "annotation_coco.json"
    with open(coco_path) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map  = {img["id"]: img for img in coco["images"]}
    frame_boxes: dict[str, list[dict]] = {}
    frame_dims:  dict[str, tuple[int, int]] = {}
    for img in coco["images"]:
        stem = img["file_name"].replace(".jpg", "")
        frame_dims[stem] = (img["width"], img["height"])
        frame_boxes.setdefault(stem, [])
    for ann in coco["annotations"]:
        img  = img_map[ann["image_id"]]
        stem = img["file_name"].replace(".jpg", "")
        W, H = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = x, y, x + w, y + h
        label = cat_map.get(ann["category_id"], str(ann["category_id"]))
        frame_boxes[stem].append({
            "label": label,
            "color": ENDOSCAPES_COLORS.get(label, "#aaa"),
            "x0": round(x0), "y0": round(y0), "x1": round(x1), "y1": round(y1),
            "x0_pct": round(x0 / W * 100, 2),
            "y0_pct": round(y0 / H * 100, 2),
            "x1_pct": round(x1 / W * 100, 2),
            "y1_pct": round(y1 / H * 100, 2),
        })
    return frame_boxes, frame_dims


@lru_cache(maxsize=1)
def endoscapes_annotated_test_frames() -> dict[str, list[str]]:
    """Return {video_id: [frame_key, ...]} for all BBox201 test frames (40 vids, 312 frames)."""
    frame_boxes, _ = _load_endoscapes_coco()
    result: dict[str, list[str]] = {}
    for stem in sorted(frame_boxes):
        vid = stem.split("_")[0]
        result.setdefault(vid, []).append(stem)
    return result


def get_endoscapes_bboxes(frame_key: str) -> list[dict]:
    frame_boxes, _ = _load_endoscapes_coco()
    return frame_boxes.get(frame_key, [])


def heichole_frames(video: str) -> list[Path]:
    frame_dir = DATASET_ROOT / "HeiChole" / "extracted_frames" / video
    return sorted(frame_dir.glob("*.png")) if frame_dir.exists() else []


def load_heichole_annotation(video: str, ann_type: str) -> pd.DataFrame:
    suffix = ANNOTATION_SUFFIX[ann_type]
    ann_path = DATASET_ROOT / "HeiChole" / "Annotations" / ANN_SUBDIR[ann_type] / f"{video}{suffix}"
    if not ann_path.exists():
        return pd.DataFrame()
    return pd.read_csv(ann_path, header=None)


def get_label_for_frame(df: pd.DataFrame, frame_idx: int, ann_type: str) -> dict:
    if df.empty:
        return {}
    row = df[df[0] == frame_idx]
    if row.empty:
        return {}
    row = row.iloc[0]
    cfg = HEICHOLE_LABELS[ann_type]
    if cfg["type"] == "single":
        class_idx = int(row[1])
        label_name = cfg["names"][class_idx] if class_idx < len(cfg["names"]) else str(class_idx)
        return {"type": "single", "label": label_name, "index": class_idx}
    else:
        active = []
        for name, (start, end) in zip(cfg["names"], cfg["col_groups"]):
            cols = list(range(start, end))
            if any(row[c] for c in cols if c < len(row)):
                active.append(name)
        return {"type": "multi", "labels": active}


# ── Load all prompts from prompts.py ─────────────────────────────────────────
@lru_cache(maxsize=1)
def load_all_prompts() -> dict:
    """Exec the body of get_prompts() with a dummy namespace to extract all PROMPTS entries."""
    src = PROMPTS_PY.read_text()
    lines = src.splitlines()

    # Find where function body starts (after 'def get_prompts') and strip 8-space indent
    body_lines = []
    in_func = False
    for line in lines:
        if re.match(r"^def get_prompts", line):
            in_func = True
            continue
        if not in_func:
            continue
        # Stop at the return statement
        if re.match(r"^\s+return\b", line):
            break
        # Strip 8-space indentation used in this file
        if line.startswith("        "):
            body_lines.append(line[8:])
        elif line.startswith("    "):
            body_lines.append(line[4:])
        else:
            body_lines.append(line)

    import os.path as osp
    namespace: dict = {"path": "", "task": "", "model": "", "osp": osp}
    try:
        exec("\n".join(body_lines), namespace)  # noqa: S102
    except Exception:
        pass
    return namespace.get("PROMPTS", {})


def get_prompts_for_task(task_key: str) -> dict[str, str]:
    """Return {model: prompt_text} for all models that have a prompt for task_key."""
    all_prompts = load_all_prompts()
    result = {}
    for (t, model), prompt in all_prompts.items():
        if t != task_key:
            continue
        # Render prompt to a readable string
        if isinstance(prompt, str):
            text = prompt.strip()
        elif isinstance(prompt, list):
            # Lists are used for CLIP/PaliGemma (candidate captions) or few-shot sequences
            # Show only string items; skip file paths
            text_items = [
                item.strip() for item in prompt
                if isinstance(item, str) and not os.path.isabs(item)
            ]
            text = "\n".join(f"• {s}" for s in text_items) if text_items else "(list of image paths)"
        else:
            text = str(prompt)
        result[model] = text
    return result


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/api/tasks")
def get_tasks():
    task_map = dict(parse_task_dataset_map())
    # Inject tasks that have data loaders but no prompt in prompts.py
    for full_key, (dataset, _) in TASK_DATASET_ANNOTATION.items():
        parts = full_key.split("_")
        TASK_KEYWORDS = {"phase", "tool", "action", "skill", "error", "gesture",
                         "triplet", "cvs", "maneuver", "anatomy", "object", "detection"}
        split_idx = next((i for i, p in enumerate(parts) if p in TASK_KEYWORDS), 1)
        task_type = "_".join(parts[split_idx:])
        ds = "_".join(parts[:split_idx])
        if task_type:
            task_map.setdefault(task_type, [])
            if ds not in task_map[task_type]:
                task_map[task_type].append(ds)
    return {k: sorted(v) for k, v in sorted(task_map.items())}


@app.get("/api/known_tasks")
def get_known_tasks():
    """Return list of task keys that have a registered data loader."""
    return list(TASK_DATASET_ANNOTATION.keys())


def _heichole_frame_count() -> int:
    base = DATASET_ROOT / "HeiChole" / "extracted_frames"
    if not base.exists():
        return 0
    return sum(len(list(d.glob("*.png"))) for d in base.iterdir() if d.is_dir())


def _heichole_video_count() -> int:
    base = DATASET_ROOT / "HeiChole" / "extracted_frames"
    if not base.exists():
        return 0
    return sum(1 for d in base.iterdir() if d.is_dir())


def _task_suffix(task_key: str) -> str:
    """Extract the task type suffix from a full task key like 'heichole_phase_recognition'."""
    TASK_KEYWORDS = {"phase", "tool", "action", "skill", "error", "gesture",
                     "triplet", "cvs", "maneuver", "anatomy", "object", "detection"}
    parts = task_key.split("_")
    split_idx = next((i for i, p in enumerate(parts) if p in TASK_KEYWORDS), len(parts))
    return "_".join(parts[split_idx:])


@app.get("/api/dashboard")
def get_dashboard():
    entries = []
    for task_key, (dataset, ann_type) in TASK_DATASET_ANNOTATION.items():
        task_type = _task_suffix(task_key)
        granularity = TASK_GRANULARITY.get(task_type, "frame")

        if dataset == "HeiChole":
            if granularity == "frame":
                count = _heichole_frame_count()
                unit = "frames"
                videos = _heichole_video_count()
            else:
                # video-level: count videos with skill annotations
                skill_dir = DATASET_ROOT / "HeiChole" / "Annotations" / "Skill"
                count = len(set(
                    p.name.split("_Skill")[0].split("_Calot")[0].split("_Dissection")[0]
                    for p in skill_dir.glob("*_Skill.csv")
                )) if skill_dir.exists() else 0
                unit = "videos"
                videos = count

        elif dataset == "endoscapes":
            frame_boxes, _ = _load_endoscapes_coco()
            count = sum(1 for v in frame_boxes.values() if v)
            unit = "frames"
            videos = len(endoscapes_annotated_test_frames())

        elif dataset == "JIGSAWS":
            base = DATASET_ROOT / "JIGSAWS"
            if ann_type == "skill":
                count = sum(
                    len([l for l in (base / cat / f"meta_file_{cat}.txt").read_text().splitlines()
                         if l.strip() and len(l.strip().split()) > 3])
                    for cat in JIGSAWS_CATEGORIES if (base / cat).exists()
                )
                unit = "videos"
                videos = count
            else:  # gesture
                count = sum(
                    sum(1 for line in (base / cat / "transcriptions" / f).read_text().splitlines() if line.strip())
                    for cat in JIGSAWS_CATEGORIES if (base / cat).exists()
                    for f in (base / cat / "transcriptions").iterdir()
                )
                unit = "segments"
                videos = sum(
                    len(list((base / cat / "transcriptions").iterdir()))
                    for cat in JIGSAWS_CATEGORIES if (base / cat).exists()
                )

        elif dataset == "autolaparo":
            label_file = DATASET_ROOT / "autolaparo" / "task2" / "laparoscope_motion_label.txt"
            if label_file.exists():
                lines = [l for l in label_file.read_text().splitlines()[1:] if l.strip()]
                count = sum(1 for l in lines if int(l.split()[0]) >= 228)
                unit = "clips"
                videos = count
            else:
                count = 0; unit = "clips"; videos = 0

        else:
            count = 0
            unit = "frames"
            videos = 0

        entries.append({
            "task_key": task_key,
            "dataset": dataset,
            "task_type": task_type,
            "granularity": granularity,
            "count": count,
            "unit": unit,
            "videos": videos,
        })

    sorted_entries = sorted(entries, key=lambda e: (e["granularity"], e["dataset"], e["task_type"]))
    # Unique frame counts per dataset (avoid triple-counting HeiChole frame tasks)
    frame_datasets: dict[str, int] = {}
    video_total = 0
    for e in sorted_entries:
        if e["granularity"] == "frame":
            frame_datasets[e["dataset"]] = max(frame_datasets.get(e["dataset"], 0), e["count"])
        else:
            video_total += e["count"]
    return {
        "entries": sorted_entries,
        "totals": {
            "frames": sum(frame_datasets.values()),
            "video_items": video_total,
        }
    }


# ── Video task summary helpers ────────────────────────────────────────────────
def _heichole_skill_summary() -> dict:
    skill_dir = DATASET_ROOT / "HeiChole" / "Annotations" / "Skill"
    videos = []
    test_ids = {1, 4, 13, 16, 22}
    for f in sorted(skill_dir.glob("*_Skill.csv")):
        name = f.stem
        if "Calot" in name or "Dissection" in name:
            continue
        vid_name = name.replace("_Skill", "")
        vid_id = int(vid_name.split("Chole")[1])
        scores = [int(x) for x in f.read_text().strip().split(",")]
        mp4 = DATASET_ROOT / "HeiChole" / "Videos" / "Full" / f"{vid_name}.mp4"
        videos.append({
            "video": vid_name, "vid_id": vid_id,
            "scores": scores, "is_test": vid_id in test_ids,
            "video_url": f"/video/HeiChole/Videos/Full/{vid_name}.mp4" if mp4.exists() else None,
        })
    videos.sort(key=lambda v: v["vid_id"])
    return {"type": "skill", "criteria": HEICHOLE_SKILL_CRITERIA,
            "score_colors": SCORE_COLORS, "videos": videos}


def _jigsaws_skill_summary() -> dict:
    base = DATASET_ROOT / "JIGSAWS"
    cols = ["video_name","experience","grs","respect_for_tissue","suture_needle_handling",
            "time_and_motion","flow_of_operation","overall_performance","quality_of_final_product"]
    score_cols = cols[3:]
    videos = []
    for cat in JIGSAWS_CATEGORIES:
        meta = base / cat / f"meta_file_{cat}.txt"
        if not meta.exists():
            continue
        # prefer video_mp4 dir, else video dir
        vid_dir = base / cat / ("video_mp4" if (base / cat / "video_mp4").exists() else "video")
        for line in meta.read_text().splitlines():
            parts = [s for s in line.strip().split("\t") if s.strip()]
            if len(parts) != len(cols):
                continue
            row = dict(zip(cols, parts))
            vid_name = row["video_name"]
            # strip category prefix: "Knot_Tying_B001" → "B001"
            short = vid_name.replace(f"{cat}_", "")
            # look for capture1 file
            mp4_name = f"{vid_name}_capture1.mp4"
            mp4_path = vid_dir / mp4_name
            video_url = f"/transcode/JIGSAWS/{cat}/{vid_dir.name}/{mp4_name}" if mp4_path.exists() else None
            videos.append({
                "video": vid_name, "category": cat.replace("_", " "),
                "experience": JIGSAWS_EXPERIENCE.get(row["experience"], row["experience"]),
                "scores": [int(row[c]) for c in score_cols],
                "video_url": video_url,
            })
    return {"type": "skill", "criteria": JIGSAWS_SKILL_CRITERIA,
            "score_colors": SCORE_COLORS, "videos": videos}


def _jigsaws_gesture_summary() -> dict:
    base = DATASET_ROOT / "JIGSAWS"
    from collections import Counter
    all_clips = []
    for cat in JIGSAWS_CATEGORIES:
        trans_dir = base / cat / "transcriptions"
        if not trans_dir.exists():
            continue
        vid_dir = base / cat / ("video_mp4" if (base / cat / "video_mp4").exists() else "video")
        for f in sorted(trans_dir.iterdir()):
            vid_name = f.stem  # e.g. Knot_Tying_B001
            mp4_name = f"{vid_name}_capture1.mp4"
            mp4_path = vid_dir / mp4_name
            video_url = f"/transcode/JIGSAWS/{cat}/{vid_dir.name}/{mp4_name}" if mp4_path.exists() else None
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, label = parts
                    all_clips.append({
                        "video": vid_name, "category": cat.replace("_", " "),
                        "start": int(start), "end": int(end), "label": label,
                        "color": GESTURE_COLORS.get(label, "#6b7280"),
                        "video_url": video_url,
                    })
    distribution = dict(Counter(c["label"] for c in all_clips))
    return {"type": "classification", "label_type": "gesture",
            "colors": GESTURE_COLORS, "distribution": distribution, "clips": all_clips}


def _autolaparo_maneuver_summary() -> dict:
    label_file = DATASET_ROOT / "autolaparo" / "task2" / "laparoscope_motion_label.txt"
    clips_dir = DATASET_ROOT / "autolaparo" / "task2" / "clips"
    label_map = {"0":"Static","1":"Up","2":"Down","3":"Left","4":"Right","5":"Zoom-in","6":"Zoom-out"}
    from collections import Counter
    clips = []
    for line in label_file.read_text().splitlines()[1:]:
        parts = line.strip().split()
        if not parts: continue
        clip_id, maneuver_num = parts[0], parts[1]
        if int(clip_id) < 228: continue
        label = label_map[maneuver_num]
        mp4_name = f"{int(clip_id):03d}.mp4"
        mp4_path = clips_dir / mp4_name
        clips.append({
            "clip": clip_id, "label": label,
            "color": AUTOLAPARO_MANEUVER_COLORS[label],
            "video_url": f"/transcode/autolaparo/task2/clips/{mp4_name}" if mp4_path.exists() else None,
        })
    distribution = dict(Counter(c["label"] for c in clips))
    return {"type": "classification", "label_type": "maneuver",
            "colors": AUTOLAPARO_MANEUVER_COLORS, "distribution": distribution, "clips": clips}


@app.get("/api/video_summary")
def get_video_summary(task: str = Query(...), dataset: str = Query(...)):
    key = f"{dataset}_{task}"
    if key not in TASK_DATASET_ANNOTATION:
        raise HTTPException(404, f"No data for {dataset}/{task}")
    ds, ann_type = TASK_DATASET_ANNOTATION[key]
    if ds == "HeiChole" and ann_type == "skill":
        return _heichole_skill_summary()
    if ds == "JIGSAWS" and ann_type == "skill":
        return _jigsaws_skill_summary()
    if ds == "JIGSAWS" and ann_type == "gesture":
        return _jigsaws_gesture_summary()
    if ds == "autolaparo" and ann_type == "maneuver":
        return _autolaparo_maneuver_summary()
    raise HTTPException(404, "Video summary not available")


@app.get("/api/videos")
def get_videos(task: str = Query(...)):
    dataset_ann = TASK_DATASET_ANNOTATION.get(task)
    if not dataset_ann:
        return []
    dataset, _ = dataset_ann
    # Video-level tasks (skill, gesture, maneuver) don't use the per-video frame selector
    task_type = _task_suffix(task)
    if TASK_GRANULARITY.get(task_type) == "video":
        return []
    if dataset == "HeiChole":
        return heichole_videos()
    return []


@app.get("/api/samples")
def get_samples(
    task: str = Query(...),
    video: str = Query(None),
    n: int = Query(12),
    seed: int = Query(42),
):
    dataset_ann = TASK_DATASET_ANNOTATION.get(task)
    if not dataset_ann:
        raise HTTPException(404, f"No data loader for task '{task}'")
    dataset, ann_type = dataset_ann

    if dataset == "HeiChole":
        frames = heichole_frames(video)
        if not frames:
            raise HTTPException(404, f"No frames found for video '{video}'")
        df = load_heichole_annotation(video, ann_type)
        if n == 0:
            candidates = frames
        else:
            stride = max(1, len(frames) // n)
            candidates = frames[::stride][:n]
        results = []
        for frame_path in candidates:
            frame_idx = int(frame_path.stem.split("_")[-1])
            label = get_label_for_frame(df, frame_idx, ann_type)
            results.append({
                "image_url": f"/image/HeiChole/extracted_frames/{video}/{frame_path.name}",
                "frame_idx": frame_idx,
                "video": video,
                "label": label,
            })
        return results

    if dataset == "endoscapes":
        frame_boxes, _ = _load_endoscapes_coco()
        # Only frames that have at least one bounding box
        annotated = sorted(k for k, v in frame_boxes.items() if v)
        if n == 0:
            candidates = annotated
        else:
            stride = max(1, len(annotated) // n)
            candidates = annotated[::stride][:n]
        results = []
        for key in candidates:
            vid_id, frame_num = key.split("_", 1)
            results.append({
                "image_url": f"/image/endoscapes/test/{key}.jpg",
                "frame_idx": int(frame_num),
                "video": vid_id,
                "label": {"type": "bbox", "boxes": frame_boxes[key]},
            })
        return results

    raise HTTPException(404, f"Dataset '{dataset}' not yet supported")


PHASE_COLORS = [
    "#c0392b",  # Preparation
    "#27ae60",  # Calot Triangle Dissection
    "#5dade2",  # Clipping & Cutting
    "#d4ac0d",  # Gallbladder Dissection
    "#e67e22",  # Gallbladder Packaging
    "#8e44ad",  # Cleaning & Coagulation
    "#b39ddb",  # Gallbladder Retraction
]


@app.get("/api/phase_timeline")
def get_phase_timeline(video: str = Query(...)):
    """Return run-length-encoded phase segments for the full video timeline."""
    df = load_heichole_annotation(video, "phase")
    if df.empty:
        raise HTTPException(404, f"No phase annotation for '{video}'")

    names = HEICHOLE_LABELS["phase"]["names"]
    total = len(df)
    segments = []
    prev_phase = int(df.iloc[0][1])
    run_start = 0

    for i, row in df.iterrows():
        phase = int(row[1])
        if phase != prev_phase:
            segments.append({
                "label": names[prev_phase] if prev_phase < len(names) else str(prev_phase),
                "color": PHASE_COLORS[prev_phase] if prev_phase < len(PHASE_COLORS) else "#aaa",
                "pct": round((i - run_start) / total * 100, 3),
                "start_frame": int(df.iloc[run_start][0]),
                "end_frame": int(df.iloc[i - 1][0]),
            })
            run_start = i
            prev_phase = phase

    # Last segment
    segments.append({
        "label": names[prev_phase] if prev_phase < len(names) else str(prev_phase),
        "color": PHASE_COLORS[prev_phase] if prev_phase < len(PHASE_COLORS) else "#aaa",
        "pct": round((total - run_start) / total * 100, 3),
        "start_frame": int(df.iloc[run_start][0]),
        "end_frame": int(df.iloc[-1][0]),
    })
    return segments


@app.get("/api/prompt")
def get_prompt(task: str = Query(...)):
    """Return all model prompts for a given full task key (e.g. heichole_phase_recognition)."""
    prompts = get_prompts_for_task(task)
    if not prompts:
        return {}
    return prompts


@app.get("/image/{rest_of_path:path}")
def serve_image(rest_of_path: str):
    image_path = DATASET_ROOT / rest_of_path
    if not image_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(image_path))


@app.get("/api/action_histogram")
def get_action_histogram(video: str = Query(...)):
    """Return total frame counts per action label across the full annotation CSV."""
    df = load_heichole_annotation(video, "action")
    if df.empty:
        raise HTTPException(404, f"No action annotation for '{video}'")

    cfg = HEICHOLE_LABELS["action"]
    total = len(df)
    result = []
    for name, (start, end) in zip(cfg["names"], cfg["col_groups"]):
        cols = list(range(start, end))
        count = int(df[cols].any(axis=1).sum())
        result.append({"label": name, "count": count, "pct": round(count / total * 100, 2)})
    return {"total_frames": total, "labels": result}


@app.get("/video/{rest_of_path:path}")
def serve_video(rest_of_path: str):
    video_path = DATASET_ROOT / rest_of_path
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/transcode/{rest_of_path:path}")
def transcode_video(rest_of_path: str, ss: float = Query(0.0)):
    """Transcode any video to H.264/MP4 on-the-fly for browser playback."""
    import subprocess
    from fastapi.responses import StreamingResponse
    video_path = DATASET_ROOT / rest_of_path
    if not video_path.exists():
        raise HTTPException(404, "Video not found")

    def stream():
        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-ss", str(ss),
            "-i", str(video_path),
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "26",
            "-acodec", "aac", "-ac", "1",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            proc.kill()

    return StreamingResponse(stream(), media_type="video/mp4")


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
