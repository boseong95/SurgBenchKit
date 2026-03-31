import re
import os
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
    "heichole_phase_recognition":   ("HeiChole", "phase"),
    "heichole_tool_recognition":    ("HeiChole", "tool"),
    "heichole_action_recognition":  ("HeiChole", "action"),
}

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
    return parse_task_dataset_map()


@app.get("/api/videos")
def get_videos(task: str = Query(...)):
    dataset_ann = TASK_DATASET_ANNOTATION.get(task)
    if dataset_ann and dataset_ann[0] == "HeiChole":
        return heichole_videos()
    return []


@app.get("/api/samples")
def get_samples(
    task: str = Query(...),
    video: str = Query(...),
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
        # Sample evenly across the video
        df = load_heichole_annotation(video, ann_type)
        rng = random.Random(seed)
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


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
