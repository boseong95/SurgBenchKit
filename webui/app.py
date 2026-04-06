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
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/home/ubuntu/datasets/vlm"))

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

CHOLEC80_LABELS = {
    "phase": {
        "type": "single",
        "names": ["Preparation", "CalotTriangleDissection", "ClippingCutting",
                  "GallbladderDissection", "GallbladderPackaging",
                  "CleaningCoagulation", "GallbladderRetraction"],
        # Display-friendly names for UI
        "display": ["Preparation", "Calot Triangle Dissection", "Clipping & Cutting",
                    "Gallbladder Dissection", "Gallbladder Packaging",
                    "Cleaning & Coagulation", "Gallbladder Retraction"],
    },
    "tool": {
        "type": "multi",
        # Column headers in tool annotation file
        "names": ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"],
        "display": ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "Specimen Bag"],
    },
}

TASK_DATASET_ANNOTATION = {
    # Frame-based
    "heichole_phase_recognition":           ("HeiChole",    "phase"),
    "heichole_tool_recognition":            ("HeiChole",    "tool"),
    "heichole_action_recognition":          ("HeiChole",    "action"),
    "cholec80_phase_recognition":           ("Cholec80",    "phase"),
    "cholec80_tool_recognition":            ("Cholec80",    "tool"),
    "cholect45_triplet_recognition":        ("CholecT45",   "triplet"),
    "cholect50_phase_recognition":          ("CholecT50",   "phase"),
    "cholect50_tool_recognition":           ("CholecT50",   "tool"),
    "cholect50_phase_planning":             ("CholecT50",   "phase_planning"),
    "endoscapes_object_detection":          ("endoscapes",            "bbox"),
    "vtrb_suturing_object_detection":       ("vtrb_suturing",         "bbox"),
    "vtrb_suturing_phase_recognition":      ("vtrb_suturing_phase",   "phase_easy"),
    # Video-based
    "heichole_skill_assessment":            ("HeiChole",    "skill"),
    "knot_tying_skill_assessment":           ("Knot_Tying",  "skill"),
    "needle_passing_skill_assessment":       ("Needle_Passing", "skill"),
    "suturing_skill_assessment":             ("Suturing",    "skill"),
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
    "phase_planning":           "video",
    "error_detection":          "frame",
    "error_recognition":        "frame",
    "anatomy_presence":         "frame",
    "skill_assessment":         "video",
    "gesture_classification":   "video",
    "maneuver_classification":  "video",
}

# ── Video task label definitions ─────────────────────────────────────────────
HEICHOLE_SKILL_CRITERIA = [
    "Depth Perception", "Bimanual Dexterity", "Efficiency",
    "Tissue Handling", "Case Difficulty",
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

VTRB_SUTURING_DIR = DATASET_ROOT / "VTRB-Suturing-VQA"
VTRB_PHASE_EASY_DIR = VTRB_SUTURING_DIR / "phase_predict_easy"
VTRB_SUTURING_COLORS = {
    "grippers":      "#00ff88",
    "target_tissue": "#ff6644",
    "wound_gap":     "#44aaff",
    "bites":         "#ff00ff",
    "needle":        "#dd88ff",
}

OUTPUTS_DIR = REPO_ROOT / "outputs"

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


# ── VTRB-Suturing helpers ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_vtrb_suturing_annotations() -> dict[str, list[dict]]:
    """Load bbox_annotations.json → {folder/frame.png: [box_dict, ...]}."""
    ann_file = VTRB_SUTURING_DIR / "bbox_annotations.json"
    if not ann_file.exists():
        return {}
    with open(ann_file) as f:
        raw = json.load(f)
    result: dict[str, list[dict]] = {}
    for frame_key, boxes in raw.items():
        result[frame_key] = [
            {
                "label": b["class"],
                "color": VTRB_SUTURING_COLORS.get(b["class"], "#aaa"),
                "x0": b["x"], "y0": b["y"],
                "x1": b["x"] + b["w"], "y1": b["y"] + b["h"],
                "x0_pct": round(b["x_pct"], 2),
                "y0_pct": round(b["y_pct"], 2),
                "x1_pct": round(b["x_pct"] + b["w_pct"], 2),
                "y1_pct": round(b["y_pct"] + b["h_pct"], 2),
            }
            for b in boxes
        ]
    return result


def vtrb_suturing_videos() -> list[str]:
    frames_dir = VTRB_SUTURING_DIR / "frames"
    if not frames_dir.exists():
        return []
    return sorted(d.name for d in frames_dir.iterdir() if d.is_dir())


def vtrb_suturing_frames(video: str) -> list[Path]:
    frame_dir = VTRB_SUTURING_DIR / "frames" / video
    return sorted(frame_dir.glob("*.png")) if frame_dir.exists() else []


def heichole_frames(video: str) -> list[Path]:
    frame_dir = DATASET_ROOT / "HeiChole" / "extracted_frames" / video
    return sorted(frame_dir.glob("*.png")) if frame_dir.exists() else []


def cholec80_videos(split: str = "test") -> list[str]:
    frame_dir = DATASET_ROOT / "Cholec80" / "frames_25fps" / split
    if not frame_dir.exists():
        return []
    return sorted(p.name for p in frame_dir.iterdir() if p.is_dir())


def cholec80_frames(video: str, split: str = "test") -> list[Path]:
    frame_dir = DATASET_ROOT / "Cholec80" / "frames_25fps" / split / video
    return sorted(frame_dir.glob("*.jpg"), key=lambda p: int(p.stem)) if frame_dir.exists() else []


def load_cholec80_phase_annotation(video: str) -> pd.DataFrame:
    """Load phase annotation as DataFrame with columns [frame_idx, phase_name]."""
    ann_path = DATASET_ROOT / "Cholec80" / "phase_annotations" / f"{video}-phase.txt"
    if not ann_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(ann_path, sep="\t", header=0, names=["frame", "phase"])
    return df


def load_cholec80_tool_annotation(video: str) -> pd.DataFrame:
    """Load tool annotation as DataFrame with Frame + 7 binary tool columns."""
    ann_path = DATASET_ROOT / "Cholec80" / "tool_annotations" / f"{video}-tool.txt"
    if not ann_path.exists():
        return pd.DataFrame()
    return pd.read_csv(ann_path, sep="\t", header=0)


def get_cholec80_phase_label(df: pd.DataFrame, frame_idx: int) -> dict:
    if df.empty:
        return {}
    row = df[df["frame"] == frame_idx]
    if row.empty:
        # Phase annotations are every frame; find nearest
        nearest = df.iloc[(df["frame"] - frame_idx).abs().argsort().iloc[0]]
        phase_name = nearest["phase"]
    else:
        phase_name = row.iloc[0]["phase"]
    names = CHOLEC80_LABELS["phase"]["names"]
    display = CHOLEC80_LABELS["phase"]["display"]
    idx = names.index(phase_name) if phase_name in names else -1
    label = display[idx] if idx >= 0 else phase_name
    return {"type": "single", "label": label, "index": idx}


def get_cholec80_tool_label(df: pd.DataFrame, frame_idx: int) -> dict:
    if df.empty:
        return {}
    row = df[df["Frame"] == frame_idx]
    if row.empty:
        return {"type": "multi", "labels": []}
    row = row.iloc[0]
    names = CHOLEC80_LABELS["tool"]["names"]
    display = CHOLEC80_LABELS["tool"]["display"]
    active = [d for n, d in zip(names, display) if row.get(n, 0)]
    return {"type": "multi", "labels": active}


CHOLECT45_ROOT = DATASET_ROOT / "CholecT45"
CHOLECT45_TEST_VIDEOS = ['VID06', 'VID10', 'VID14', 'VID32', 'VID42', 'VID51', 'VID73', 'VID74', 'VID80']

CHOLECT50_ROOT = DATASET_ROOT / "CholecT50"
CHOLECT50_TEST_VIDEOS = ['VID06', 'VID10', 'VID14', 'VID32', 'VID42', 'VID51', 'VID73', 'VID74', 'VID80']
CHOLECT50_PHASE_LABELS = [
    "Preparation", "Calot Triangle Dissection", "Clipping & Cutting",
    "Gallbladder Dissection", "Gallbladder Packaging", "Cleaning & Coagulation", "Gallbladder Extraction",
]
CHOLECT50_TOOL_LABELS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator"]

@lru_cache(maxsize=1)
def _load_cholect45_triplet_names() -> list[str]:
    """Return list of 100 triplet display strings: 'grasper → dissect → cystic_plate'."""
    triplet_txt = CHOLECT45_ROOT / "dict" / "triplet.txt"
    names = []
    with open(triplet_txt) as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                comps = parts[1].split(",")
                names.append(f"{comps[0]} → {comps[1]} → {comps[2]}")
    return names


def cholect45_videos() -> list[str]:
    rgb_dir = CHOLECT45_ROOT / "rgb"
    if not rgb_dir.exists():
        return []
    return sorted(
        p.name for p in rgb_dir.iterdir()
        if p.is_dir() and p.name in CHOLECT45_TEST_VIDEOS
    )


def cholect45_frames(video: str) -> list[Path]:
    frame_dir = CHOLECT45_ROOT / "rgb" / video
    return sorted(frame_dir.glob("*.png"), key=lambda p: int(p.stem)) if frame_dir.exists() else []


def load_cholect45_triplet_annotation(video: str) -> dict[int, list[str]]:
    """Return {frame_idx: [active triplet display strings]}."""
    txt_path = CHOLECT45_ROOT / "triplet" / f"{video}.txt"
    if not txt_path.exists():
        return {}
    names = _load_cholect45_triplet_names()
    result = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame_idx = int(parts[0])
            active = [names[i] for i, v in enumerate(parts[1:]) if v == "1" and i < len(names)]
            result[frame_idx] = active
    return result


def cholect50_videos() -> list[str]:
    videos_dir = CHOLECT50_ROOT / "videos"
    if not videos_dir.exists():
        return []
    return sorted(
        p.name for p in videos_dir.iterdir()
        if p.is_dir() and p.name in CHOLECT50_TEST_VIDEOS
    )


def cholect50_frames(video: str) -> list[Path]:
    frame_dir = CHOLECT50_ROOT / "videos" / video
    return sorted(frame_dir.glob("*.png"), key=lambda p: int(p.stem)) if frame_dir.exists() else []


@lru_cache(maxsize=32)
def load_cholect50_annotation(video: str) -> dict[int, dict]:
    """Return {frame_idx: {"phase": int, "tools": list[str]}} from CholecT50 JSON labels."""
    json_path = CHOLECT50_ROOT / "labels" / f"{video}.json"
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    annotations = data.get("annotations", {})
    result = {}
    for frame_str, instances in annotations.items():
        frame_idx = int(frame_str)
        phase_id = -1
        active_tools = []
        for instance in instances:
            if len(instance) > 9:
                p = instance[9]
                if isinstance(p, (int, float)) and p >= 0:
                    phase_id = int(p)
            if len(instance) > 1:
                tool_id = instance[1]
                if isinstance(tool_id, int) and 0 <= tool_id < len(CHOLECT50_TOOL_LABELS):
                    tool_name = CHOLECT50_TOOL_LABELS[tool_id]
                    if tool_name not in active_tools:
                        active_tools.append(tool_name)
        result[frame_idx] = {"phase": phase_id, "tools": active_tools}
    return result


CHOLECT50_PHASE_PLANNING_PATH = CHOLECT50_ROOT / "phase_planning" / "phase_planning.json"


@lru_cache(maxsize=1)
def load_phase_planning_data() -> dict:
    """Load CholecT50 phase planning JSON."""
    if not CHOLECT50_PHASE_PLANNING_PATH.exists():
        return {"samples": []}
    return json.loads(CHOLECT50_PHASE_PLANNING_PATH.read_text())


def phase_planning_videos() -> list[str]:
    data = load_phase_planning_data()
    return sorted(set(s["video_id"] for s in data["samples"]))


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


_JIGSAWS_SKILL_DIMENSIONS = [
    ("jigsaws_skill_assessment",                      "Respect for Tissue"),
    ("jigsaws_skill_assessment_suture_needle_handling", "Suture / Needle Handling"),
    ("jigsaws_skill_assessment_time_and_motion",        "Time & Motion"),
    ("jigsaws_skill_assessment_flow_of_operation",      "Flow of Operation"),
    ("jigsaws_skill_assessment_overall_performance",    "Overall Performance"),
    ("jigsaws_skill_assessment_quality_final_product",  "Quality of Final Product"),
]
_HEICHOLE_SKILL_DIMENSIONS = [
    ("heichole_skill_assessment",              "Tissue Handling"),
    ("heichole_skill_assessment_depth_perception",  "Depth Perception"),
    ("heichole_skill_assessment_bimanual_dexterity", "Bimanual Dexterity"),
    ("heichole_skill_assessment_efficiency",   "Efficiency"),
    ("heichole_skill_assessment_difficulty",   "Case Difficulty"),
]
_JIGSAWS_CATEGORY_SKILL_KEYS = frozenset(
    {"knot_tying_skill_assessment", "needle_passing_skill_assessment", "suturing_skill_assessment"}
)


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        items = [s.strip() for s in prompt if isinstance(s, str) and not os.path.isabs(s)]
        return "\n".join(f"• {s}" for s in items) if items else "(list of image paths)"
    return str(prompt)


def get_prompts_for_task(task_key: str) -> dict[str, str]:
    """Return {model: prompt_text} for all models that have a prompt for task_key."""
    all_prompts = load_all_prompts()

    # For JIGSAWS per-category skill tasks, merge all dimension prompts per model
    if task_key in _JIGSAWS_CATEGORY_SKILL_KEYS:
        model_parts: dict[str, list[tuple[str, str]]] = {}
        for dim_key, dim_label in _JIGSAWS_SKILL_DIMENSIONS:
            for (t, model), prompt in all_prompts.items():
                if t == dim_key:
                    model_parts.setdefault(model, []).append((dim_label, _prompt_text(prompt)))
        return {
            model: "\n\n".join(f"=== {lbl} ===\n{txt}" for lbl, txt in parts)
            for model, parts in model_parts.items()
        }

    # For heichole skill assessment, also merge all dimension prompts per model
    if task_key == "heichole_skill_assessment":
        model_parts = {}
        for dim_key, dim_label in _HEICHOLE_SKILL_DIMENSIONS:
            for (t, model), prompt in all_prompts.items():
                if t == dim_key:
                    model_parts.setdefault(model, []).append((dim_label, _prompt_text(prompt)))
        return {
            model: "\n\n".join(f"=== {lbl} ===\n{txt}" for lbl, txt in parts)
            for model, parts in model_parts.items()
        }

    result = {}
    for (t, model), prompt in all_prompts.items():
        if t != task_key:
            continue
        result[model] = _prompt_text(prompt)
    return result


# ── Prediction helpers ────────────────────────────────────────────────────────
def _image_url_to_pred_key(image_url: str) -> str:
    """Map /image/… URL to the filename stem used in outputs/."""
    path = image_url.removeprefix("/image/")
    # HeiChole outputs were generated without the top-level 'HeiChole/' prefix
    if path.startswith("HeiChole/"):
        path = path.removeprefix("HeiChole/")
    # Cholec80 pred files: frames_25fps/test/video43/0.jpg → test-video43-0.jpg
    if path.startswith("Cholec80/frames_25fps/"):
        path = path.removeprefix("Cholec80/frames_25fps/")
        return path.replace("/", "-")
    key = path.replace("/", "-")
    # CholecT45/CholecT50 pred files have no dataset prefix
    if key.startswith("CholecT45-"):
        key = key.removeprefix("CholecT45-")
    elif key.startswith("CholecT50-"):
        key = key.removeprefix("CholecT50-")
    elif key.startswith("VTRB-Suturing-VQA-frames-"):
        key = key.removeprefix("VTRB-Suturing-VQA-frames-")
    return key


def _normalize_prediction(raw: dict, ann_type: str, gt_label: dict, dataset: str = "HeiChole") -> dict:
    """Convert raw prediction JSON into the same label format as GT, plus 'correct' bool."""
    if ann_type == "phase":
        idx = raw.get("phase", -1)
        if dataset == "Cholec80":
            display = CHOLEC80_LABELS["phase"]["display"]
            name = display[idx] if 0 <= idx < len(display) else str(idx)
        elif dataset == "CholecT50":
            name = CHOLECT50_PHASE_LABELS[idx] if 0 <= idx < len(CHOLECT50_PHASE_LABELS) else str(idx)
        else:
            names = HEICHOLE_LABELS["phase"]["names"]
            name = names[idx] if 0 <= idx < len(names) else str(idx)
        return {"type": "single", "label": name, "index": idx,
                "correct": idx == gt_label.get("index", -2)}
    if ann_type == "action":
        names = HEICHOLE_LABELS["action"]["names"]
        active = [n for n in names if raw.get(n.lower(), 0)]
        correct = set(active) == set(gt_label.get("labels", []))
        return {"type": "multi", "labels": active, "correct": correct}
    if ann_type == "tool":
        if dataset == "Cholec80":
            names = CHOLEC80_LABELS["tool"]["names"]
            display = CHOLEC80_LABELS["tool"]["display"]
            active = [d for n, d in zip(names, display) if raw.get(n, 0)]
        elif dataset == "CholecT50":
            active = [n for n in CHOLECT50_TOOL_LABELS if raw.get(n, False)]
        else:
            names = HEICHOLE_LABELS["tool"]["names"]
            active = [n for n in names if raw.get(n, 0)]
        correct = set(active) == set(gt_label.get("labels", []))
        return {"type": "multi", "labels": active, "correct": correct}
    return {"type": "raw", "data": raw, "correct": None}


# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/api/tasks")
def get_tasks():
    # Build authoritative task_type → [dataset, ...] from TASK_DATASET_ANNOTATION
    TASK_KEYWORDS_SET = {"phase", "tool", "action", "skill", "error", "gesture",
                         "triplet", "cvs", "maneuver", "anatomy", "object", "detection"}
    task_map: dict[str, list[str]] = {}
    for full_key in TASK_DATASET_ANNOTATION:
        parts = full_key.split("_")
        split_idx = next((i for i, p in enumerate(parts) if p in TASK_KEYWORDS_SET), 1)
        task_type = "_".join(parts[split_idx:])
        ds = "_".join(parts[:split_idx])
        if task_type:
            task_map.setdefault(task_type, [])
            if ds not in task_map[task_type]:
                task_map[task_type].append(ds)

    # Also include tasks found in prompts.py that are not in TASK_DATASET_ANNOTATION,
    # but filter out sub-dimension prompt keys (they are not standalone tasks)
    _valid_keys = set(TASK_DATASET_ANNOTATION.keys())
    for full_key, datasets in parse_task_dataset_map().items():
        parts = full_key.split("_")
        split_idx = next((i for i, p in enumerate(parts) if p in TASK_KEYWORDS_SET), 1)
        task_type = "_".join(parts[split_idx:])
        ds_prefix = "_".join(parts[:split_idx])
        # Only include if a matching full_key exists in TASK_DATASET_ANNOTATION
        candidate_key = f"{ds_prefix}_{task_type}"
        if candidate_key not in _valid_keys:
            continue
        task_map.setdefault(task_type, [])
        for ds in datasets:
            if ds not in task_map[task_type]:
                task_map[task_type].append(ds)

    return {k: sorted(v) for k, v in sorted(task_map.items())}


@app.get("/api/all_leaderboard")
def get_all_leaderboard():
    """Return leaderboard for every task that has metrics files."""
    result = {}
    if not OUTPUTS_DIR.exists():
        return result
    for task_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        lb = get_leaderboard(task=task_dir.name)
        if lb:
            result[task_dir.name] = lb
    # Include per-category JIGSAWS skill tasks (computed from prediction JSONs)
    for cat_task in sorted(_JIGSAWS_CATEGORY_SKILL_KEYS):
        lb = get_leaderboard(task=cat_task)
        if lb:
            result[cat_task] = lb
    return result


def _iou(a: list, b: list) -> float:
    """IoU between two [x1,y1,x2,y2] boxes in pixel coords."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


_JIGSAWS_SKILL_DIM_DIRS = [
    # (output_dir_suffix, display_label, meta_col)
    ("jigsaws_skill_assessment",                       "Respect for Tissue",        "respect_for_tissue"),
    ("jigsaws_skill_assessment_suture_needle_handling", "Suture / Needle Handling",  "suture_needle_handling"),
    ("jigsaws_skill_assessment_time_and_motion",        "Time & Motion",             "time_and_motion"),
    ("jigsaws_skill_assessment_flow_of_operation",      "Flow of Operation",         "flow_of_operation"),
    ("jigsaws_skill_assessment_overall_performance",    "Overall Performance",       "overall_performance"),
]
_META_COLS = ["video_name", "experience", "grs", "respect_for_tissue", "suture_needle_handling",
              "time_and_motion", "flow_of_operation", "overall_performance", "quality_of_final_product"]


def _load_jigsaws_gt(category: str) -> dict[str, dict[str, int]]:
    """Return {video_name: {col: gt_score}} for a JIGSAWS category."""
    base = DATASET_ROOT / "JIGSAWS"
    meta = base / category / f"meta_file_{category}.txt"
    if not meta.exists():
        return {}
    gt = {}
    for line in meta.read_text().splitlines():
        parts = [s for s in line.strip().split("\t") if s.strip()]
        if len(parts) != len(_META_COLS):
            continue
        row = dict(zip(_META_COLS, parts))
        gt[row["video_name"]] = {c: int(row[c]) for c in _META_COLS[3:]}
    return gt


def _compute_jigsaws_category_leaderboard(category: str) -> list:
    """Compute per-dimension skill leaderboard for one JIGSAWS category from prediction JSONs."""
    import json as _json
    from scipy.stats import spearmanr
    gt = _load_jigsaws_gt(category)
    if not gt:
        return []

    # Collect all (model, split) pairs present in any dimension output dir
    model_splits: set[tuple[str, str]] = set()
    for dir_suffix, _, _ in _JIGSAWS_SKILL_DIM_DIRS:
        task_dir = OUTPUTS_DIR / dir_suffix
        if not task_dir.exists():
            continue
        for model_dir in task_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for split_dir in model_dir.iterdir():
                if split_dir.is_dir():
                    model_splits.add((model_dir.name, split_dir.name))

    results = []
    for model, split in sorted(model_splits):
        per_dim = []
        all_acc, all_mae, all_spearman = [], [], []
        for dir_suffix, dim_label, gt_col in _JIGSAWS_SKILL_DIM_DIRS:
            preds_dir = OUTPUTS_DIR / dir_suffix / model / split
            if not preds_dir.exists():
                continue
            correct, total, mae_sum = 0, 0, 0.0
            gt_vals, pred_vals = [], []
            for vid_name, gt_scores in gt.items():
                fname = f"{category}-video-{vid_name}_capture1.avi.json"
                pred_file = preds_dir / fname
                if not pred_file.exists():
                    continue
                try:
                    raw = _json.loads(pred_file.read_text().strip().strip('"'))
                    pred = int(float(raw))
                except Exception:
                    continue
                gt_val = gt_scores.get(gt_col)
                if gt_val is None:
                    continue
                correct += int(pred == gt_val)
                mae_sum += abs(pred - gt_val)
                total += 1
                gt_vals.append(gt_val)
                pred_vals.append(pred)
            if total == 0:
                continue
            acc = round(correct / total, 4)
            mae = round(mae_sum / total, 4)
            spearman = None
            if len(gt_vals) >= 3:
                rho, _ = spearmanr(gt_vals, pred_vals)
                spearman = round(float(rho), 4) if rho == rho else None  # NaN check
            per_dim.append({"class": dim_label, "accuracy": acc, "mae": mae, "spearman": spearman})
            all_acc.append(acc)
            all_mae.append(mae)
            if spearman is not None:
                all_spearman.append(spearman)

        if not per_dim:
            continue
        overall_acc = round(sum(all_acc) / len(all_acc), 4)
        overall_mae = round(sum(all_mae) / len(all_mae), 4)
        overall_spearman = round(sum(all_spearman) / len(all_spearman), 4) if all_spearman else None
        results.append({
            "model": model, "split": split, "metric_type": "skill",
            "overall": {"accuracy": overall_acc, "mae": overall_mae, "spearman": overall_spearman},
            "per_class": per_dim,
        })
    return results


def _compute_skill_leaderboard(task: str) -> list:
    """Read root-level metrics_{task}_{model}_{split}.csv files for skill assessment tasks."""
    results = []
    prefix = f"metrics_{task}_"
    for csv_file in sorted(OUTPUTS_DIR.glob(f"metrics_{task}_*.csv")):
        stem = csv_file.stem  # metrics_{task}_{model}_{split}
        remainder = stem[len(prefix):]  # {model}_{split}
        # Known split suffix is "deleteme_for_pub"
        if remainder.endswith("_deleteme_for_pub"):
            model = remainder[:-len("_deleteme_for_pub")]
            split = "deleteme_for_pub"
        else:
            idx = remainder.rfind("_")
            model, split = remainder[:idx], remainder[idx + 1:]

        df = pd.read_csv(csv_file)
        overall_row = df[df["Dimension"] == "Overall"]
        per_class_rows = df[df["Dimension"] != "Overall"]
        if overall_row.empty:
            continue

        r = overall_row.iloc[0]
        def _f(col, _r=r):
            v = _r.get(col)
            return round(float(v), 4) if v is not None and pd.notna(v) else None

        per_class = []
        for _, pr in per_class_rows.iterrows():
            per_class.append({
                "class": str(pr["Dimension"]).replace("_", " ").title(),
                "accuracy": round(float(pr["Accuracy"]), 4) if pd.notna(pr["Accuracy"]) else None,
                "mae":      round(float(pr["MAE"]), 4)      if pd.notna(pr["MAE"])      else None,
                "spearman": round(float(pr["Spearman_rho"]), 4) if pd.notna(pr["Spearman_rho"]) else None,
            })

        results.append({
            "model": model,
            "split": split,
            "metric_type": "skill",
            "overall": {
                "accuracy": _f("Accuracy"),
                "mae":      _f("MAE"),
                "spearman": _f("Spearman_rho"),
            },
            "per_class": per_class,
        })
    return results


def _compute_triplet_leaderboard(task: str) -> list:
    """Aggregate instrument/verb/target component CSVs into a single leaderboard."""
    components = ["instrument", "verb", "target"]
    results = []

    def _float(row, col):
        v = row.get(col)
        return round(float(v), 4) if v is not None and pd.notna(v) and v != "" else None

    # Discover all models from instrument CSVs (most complete)
    inst_csvs = list(OUTPUTS_DIR.glob(f"metrics_{task}_instrument_*.csv"))
    for inst_csv in sorted(inst_csvs):
        # Parse model and split from filename:
        # metrics_cholect45_triplet_recognition_instrument_{model}_{split}.csv
        stem = inst_csv.stem  # metrics_...instrument_MODEL_SPLIT
        prefix = f"metrics_{task}_instrument_"
        rest = stem[len(prefix):]  # MODEL_SPLIT
        # split dir is last underscore-free part, model is the rest
        # Actually split dir ends with deleteme_for_pub or similar — split on last '_'
        # But model can contain '_', so we need to find the split from actual dirs
        task_dir = OUTPUTS_DIR / task
        split_name = None
        model_name = None
        for md in task_dir.iterdir():
            if not md.is_dir():
                continue
            for sd in md.iterdir():
                if not sd.is_dir():
                    continue
                candidate = f"{md.name}_{sd.name}"
                if rest == candidate:
                    model_name, split_name = md.name, sd.name
                    break
            if model_name:
                break
        if not model_name:
            continue

        per_class = []
        component_f1s, component_recalls, component_precs, component_accs = [], [], [], []
        succ = 0
        for comp in components:
            csv_path = OUTPUTS_DIR / f"metrics_{task}_{comp}_{model_name}_{split_name}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if "Class" not in df.columns:
                continue
            avg_row = df[df["Class"] == "Average"]
            wavg_row = df[df["Class"] == "Weighted Average"]
            if avg_row.empty:
                continue
            r = avg_row.iloc[0]
            f1_val  = _float(r, "F1 Score")
            rec_val = _float(r, "Recall")
            pre_val = _float(r, "Precision")
            acc_val = _float(r, "Accuracy")
            if f1_val  is not None: component_f1s.append(f1_val)
            if rec_val is not None: component_recalls.append(rec_val)
            if pre_val is not None: component_precs.append(pre_val)
            if acc_val is not None: component_accs.append(acc_val)
            # successful preds (take max across components — same inference run)
            sp = r.get("Successful Preds")
            if sp is not None and pd.notna(sp):
                succ = max(succ, int(sp))
            per_class.append({
                "class":     comp.capitalize(),
                "f1":        f1_val,
                "recall":    rec_val,
                "precision": pre_val,
                "jaccard":   _float(r, "Jaccard"),
                "accuracy":  acc_val,
            })

        if not component_f1s:
            continue

        avg_f1  = round(sum(component_f1s)    / len(component_f1s),    4)
        avg_rec = round(sum(component_recalls) / len(component_recalls), 4) if component_recalls else None
        avg_pre = round(sum(component_precs)   / len(component_precs),   4) if component_precs   else None
        avg_acc = round(sum(component_accs)    / len(component_accs),    4) if component_accs    else None

        # Per-component values indexed by name for the frontend triplet column set
        comp_f1  = {pc["class"].lower(): pc["f1"]       for pc in per_class}
        comp_acc = {pc["class"].lower(): pc["accuracy"]  for pc in per_class}
        results.append({
            "model": model_name,
            "split": split_name,
            "overall": {
                "accuracy":           avg_acc,
                "f1":                 avg_f1,
                "recall":             avg_rec,
                "precision":          avg_pre,
                "successful_preds":   succ,
                "instrument_f1":      comp_f1.get("instrument"),
                "verb_f1":            comp_f1.get("verb"),
                "target_f1":          comp_f1.get("target"),
                "instrument_acc":     comp_acc.get("instrument"),
                "verb_acc":           comp_acc.get("verb"),
                "target_acc":         comp_acc.get("target"),
            },
            "per_class": per_class,
            "metric_type": "triplet",
        })

    results.sort(key=lambda x: x["overall"]["f1"] or 0, reverse=True)
    return results


def _compute_bbox_leaderboard(task_dir: Path) -> list:
    """Compute Prec/Rec/F1/mIoU @IoU=0.5 from prediction JSON + GT COCO annotations."""
    from collections import defaultdict
    frame_boxes, _ = _load_endoscapes_coco()

    results = []
    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            pred_files = sorted(split_dir.glob("endoscapes-test-*.json"))
            if not pred_files:
                continue

            # Per-category accumulators
            cat_tp: dict[str, int] = defaultdict(int)
            cat_fp: dict[str, int] = defaultdict(int)
            cat_fn: dict[str, int] = defaultdict(int)
            cat_iou_sum: dict[str, float] = defaultdict(float)
            cat_iou_n: dict[str, int] = defaultdict(int)

            for pf in pred_files:
                # stem: "endoscapes-test-163_55175.jpg"  →  frame_key: "163_55175"
                frame_key = pf.stem.removeprefix("endoscapes-test-").removesuffix(".jpg")
                gt_list = frame_boxes.get(frame_key, [])
                raw = json.loads(pf.read_text())
                parsed = raw.get("parsed_bboxes", {})
                if not isinstance(parsed, dict):
                    continue
                if not parsed and not gt_list:
                    continue

                # GT grouped by category: {cat: [[x1,y1,x2,y2], ...]}
                gt_by_cat: dict[str, list] = defaultdict(list)
                for b in gt_list:
                    gt_by_cat[b["label"]].append([b["x0"], b["y0"], b["x1"], b["y1"]])

                # Pred grouped by category (handle both flat and list-of-list formats)
                pred_by_cat: dict[str, list] = {}
                for cat, coords in parsed.items():
                    if not coords:
                        pred_by_cat[cat] = []
                    elif isinstance(coords[0], list):
                        pred_by_cat[cat] = coords          # [[x1,y1,x2,y2], ...]
                    else:
                        pred_by_cat[cat] = [coords]        # [x1,y1,x2,y2]

                all_cats = set(gt_by_cat) | set(pred_by_cat)
                for cat in all_cats:
                    gts = gt_by_cat.get(cat, [])
                    preds = pred_by_cat.get(cat, [])

                    if not gts:
                        cat_fp[cat] += len(preds)
                        continue
                    if not preds:
                        cat_fn[cat] += len(gts)
                        continue

                    # Greedy matching: each GT/pred matched at most once
                    matched_gt: set[int] = set()
                    for pb in preds:
                        best_iou, best_gi = 0.0, -1
                        for gi, gb in enumerate(gts):
                            if gi in matched_gt:
                                continue
                            v = _iou(pb, gb)
                            if v > best_iou:
                                best_iou, best_gi = v, gi
                        if best_iou >= 0.5 and best_gi >= 0:
                            matched_gt.add(best_gi)
                            cat_tp[cat] += 1
                            cat_iou_sum[cat] += best_iou
                            cat_iou_n[cat] += 1
                        else:
                            cat_fp[cat] += 1
                    cat_fn[cat] += len(gts) - len(matched_gt)

            if not (cat_tp or cat_fp or cat_fn):
                continue

            all_cats = sorted(set(cat_tp) | set(cat_fp) | set(cat_fn))
            classes, precs, recs, f1s, ious = [], [], [], [], []
            for cat in all_cats:
                tp, fp, fn = cat_tp[cat], cat_fp[cat], cat_fn[cat]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                miou = cat_iou_sum[cat] / cat_iou_n[cat] if cat_iou_n[cat] > 0 else 0.0
                classes.append({"class": cat,
                                 "precision": round(p, 4), "recall": round(r, 4),
                                 "f1": round(f, 4), "jaccard": round(miou, 4)})
                precs.append(p); recs.append(r); f1s.append(f); ious.append(miou)

            n = len(all_cats)
            overall = {
                "accuracy":    round(sum(ious) / n, 4),   # mIoU@0.5
                "weighted_f1": round(sum(f1s) / n, 4),    # mF1@0.5
                "f1":          round(sum(f1s) / n, 4),
                "recall":      round(sum(recs) / n, 4),
                "precision":   round(sum(precs) / n, 4),
                "n": len(pred_files),
            }
            results.append({"model": model_dir.name, "split": split_dir.name,
                             "overall": overall, "per_class": classes,
                             "metric_type": "bbox"})
    return results


def _compute_phase_planning_leaderboard() -> list:
    """Compute accuracy per model for cholect50_phase_planning."""
    data = load_phase_planning_data()
    gt_map = {s["id"]: {"current_phase": s["phase"], "next_phase": s["next_phase"]}
              for s in data["samples"]}

    pred_root = OUTPUTS_DIR / "cholect50_phase_planning"
    if not pred_root.exists():
        return []

    results = []
    for model_dir in sorted(pred_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            pred_files = [p for p in split_dir.glob("*.json") if p.stem != "prompt"]
            if not pred_files:
                continue

            curr_correct = 0
            next_correct = 0
            both_correct = 0
            total = 0

            for pf in pred_files:
                sid = pf.stem
                gt = gt_map.get(sid)
                if gt is None:
                    continue
                raw = json.loads(pf.read_text())
                ans = raw.get("answer") or {}
                if isinstance(ans, dict):
                    pred_curr = (ans.get("current_phase") or "").strip()
                    pred_next = (ans.get("next_phase") or "").strip()
                    c = pred_curr == gt["current_phase"]
                    n = pred_next == gt["next_phase"]
                    curr_correct += int(c)
                    next_correct += int(n)
                    both_correct += int(c and n)
                    total += 1

            if total == 0:
                continue

            results.append({
                "model": model_dir.name,
                "split": split_dir.name,
                "overall": {
                    "curr_acc": round(curr_correct / total, 4),
                    "next_acc": round(next_correct / total, 4),
                    "both_acc": round(both_correct / total, 4),
                    "total": total,
                },
                "metric_type": "phase_planning",
            })

    results.sort(key=lambda x: x["overall"]["both_acc"], reverse=True)
    return results


def _compute_phase_easy_leaderboard() -> list:
    """Compute accuracy per model for vtrb_suturing_phase_predict_easy."""
    samples = _load_vtrb_phase_easy()
    gt_map = {s["id"]: s["answer"] for s in samples}
    subtask_map = {s["id"]: s["answer_text"] for s in samples}

    pred_root = OUTPUTS_DIR / "vtrb_suturing_phase_predict_easy"
    if not pred_root.exists():
        return []

    results = []
    for model_dir in sorted(pred_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            pred_files = [p for p in split_dir.glob("*.json") if p.stem != "prompt"]
            if not pred_files:
                continue

            correct = 0
            total = 0
            subtask_correct: dict[str, int] = {}
            subtask_total: dict[str, int] = {}

            for pf in pred_files:
                sid = pf.stem
                gt = gt_map.get(sid)
                if gt is None:
                    continue
                raw = json.loads(pf.read_text())
                pred = raw.get("answer", "")
                st = subtask_map.get(sid, "unknown")
                subtask_total[st] = subtask_total.get(st, 0) + 1
                total += 1
                if pred and pred.strip().upper() == gt:
                    correct += 1
                    subtask_correct[st] = subtask_correct.get(st, 0) + 1

            if total == 0:
                continue

            accuracy = round(correct / total, 4)
            per_class = [
                {
                    "class": st,
                    "accuracy": round(subtask_correct.get(st, 0) / subtask_total[st], 4),
                    "correct": subtask_correct.get(st, 0),
                    "total": subtask_total[st],
                }
                for st in sorted(subtask_total)
            ]

            results.append({
                "model": model_dir.name,
                "split": split_dir.name,
                "overall": {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                },
                "per_class": per_class,
                "metric_type": "phase_easy",
            })

    results.sort(key=lambda x: x["overall"]["accuracy"], reverse=True)
    return results


@app.get("/api/leaderboard")
def get_leaderboard(task: str = Query(...)):
    """Return per-model metrics for a task, parsed from metrics_*.csv files."""
    # JIGSAWS per-category skill tasks: compute from prediction JSONs
    _CAT_MAP = {
        "knot_tying_skill_assessment":     "Knot_Tying",
        "needle_passing_skill_assessment": "Needle_Passing",
        "suturing_skill_assessment":       "Suturing",
    }
    if task in _CAT_MAP:
        return _compute_jigsaws_category_leaderboard(_CAT_MAP[task])

    _lb_dir_map = {
        "vtrb_suturing_object_detection":   "vtrb_suturing_recognition",
        "vtrb_suturing_phase_recognition":  "vtrb_suturing_phase_predict_easy",
    }
    task_dir = OUTPUTS_DIR / _lb_dir_map.get(task, task)
    results = []
    if not task_dir.exists():
        return results

    # Skill tasks: root-level CSVs with a "Dimension" column (not "Class")
    skill_csvs = list(OUTPUTS_DIR.glob(f"metrics_{task}_*.csv"))
    if skill_csvs and "Dimension" in pd.read_csv(skill_csvs[0], nrows=0).columns:
        return _compute_skill_leaderboard(task)

    # Triplet tasks: separate instrument/verb/target CSVs
    if task == "cholect45_triplet_recognition":
        return _compute_triplet_leaderboard(task)

    # Bbox tasks: compute metrics on-the-fly (no CSV available)
    ann_type = TASK_DATASET_ANNOTATION.get(task, (None, None))[1]
    if ann_type == "bbox":
        if task == "vtrb_suturing_object_detection":
            return []  # No complete GT yet
        return _compute_bbox_leaderboard(task_dir)
    if ann_type == "phase_easy":
        return _compute_phase_easy_leaderboard()
    if ann_type == "phase_planning":
        return _compute_phase_planning_leaderboard()

    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            metrics_file = OUTPUTS_DIR / f"metrics_{task}_{model_dir.name}_{split_dir.name}.csv"
            if not metrics_file.exists():
                continue
            df = pd.read_csv(metrics_file)

            # Skip metrics files that don't have a "Class" column
            # (e.g. jigsaws_skill_assessment uses Accuracy/MAE/Spearman format)
            if "Class" not in df.columns:
                continue

            def _float(row, col):
                v = row.get(col)
                return round(float(v), 4) if v is not None and pd.notna(v) and v != "" else None

            avg  = df[df["Class"] == "Average"]
            wavg = df[df["Class"] == "Weighted Average"]
            per_class = df[~df["Class"].isin(["Average", "Weighted Average"])].copy()

            overall = {}
            if not avg.empty:
                r = avg.iloc[0]
                overall["accuracy"]        = _float(r, "Accuracy")
                overall["f1"]              = _float(r, "F1 Score")
                overall["recall"]          = _float(r, "Recall")
                overall["precision"]       = _float(r, "Precision")
                overall["successful_preds"] = int(r.get("Successful Preds") or 0)
            if not wavg.empty:
                r = wavg.iloc[0]
                overall["weighted_f1"]      = _float(r, "F1 Score")
                overall["weighted_recall"]  = _float(r, "Recall")
                overall["weighted_precision"] = _float(r, "Precision")

            classes = []
            for _, r in per_class.iterrows():
                classes.append({
                    "class":     str(r["Class"]),
                    "f1":        _float(r, "F1 Score"),
                    "recall":    _float(r, "Recall"),
                    "precision": _float(r, "Precision"),
                    "jaccard":   _float(r, "Jaccard"),
                })
            results.append({
                "model": model_dir.name,
                "split": split_dir.name,
                "overall": overall,
                "per_class": classes,
            })
    return results


@app.get("/api/models")
def get_models(task: str = Query(...)):
    """Return {model: [split, ...]} for available prediction outputs for a task."""
    _task_dir_map = {
        "vtrb_suturing_object_detection": "vtrb_suturing_recognition",
        "vtrb_suturing_phase_recognition": "vtrb_suturing_phase_predict_easy",
    }
    task_dir = OUTPUTS_DIR / _task_dir_map.get(task, task)
    if not task_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        splits = sorted(s.name for s in model_dir.iterdir() if s.is_dir())
        if splits:
            result[model_dir.name] = splits
    return result


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

        elif dataset == "Cholec80":
            frame_dir = DATASET_ROOT / "Cholec80" / "frames_25fps" / "test"
            if frame_dir.exists():
                vids = [d for d in frame_dir.iterdir() if d.is_dir()]
                count = sum(len(list(d.glob("*.jpg"))) for d in vids)
                videos = len(vids)
            else:
                count = 0; videos = 0
            unit = "frames"

        elif dataset == "CholecT45":
            rgb_dir = CHOLECT45_ROOT / "rgb"
            if rgb_dir.exists():
                test_vids = [rgb_dir / v for v in CHOLECT45_TEST_VIDEOS if (rgb_dir / v).exists()]
                count = sum(len(list(d.glob("*.png"))) for d in test_vids)
                videos = len(test_vids)
            else:
                count = 0; videos = 0
            unit = "frames"

        elif dataset == "CholecT50":
            if ann_type == "phase_planning":
                pp_data = load_phase_planning_data()
                count = len(pp_data["samples"])
                videos = len(set(s["video_id"] for s in pp_data["samples"]))
                unit = "samples"
            else:
                videos_dir = CHOLECT50_ROOT / "videos"
                if videos_dir.exists():
                    test_vids = [videos_dir / v for v in CHOLECT50_TEST_VIDEOS if (videos_dir / v).exists()]
                    count = sum(len(list(d.glob("*.png"))) for d in test_vids)
                    videos = len(test_vids)
                else:
                    count = 0; videos = 0
                unit = "frames"

        elif dataset == "endoscapes":
            frame_boxes, _ = _load_endoscapes_coco()
            count = sum(1 for v in frame_boxes.values() if v)
            unit = "frames"
            videos = len(endoscapes_annotated_test_frames())

        elif dataset in ("Knot_Tying", "Needle_Passing", "Suturing"):
            base = DATASET_ROOT / "JIGSAWS"
            meta = base / dataset / f"meta_file_{dataset}.txt"
            count = len([l for l in meta.read_text().splitlines()
                         if l.strip() and len(l.strip().split()) > 3]) if meta.exists() else 0
            unit = "videos"
            videos = count

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

        elif dataset == "vtrb_suturing":
            vids = vtrb_suturing_videos()
            count = sum(len(list((VTRB_SUTURING_DIR / "frames" / v).glob("*.png"))) for v in vids)
            unit = "frames"
            videos = len(vids)

        elif dataset == "vtrb_suturing_phase":
            samples = _load_vtrb_phase_easy()
            count = len(samples)
            unit = "frames"
            videos = len({s["folder"] for s in samples})
            # Override dataset name to match what get_tasks() parses for chip matching
            dataset = "vtrb_suturing"

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


def _jigsaws_skill_summary(category: str = None) -> dict:
    base = DATASET_ROOT / "JIGSAWS"
    cols = ["video_name","experience","grs","respect_for_tissue","suture_needle_handling",
            "time_and_motion","flow_of_operation","overall_performance","quality_of_final_product"]
    score_cols = cols[3:]
    videos = []
    categories = [category] if category else JIGSAWS_CATEGORIES
    for cat in categories:
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
    if ds in ("Knot_Tying", "Needle_Passing", "Suturing") and ann_type == "skill":
        return _jigsaws_skill_summary(category=ds)
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
    if dataset == "Cholec80":
        return cholec80_videos()
    if dataset == "CholecT45":
        return cholect45_videos()
    if dataset == "CholecT50":
        if task_type == "phase_planning":
            return phase_planning_videos()
        return cholect50_videos()
    return []


@app.get("/api/samples")
def get_samples(
    task: str = Query(...),
    video: str = Query(None),
    n: int = Query(12),
    seed: int = Query(42),
    model: list[str] = Query(default=[]),
    split: list[str] = Query(default=[]),
):
    dataset_ann = TASK_DATASET_ANNOTATION.get(task)
    if not dataset_ann:
        raise HTTPException(404, f"No data loader for task '{task}'")
    dataset, ann_type = dataset_ann

    if ann_type == "phase_easy":
        return []  # Handled by /api/vtrb_phase_easy

    if dataset == "HeiChole":
        frames = heichole_frames(video)
        if not frames:
            raise HTTPException(404, f"No frames found for video '{video}'")
        df = load_heichole_annotation(video, ann_type)

        pred_dirs = [
            (m, s, OUTPUTS_DIR / task / m / s)
            for m, s in zip(model, split)
            if (OUTPUTS_DIR / task / m / s).exists()
        ]

        # Restrict to frames that have predictions in ALL selected models (intersection)
        if pred_dirs:
            prefix = f"extracted_frames-{video}-"
            stem_sets = [
                {p.name[len(prefix):].removesuffix(".json") for p in pd.glob(f"{prefix}*.json")}
                for _, _, pd in pred_dirs
            ]
            common_stems = set.intersection(*stem_sets)
            frames_filtered = sorted(
                (f for f in frames if f.name in common_stems),
                key=lambda f: int(f.stem.split("_")[-1])
            )
            candidates = frames_filtered if n == 0 else frames_filtered[::max(1, len(frames_filtered) // n)][:n]
        else:
            candidates = frames if n == 0 else frames[::max(1, len(frames) // n)][:n]

        results = []
        for frame_path in candidates:
            frame_idx = int(frame_path.stem.split("_")[-1])
            label = get_label_for_frame(df, frame_idx, ann_type)
            image_url = f"/image/HeiChole/extracted_frames/{video}/{frame_path.name}"
            row = {"image_url": image_url, "frame_idx": frame_idx, "video": video, "label": label}
            if pred_dirs:
                pred_key = _image_url_to_pred_key(image_url)
                predictions = []
                for m, s, pd in pred_dirs:
                    pred_file = pd / f"{pred_key}.json"
                    if pred_file.exists():
                        raw = json.loads(pred_file.read_text())
                        pred = _normalize_prediction(raw, ann_type, label)
                        pred["model"] = m
                        pred["split"] = s
                        predictions.append(pred)
                if predictions:
                    row["predictions"] = predictions
            results.append(row)
        return results

    if dataset == "Cholec80":
        frames = cholec80_frames(video)
        if not frames:
            raise HTTPException(404, f"No frames found for Cholec80 video '{video}'")

        if ann_type == "phase":
            df = load_cholec80_phase_annotation(video)
        else:
            df = load_cholec80_tool_annotation(video)

        pred_dirs = [
            (m, s, OUTPUTS_DIR / task / m / s)
            for m, s in zip(model, split)
            if (OUTPUTS_DIR / task / m / s).exists()
        ]

        # Restrict to frames with predictions in ALL selected models (intersection)
        # Pred files: test-{video}-{frame_idx}.jpg.json  e.g. test-video43-0.jpg.json
        if pred_dirs:
            prefix = f"test-{video}-"
            stem_sets = [
                {p.name[len(prefix):].removesuffix(".jpg.json")
                 for p in pd.glob(f"{prefix}*.jpg.json")}
                for _, _, pd in pred_dirs
            ]
            common_stems = set.intersection(*stem_sets)
            frames_filtered = sorted(
                (f for f in frames if f.stem in common_stems),
                key=lambda f: int(f.stem)
            )
            candidates = frames_filtered if n == 0 else frames_filtered[::max(1, len(frames_filtered) // n)][:n]
        else:
            candidates = frames if n == 0 else frames[::max(1, len(frames) // n)][:n]

        results = []
        for frame_path in candidates:
            frame_idx = int(frame_path.stem)
            if ann_type == "phase":
                label = get_cholec80_phase_label(df, frame_idx)
            else:
                label = get_cholec80_tool_label(df, frame_idx)
            image_url = f"/image/Cholec80/frames_25fps/test/{video}/{frame_path.name}"
            row = {"image_url": image_url, "frame_idx": frame_idx, "video": video, "label": label}
            if pred_dirs:
                pred_key = _image_url_to_pred_key(image_url)
                predictions = []
                for m, s, pd in pred_dirs:
                    pred_file = pd / f"{pred_key}.json"
                    if pred_file.exists():
                        raw = json.loads(pred_file.read_text())
                        pred = _normalize_prediction(raw, ann_type, label, dataset="Cholec80")
                        pred["model"] = m
                        pred["split"] = s
                        predictions.append(pred)
                if predictions:
                    row["predictions"] = predictions
            results.append(row)
        return results

    if dataset == "endoscapes":
        frame_boxes, _ = _load_endoscapes_coco()
        annotated = sorted(k for k, v in frame_boxes.items() if v)

        pred_dirs = [
            (m, s, OUTPUTS_DIR / task / m / s)
            for m, s in zip(model, split)
            if (OUTPUTS_DIR / task / m / s).exists()
        ]

        # Restrict to frames with predictions in ALL selected models
        if pred_dirs:
            stem_sets = [
                {p.name.removesuffix(".json") for p in pd.glob("*.json")}
                for _, _, pd in pred_dirs
            ]
            common_stems = set.intersection(*stem_sets)
            annotated = [k for k in annotated if f"endoscapes-test-{k}.jpg" in common_stems]

        candidates = annotated if n == 0 else annotated[::max(1, len(annotated) // n)][:n]

        results = []
        for key in candidates:
            vid_id, frame_num = key.split("_", 1)
            image_url = f"/image/endoscapes/test/{key}.jpg"
            label = {"type": "bbox", "boxes": frame_boxes[key]}
            row = {"image_url": image_url, "frame_idx": int(frame_num), "video": vid_id, "label": label}

            if pred_dirs:
                pred_key = _image_url_to_pred_key(image_url)
                predictions = []
                for m, s, pd in pred_dirs:
                    pred_file = pd / f"{pred_key}.json"
                    if pred_file.exists():
                        raw = json.loads(pred_file.read_text())
                        pred = _normalize_prediction(raw, ann_type, label)
                        pred["model"] = m
                        pred["split"] = s
                        predictions.append(pred)
                if predictions:
                    row["predictions"] = predictions

            results.append(row)
        return results

    if dataset == "vtrb_suturing":
        recog_dir = OUTPUTS_DIR / "vtrb_suturing_recognition"

        pred_dirs = [
            (mdl, spl, recog_dir / mdl / spl)
            for mdl, spl in zip(model, split)
            if (recog_dir / mdl / spl).exists()
        ]

        def _vtrb_pred_stems(d: Path) -> set[str]:
            return {p.stem for p in d.glob("recog_*.json")}

        if pred_dirs:
            stem_sets = [_vtrb_pred_stems(d) for _, _, d in pred_dirs]
            common_stems = set.intersection(*stem_sets)
        else:
            common_stems: set[str] = set()
            if recog_dir.exists():
                for mdir in recog_dir.iterdir():
                    if mdir.is_dir():
                        for sdir in mdir.iterdir():
                            if sdir.is_dir():
                                common_stems |= _vtrb_pred_stems(sdir)

        # Parse stems: recog_{idx}_{folder}_ep{ep}_f{frame}
        parsed: list[tuple[int, str, str, str, str]] = []
        for stem in common_stems:
            m2 = re.match(r'recog_(\d+)_(.+)_ep(\d+)_f(\d+)$', stem)
            if not m2:
                continue
            parsed.append((int(m2.group(1)), stem, m2.group(2), m2.group(3), m2.group(4)))
        parsed.sort(key=lambda x: x[0])

        # Filter to frames where the image actually exists locally
        fc_root = VTRB_SUTURING_DIR / "frame_cache"
        parsed = [
            p for p in parsed
            if (fc_root / f"{p[2]}_episode_{int(p[3]):06d}" / f"{p[4]}.png").exists()
        ]

        candidates = parsed if n == 0 else parsed[::max(1, len(parsed) // n)][:n]

        results = []
        for idx, stem, folder_name, ep_str, frame_str in candidates:
            image_url = (
                f"/image/VTRB-Suturing-VQA/frame_cache/"
                f"{folder_name}_episode_{int(ep_str):06d}/{frame_str}.png"
            )
            label: dict = {"type": "bbox", "boxes": []}
            row: dict = {
                "image_url": image_url,
                "frame_idx": int(frame_str),
                "video": folder_name,
                "label": label,
            }
            if pred_dirs:
                predictions = []
                for mdl, spl, pd in pred_dirs:
                    pred_file = pd / f"{stem}.json"
                    if pred_file.exists():
                        raw = json.loads(pred_file.read_text())
                        objs = (raw.get("answer") or {}).get("objects", [])
                        parsed_pct: dict[str, list] = {}
                        for obj in objs:
                            cls = obj.get("class", "unknown")
                            bbox = obj.get("bbox", [])
                            if len(bbox) == 4:
                                parsed_pct.setdefault(cls, []).append(bbox)
                        predictions.append({
                            "type": "raw",
                            "data": {"parsed_bboxes_pct": parsed_pct},
                            "correct": None,
                            "model": mdl,
                            "split": spl,
                        })
                if predictions:
                    row["predictions"] = predictions
            results.append(row)
        return results

    if dataset == "CholecT45":
        frames = cholect45_frames(video)
        if not frames:
            raise HTTPException(404, f"No frames found for CholecT45 video '{video}'")
        ann = load_cholect45_triplet_annotation(video)

        pred_dirs = [
            (m, s, OUTPUTS_DIR / task / m / s)
            for m, s in zip(model, split)
            if (OUTPUTS_DIR / task / m / s).exists()
        ]

        if pred_dirs:
            prefix = f"rgb-{video}-"
            stem_sets = [
                {p.name[len(prefix):].removesuffix(".json") for p in pd.glob(f"{prefix}*.json")}
                for _, _, pd in pred_dirs
            ]
            common_stems = set.intersection(*stem_sets)
            frames_filtered = sorted(
                (f for f in frames if f.name in common_stems),
                key=lambda f: int(f.stem)
            )
            candidates = frames_filtered if n == 0 else frames_filtered[::max(1, len(frames_filtered) // n)][:n]
        else:
            candidates = frames if n == 0 else frames[::max(1, len(frames) // n)][:n]

        results = []
        for frame_path in candidates:
            frame_idx = int(frame_path.stem)
            active = ann.get(frame_idx, [])
            label = {"type": "multi", "labels": active}
            image_url = f"/image/CholecT45/rgb/{video}/{frame_path.name}"
            row = {"image_url": image_url, "frame_idx": frame_idx, "video": video, "label": label}
            if pred_dirs:
                pred_key = _image_url_to_pred_key(image_url)
                predictions = []
                for m, s, pd in pred_dirs:
                    pred_file = pd / f"{pred_key}.json"
                    if pred_file.exists():
                        raw = json.loads(pred_file.read_text())
                        names = _load_cholect45_triplet_names()
                        if "triplets" in raw:
                            # Binary vector format: [0,1,0,...] indexed by triplet ID
                            active_pred = [names[i] for i, v in enumerate(raw["triplets"]) if v and i < len(names)]
                        else:
                            # Component format: {"instrument": [...], "verb": [...], "target": [...]}
                            inst_preds = {s.lower() for s in raw.get("instrument", [])}
                            verb_preds = {s.lower() for s in raw.get("verb", [])}
                            tgt_preds  = {s.lower() for s in raw.get("target", [])}
                            active_pred = []
                            for n in names:
                                parts = n.split(" → ")
                                if len(parts) == 3 and parts[0] in inst_preds and parts[1] in verb_preds and parts[2] in tgt_preds:
                                    active_pred.append(n)
                        correct = set(active_pred) == set(active)
                        pred = {"type": "multi", "labels": active_pred, "correct": correct,
                                "model": m, "split": s}
                        predictions.append(pred)
                if predictions:
                    row["predictions"] = predictions
            results.append(row)
        return results

    if dataset == "CholecT50":
        frames = cholect50_frames(video)
        if not frames:
            raise HTTPException(404, f"No frames found for CholecT50 video '{video}'")
        ann = load_cholect50_annotation(video)
        ann_type = TASK_DATASET_ANNOTATION.get(task, (None, None))[1]

        pred_dirs = [
            (m, s, OUTPUTS_DIR / task / m / s)
            for m, s in zip(model, split)
            if (OUTPUTS_DIR / task / m / s).exists()
        ]

        if pred_dirs:
            prefix = f"videos-{video}-"
            stem_sets = [
                {p.name[len(prefix):].removesuffix(".json") for p in pd.glob(f"{prefix}*.json")}
                for _, _, pd in pred_dirs
            ]
            common_stems = set.intersection(*stem_sets)
            frames_filtered = sorted(
                (f for f in frames if f.name in common_stems),
                key=lambda f: int(f.stem)
            )
            candidates = frames_filtered if n == 0 else frames_filtered[::max(1, len(frames_filtered) // n)][:n]
        else:
            candidates = frames if n == 0 else frames[::max(1, len(frames) // n)][:n]

        results = []
        for frame_path in candidates:
            frame_idx = int(frame_path.stem)
            frame_ann = ann.get(frame_idx, {"phase": -1, "tools": []})
            if ann_type == "phase":
                phase_idx = frame_ann["phase"]
                phase_name = CHOLECT50_PHASE_LABELS[phase_idx] if 0 <= phase_idx < len(CHOLECT50_PHASE_LABELS) else str(phase_idx)
                label = {"type": "single", "label": phase_name, "index": phase_idx}
            else:
                label = {"type": "multi", "labels": frame_ann["tools"]}
            image_url = f"/image/CholecT50/videos/{video}/{frame_path.name}"
            row = {"image_url": image_url, "frame_idx": frame_idx, "video": video, "label": label}
            if pred_dirs:
                pred_key = _image_url_to_pred_key(image_url)
                predictions = []
                for m, s, pd in pred_dirs:
                    pred_file = pd / f"{pred_key}.json"
                    if pred_file.exists():
                        raw = json.loads(pred_file.read_text())
                        pred = _normalize_prediction(raw, ann_type, label, dataset="CholecT50")
                        pred["model"] = m
                        pred["split"] = s
                        predictions.append(pred)
                if predictions:
                    row["predictions"] = predictions
            results.append(row)
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


@app.get("/api/phase_planning_samples")
def get_phase_planning_samples(
    task: str = Query(...),
    video: str = Query(None),
    n: int = Query(0),
):
    data = load_phase_planning_data()
    samples = data["samples"]
    if video:
        samples = [s for s in samples if s["video_id"] == video]
    if n > 0 and n < len(samples):
        step = max(1, len(samples) // n)
        samples = samples[::step][:n]
    result = []
    for s in samples:
        frame_num = s["anchor_frame"]
        image_url = f"/image/CholecT50/videos/{s['video_id']}/{frame_num:06d}.png"
        result.append({**s, "image_url": image_url})
    return result


@app.get("/api/phase_planning_predictions")
def get_phase_planning_predictions():
    """Return per-sample model predictions: {sample_id: {model: {current_phase, next_phase}}}."""
    pred_root = OUTPUTS_DIR / "cholect50_phase_planning"
    result: dict = {}
    if not pred_root.exists():
        return result
    for model_dir in sorted(pred_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            for pf in split_dir.glob("*.json"):
                if pf.stem == "prompt":
                    continue
                raw = json.loads(pf.read_text())
                ans = raw.get("answer") or {}
                if isinstance(ans, dict):
                    sid = pf.stem
                    if sid not in result:
                        result[sid] = {}
                    result[sid][model_dir.name] = {
                        "current_phase": (ans.get("current_phase") or "").strip(),
                        "next_phase": (ans.get("next_phase") or "").strip(),
                    }
    return result


@app.get("/api/phase_planning_context")
def get_phase_planning_context(
    video_id: str = Query(...),
    anchor_frame: int = Query(...),
    total_frames: int = Query(...),
):
    """Return up to 10 prev frames (including anchor) + 5 next frames."""
    prev_start = max(0, anchor_frame - 9)
    prev_frames = [
        {
            "frame_idx": i,
            "image_url": f"/image/CholecT50/videos/{video_id}/{i:06d}.png",
            "is_anchor": i == anchor_frame,
        }
        for i in range(prev_start, anchor_frame + 1)
    ]
    next_end = min(total_frames - 1, anchor_frame + 5)
    next_frames = [
        {
            "frame_idx": i,
            "image_url": f"/image/CholecT50/videos/{video_id}/{i:06d}.png",
            "is_anchor": False,
        }
        for i in range(anchor_frame + 1, next_end + 1)
    ]
    return {"prev_frames": prev_frames, "next_frames": next_frames}


@app.get("/api/phase_timeline")
def get_phase_timeline(video: str = Query(...), dataset: str = Query("HeiChole")):
    """Return run-length-encoded phase segments for the full video timeline."""
    if dataset.lower() == "cholec80":
        df = load_cholec80_phase_annotation(video)
        if df.empty:
            raise HTTPException(404, f"No phase annotation for '{video}'")
        names = CHOLEC80_LABELS["phase"]["names"]
        display = CHOLEC80_LABELS["phase"]["display"]
        total = len(df)
        segments = []
        prev_phase = df.iloc[0]["phase"]
        run_start = 0
        for i, row in df.iterrows():
            phase = row["phase"]
            if phase != prev_phase:
                idx = names.index(prev_phase) if prev_phase in names else -1
                segments.append({
                    "label": display[idx] if idx >= 0 else prev_phase,
                    "color": PHASE_COLORS[idx] if 0 <= idx < len(PHASE_COLORS) else "#aaa",
                    "pct": round((i - run_start) / total * 100, 3),
                    "start_frame": int(df.iloc[run_start]["frame"]),
                    "end_frame": int(df.iloc[i - 1]["frame"]),
                })
                run_start = i
                prev_phase = phase
        idx = names.index(prev_phase) if prev_phase in names else -1
        segments.append({
            "label": display[idx] if idx >= 0 else prev_phase,
            "color": PHASE_COLORS[idx] if 0 <= idx < len(PHASE_COLORS) else "#aaa",
            "pct": round((total - run_start) / total * 100, 3),
            "start_frame": int(df.iloc[run_start]["frame"]),
            "end_frame": int(df.iloc[-1]["frame"]),
        })
        return segments

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
    if prompts:
        return prompts
    # Fallback: read prompt.txt files from output directories
    task_dir = OUTPUTS_DIR / task
    if not task_dir.exists():
        return {}
    result = {}
    for model_dir in sorted(task_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            prompt_file = split_dir / "prompt.txt"
            if prompt_file.exists():
                result[f"{model_dir.name}"] = prompt_file.read_text().strip()
    return result


@lru_cache(maxsize=1)
def _load_vtrb_phase_easy() -> list[dict]:
    """Load phase_predict_easy.json samples."""
    gt_file = VTRB_PHASE_EASY_DIR / "phase_predict_easy.json"
    if not gt_file.exists():
        return []
    with open(gt_file) as f:
        return json.load(f)["samples"]


@app.get("/api/vtrb_phase_easy")
def get_vtrb_phase_easy(subtask: str = Query(None)):
    """Return phase_easy samples with predictions per model, optionally filtered by subtask."""
    samples = _load_vtrb_phase_easy()

    # Discover subtasks: unique correct answer texts with counts
    all_samples = _load_vtrb_phase_easy()
    subtask_counts: dict[str, int] = {}
    for s in all_samples:
        subtask_counts[s["answer_text"]] = subtask_counts.get(s["answer_text"], 0) + 1
    subtasks = sorted(subtask_counts.keys())

    if subtask:
        samples = [s for s in samples if s["answer_text"] == subtask]

    # Load all model predictions
    pred_root = OUTPUTS_DIR / "vtrb_suturing_phase_predict_easy"
    model_preds: dict[str, dict[str, str]] = {}  # {model_split: {sample_id: answer}}
    if pred_root.exists():
        for model_dir in sorted(pred_root.iterdir()):
            if not model_dir.is_dir():
                continue
            for split_dir in sorted(model_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                key = f"{model_dir.name}/{split_dir.name}"
                model_preds[key] = {}
                for pf in split_dir.glob("*.json"):
                    if pf.stem == "prompt":
                        continue
                    sample_id = pf.stem  # e.g. 20260327_hyunjun_ep000_phase_easy
                    raw = json.loads(pf.read_text())
                    model_preds[key][sample_id] = raw.get("answer", "")

    result = []
    for s in samples:
        sid = s["id"]
        preds = []
        for model_split, id_map in model_preds.items():
            if sid in id_map:
                model, split = model_split.split("/", 1)
                pred_letter = id_map[sid]
                pred_text = s["choices"].get(pred_letter, pred_letter)
                correct = pred_letter == s["answer"]
                preds.append({
                    "model": model,
                    "split": split,
                    "answer_letter": pred_letter,
                    "answer_text": pred_text,
                    "correct": correct,
                })

        result.append({
            "id": sid,
            "question": s["question"],
            "choices": s["choices"],
            "answer": s["answer"],
            "answer_text": s["answer_text"],
            "frame_path": s["frame_path"],
            "image_url": f"/image/VTRB-Suturing-VQA/{s['frame_path']}",
            "predictions": preds,
        })

    return {"subtasks": subtasks, "subtask_counts": subtask_counts, "samples": result}


@app.get("/api/vtrb_few_shot")
def get_vtrb_few_shot():
    """Return few-shot ICL examples with GT boxes and prediction boxes."""
    fs_dir = VTRB_SUTURING_DIR / "recognition_few_shot"
    ann_file = fs_dir / "bbox_annotations.json"
    pred_dir = fs_dir / "predictions"

    ann: dict = {}
    if ann_file.exists():
        with open(ann_file) as f:
            ann = json.load(f)

    def _to_boxes(raw_list: list) -> list[dict]:
        return [
            {
                "label": b["class"],
                "color": VTRB_SUTURING_COLORS.get(b["class"], "#aaa"),
                "x0_pct": round(b["x_pct"], 2),
                "y0_pct": round(b["y_pct"], 2),
                "x1_pct": round(b["x_pct"] + b["w_pct"], 2),
                "y1_pct": round(b["y_pct"] + b["h_pct"], 2),
            }
            for b in raw_list
        ]

    # ICL examples: images in fs_dir root, boxes from bbox_annotations.json
    icl_examples = []
    for img_path in sorted(fs_dir.glob("*.png")):
        # filename: 20260328_hyunjun_ep004_000025.png
        # annotation key: 20260328_hyunjun_ep004/000025.png
        parts = img_path.stem.rsplit("_", 1)
        if len(parts) == 2:
            ann_key = f"{parts[0]}/{parts[1]}.png"
        else:
            ann_key = img_path.name
        boxes = _to_boxes(ann.get(ann_key, []))
        icl_examples.append({
            "image_url": f"/image/VTRB-Suturing-VQA/recognition_few_shot/{img_path.name}",
            "label": ann_key,
            "boxes": boxes,
        })

    # Prediction examples: json+png pairs in predictions/
    pred_examples = []
    if pred_dir.exists():
        for jf in sorted(pred_dir.glob("*.json")):
            raw = json.loads(jf.read_text())
            parsed = raw.get("parsed") or {}
            objs = parsed.get("objects", []) if isinstance(parsed, dict) else []
            pred_boxes: dict[str, list] = {}
            for obj in objs:
                cls = obj.get("class", "unknown")
                bbox = obj.get("bbox", [])
                if len(bbox) == 4:
                    pred_boxes.setdefault(cls, []).append(bbox)

            png = jf.with_suffix(".png")
            image_url = (
                f"/image/VTRB-Suturing-VQA/recognition_few_shot/predictions/{png.name}"
                if png.exists() else None
            )
            pred_examples.append({
                "id": raw.get("id", jf.stem),
                "image_url": image_url,
                "parsed_bboxes_pct": pred_boxes,
            })

    return {"icl_examples": icl_examples, "pred_examples": pred_examples}


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
