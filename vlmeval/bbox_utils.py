"""
Model-specific bounding box parsing utilities.

Each model outputs bounding boxes in a different format:
- PaliGemma: <loc####> tokens, range [0, 1023], order y0,x0,y1,x1
- Qwen2-VL / Qwen3-VL: (x1,y1),(x2,y2) format, range [0, 1000]
- InternVL3: [[x1,y1,x2,y2]] or (x1,y1),(x2,y2), range [0, 1000]

All parsers return bboxes as (x1, y1, x2, y2) in pixel coordinates.
"""
import re
import json


def parse_bbox_paligemma(raw_output, img_w, img_h):
    """
    Parse PaliGemma <loc####> tokens.
    Returns list of (category, [x1, y1, x2, y2]) in pixel coords.
    PaliGemma outputs: <loc0100><loc0200><loc0800><loc0900> = y0,x0,y1,x1 in [0,1023]
    Multiple objects separated by ';'.
    """
    results = []
    # PaliGemma outputs one detection per prompt call, may have multiple separated by ;
    for obj in raw_output.split(';'):
        loc_values = re.findall(r'<loc(\d{4})>', obj)
        if len(loc_values) != 4:
            continue
        y0, x0, y1, x1 = map(int, loc_values)
        # Convert from [0, 1023] to pixel coordinates
        px1 = x0 * img_w / 1024
        py1 = y0 * img_h / 1024
        px2 = x1 * img_w / 1024
        py2 = y1 * img_h / 1024
        results.append([px1, py1, px2, py2])
    return results


def parse_bbox_qwen(raw_output, img_w, img_h):
    """
    Parse Qwen2-VL / Qwen3-VL bbox output.
    Format after skip_special_tokens: (x1,y1),(x2,y2) in [0, 1000] range.
    May also output JSON if prompted correctly.
    """
    results = []

    # Try JSON first (if model was prompted to output JSON)
    json_results = _try_parse_json_bboxes(raw_output, img_w, img_h, coord_range=1000)
    if json_results:
        return json_results

    # Parse native format: (x1,y1),(x2,y2)
    # Pattern matches coordinate pairs like (234,156),(789,432)
    pattern = r'\((\d+),\s*(\d+)\)\s*,?\s*\((\d+),\s*(\d+)\)'
    matches = re.findall(pattern, raw_output)
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        px1 = x1 * img_w / 1000
        py1 = y1 * img_h / 1000
        px2 = x2 * img_w / 1000
        py2 = y2 * img_h / 1000
        results.append([px1, py1, px2, py2])

    return results


def parse_bbox_internvl(raw_output, img_w, img_h):
    """
    Parse InternVL3 bbox output.
    Format: [[x1,y1,x2,y2]] in [0, 1000] range, or same as Qwen format.
    """
    results = []

    # Try JSON first
    json_results = _try_parse_json_bboxes(raw_output, img_w, img_h, coord_range=1000)
    if json_results:
        return json_results

    # Try [[x1,y1,x2,y2]] format
    pattern = r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
    matches = re.findall(pattern, raw_output)
    for match in matches:
        x1, y1, x2, y2 = map(int, match)
        px1 = x1 * img_w / 1000
        py1 = y1 * img_h / 1000
        px2 = x2 * img_w / 1000
        py2 = y2 * img_h / 1000
        results.append([px1, py1, px2, py2])

    if results:
        return results

    # Fallback to Qwen-style (x1,y1),(x2,y2)
    return parse_bbox_qwen(raw_output, img_w, img_h)


def _try_parse_json_bboxes(raw_output, img_w, img_h, coord_range):
    """
    Try to parse JSON-formatted bbox output.
    Expected: {"category": [x1, y1, x2, y2], ...} or {"category": [[x1,y1,x2,y2], ...]}
    Coordinates assumed in [0, coord_range].
    Returns dict of {category: [[px1,py1,px2,py2], ...]} or None if parsing fails.
    """
    start = raw_output.find('{')
    end = raw_output.rfind('}')
    if start == -1 or end == -1:
        return None

    try:
        data = json.loads(raw_output[start:end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    results = {}
    for cat, coords in data.items():
        if cat in ('im_size_wh',):
            continue
        if isinstance(coords, list) and len(coords) == 4 and all(isinstance(c, (int, float)) for c in coords):
            # Single box: [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            px1 = x1 * img_w / coord_range
            py1 = y1 * img_h / coord_range
            px2 = x2 * img_w / coord_range
            py2 = y2 * img_h / coord_range
            results[cat] = [[px1, py1, px2, py2]]
        elif isinstance(coords, list) and all(isinstance(b, list) and len(b) == 4 for b in coords):
            # Multiple boxes: [[x1,y1,x2,y2], ...]
            results[cat] = []
            for box in coords:
                x1, y1, x2, y2 = box
                px1 = x1 * img_w / coord_range
                py1 = y1 * img_h / coord_range
                px2 = x2 * img_w / coord_range
                py2 = y2 * img_h / coord_range
                results[cat].append([px1, py1, px2, py2])
    return results if results else None


def get_bbox_parser(model_name):
    """Return the appropriate bbox parser for a given model."""
    model_lower = model_name.lower()
    if 'paligemma' in model_lower:
        return parse_bbox_paligemma
    elif 'qwen' in model_lower:
        return parse_bbox_qwen
    elif 'internvl' in model_lower:
        return parse_bbox_internvl
    else:
        # Default: try JSON parsing with [0, 1000] range
        return parse_bbox_qwen


def get_coord_range(model_name):
    """Return the coordinate range for a given model."""
    model_lower = model_name.lower()
    if 'paligemma' in model_lower:
        return 1024
    else:
        return 1000
