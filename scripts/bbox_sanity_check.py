"""
Visual sanity check for bounding box predictions.

Draws GT bboxes (green) and predicted bboxes (red) on test images.
Saves annotated images to outputs/bbox_sanity_check/{model_name}/.

Usage:
    python scripts/bbox_sanity_check.py --model Qwen2-VL --task endoscapes_object_detection --max_samples 5
    python scripts/bbox_sanity_check.py --eval_only  # just visualize existing predictions
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COLORS = {
    'cystic_plate': (255, 0, 0),      # blue
    'calot_triangle': (0, 255, 0),     # green
    'cystic_artery': (0, 0, 255),      # red
    'cystic_duct': (255, 255, 0),      # cyan
    'gallbladder': (255, 0, 255),      # magenta
    'tool': (0, 255, 255),             # yellow
}


def draw_bboxes(img, bboxes_dict, color_offset=0, label_prefix="", thickness=2):
    """Draw bounding boxes on image.
    bboxes_dict: {category: [[x,y,w,h], ...]} for GT (COCO format)
                 or {category: [[x1,y1,x2,y2], ...]} for preds (xyxy format)
    """
    for cat, bboxes in bboxes_dict.items():
        if cat in ('im_size_wh', 'raw_output', 'parsed_bboxes', 'coord_range'):
            continue
        base_color = COLORS.get(cat, (128, 128, 128))
        color = tuple(min(255, c + color_offset) for c in base_color)
        for bbox in bboxes:
            if len(bbox) != 4:
                continue
            x1, y1, x2_or_w, y2_or_h = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2_or_w, y2_or_h), color, thickness)
            label = f"{label_prefix}{cat}"
            cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def visualize_from_coco(pred_dir, dataset, out_dir, max_vis=10):
    """Load predictions and GT, draw side-by-side on images."""
    from pycocotools.coco import COCO

    os.makedirs(out_dir, exist_ok=True)
    img_dir = dataset.image_dir

    # Load COCO GT
    coco = COCO(os.path.join(img_dir, 'annotation_coco.json'))
    cat_map = {c['id']: c['name'] for c in coco.dataset['categories']}

    # Find prediction files
    pred_files = sorted(glob(os.path.join(pred_dir, 'endoscapes-test-*.json')))[:max_vis]
    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return

    for pred_file in pred_files:
        with open(pred_file) as f:
            pred_data = json.load(f)

        if isinstance(pred_data, str):
            print(f"  Skipping {os.path.basename(pred_file)}: {pred_data[:50]}")
            continue

        # Extract image filename: endoscapes-test-164_1950.jpg.json -> 164_1950.jpg
        basename = os.path.basename(pred_file)
        img_name = basename.replace('endoscapes-test-', '').replace('.json', '')
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Draw GT
        gt_img = img.copy()
        img_id_list = [i['id'] for i in coco.dataset['images'] if i['file_name'] == img_name]
        if img_id_list:
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id_list[0])):
                cat = cat_map[ann['category_id']]
                bx, by, bw, bh = [int(v) for v in ann['bbox']]
                color = COLORS.get(cat, (128, 128, 128))
                cv2.rectangle(gt_img, (bx, by), (bx + bw, by + bh), color, 3)
                cv2.putText(gt_img, cat, (bx, max(by - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw predictions (parsed_bboxes are in pixel [x1,y1,x2,y2])
        pred_img = img.copy()
        parsed = pred_data.get('parsed_bboxes', pred_data)
        if isinstance(parsed, dict):
            for cat, bbox_or_bboxes in parsed.items():
                if cat in ('raw_output', 'coord_range', 'im_size_wh'):
                    continue
                cat_base = cat.rstrip('0123456789')
                color = COLORS.get(cat_base, (0, 0, 255))
                # Normalize: single box or list of boxes
                if isinstance(bbox_or_bboxes, list) and len(bbox_or_bboxes) == 4 and all(isinstance(c, (int, float)) for c in bbox_or_bboxes):
                    bboxes = [bbox_or_bboxes]
                elif isinstance(bbox_or_bboxes, list) and all(isinstance(b, list) for b in bbox_or_bboxes):
                    bboxes = bbox_or_bboxes
                else:
                    continue
                for bbox in bboxes:
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(pred_img, cat, (x1, max(y1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Side by side
        combined = np.hstack([gt_img, pred_img])
        cv2.putText(combined, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(combined, "Pred", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out_path = os.path.join(out_dir, img_name.replace('.jpg', '.png'))
        cv2.imwrite(out_path, combined)
        print(f"  Saved: {out_path}")

        # Print raw output
        raw = pred_data.get('raw_output', '')
        if raw:
            print(f"    Raw: {str(raw)[:200]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen2-VL',
                        help='Model config name (e.g., Qwen2-VL, PaliGemma)')
    parser.add_argument('--task', type=str, default='endoscapes_object_detection')
    parser.add_argument('--max_samples', type=int, default=5)
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip inference, just visualize existing predictions')
    parser.add_argument('--exp_name', type=str, default='deleteme_for_pub')
    args = parser.parse_args()

    # Load dataset
    from vlmeval.config import data_map
    import yaml
    task_config_path = f'config/task/{args.task}.yaml'
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)
    task_config['max_samples'] = args.max_samples
    dataset = data_map[task_config['data']](config=task_config, split='test')
    print(f"Loaded {len(dataset)} samples")

    # Load model config to get model_name
    model_config_path = f'config/model/{args.model}.yaml'
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    model_name = model_config['name']

    if not args.eval_only:
        # Run inference
        from vlmeval.config import supported_VLM, model_map
        from vlmeval.prompts import get_prompts

        print(f"Loading model: {model_name}")
        model = supported_VLM[model_name]()
        if not hasattr(model, 'name'):
            model.name = model_name

        prompt = get_prompts(task_config['data_config']['data_dir'], args.task, model_name)
        print(f"Running inference on {len(dataset)} samples...")

        from omegaconf import OmegaConf
        task_cfg = OmegaConf.create(task_config)
        model_map['infer_data'](model, 'outputs/', args.exp_name, dataset, task_cfg, prompt, override_outputs=True)

    # Visualize — pred dir structure: outputs/{task}/{model}/{exp_name}/
    pred_dir = f'outputs/{args.task}/{model_name}/{args.exp_name}'
    out_dir = f'outputs/bbox_sanity_check/{model_name}'
    print(f"\nVisualizing predictions from: {pred_dir}")
    visualize_from_coco(pred_dir, dataset, out_dir, max_vis=args.max_samples)
    print(f"\nDone! Check {out_dir}/ for visualizations.")


if __name__ == '__main__':
    main()
