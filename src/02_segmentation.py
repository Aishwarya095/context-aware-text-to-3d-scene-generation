"""
Module 2: Object Segmentation (Segment Anything Model - SAM)
Isolates individual objects from the 2D scene layout and generates binary masks and transparent cutouts.
"""

import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

from config import MASKS_DIR, DEVICE, SAM_MODEL_TYPE, SAM_CHECKPOINT_PATH


def segment_scene_objects(
    image_path: Path,
    output_dir: Path = MASKS_DIR,
    min_mask_area: int = 1500
) -> list:
    """
    Segments the scene image into discrete object masks.
    Returns metadata list containing mask paths, bounding boxes, and object labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    segmented_objects = []

    print(f"[*] Segmenting objects from {image_path}...")

    if SAM_AVAILABLE and SAM_CHECKPOINT_PATH.exists():
        print(f"[*] Running SAM automatic mask generator on {DEVICE}...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH))
        sam.to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=min_mask_area
        )
        masks = mask_generator.generate(img_rgb)
    else:
        print("[!] SAM checkpoint not detected; using contour-based adaptive segmentation fallback...")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_mask_area:
                m = np.zeros((h, w), dtype=bool)
                cv2.drawContours(m.astype(np.uint8), [cnt], -1, 1, -1)
                x, y, bw, bh = cv2.boundingRect(cnt)
                masks.append({
                    "segmentation": m,
                    "bbox": [x, y, bw, bh],
                    "area": area
                })

    # Sort masks by area descending
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    for idx, mask_info in enumerate(masks[:15]):  # Keep top salient objects
        mask = mask_info["segmentation"]
        x, y, bw, bh = [int(v) for v in mask_info["bbox"]]

        mask_filename = output_dir / f"mask_{idx:02d}.png"
        crop_filename = output_dir / f"crop_{idx:02d}.png"

        # Save binary mask
        mask_uint8 = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(mask_filename), mask_uint8)

        # Extract RGBA cutout
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask_uint8
        crop = rgba[y:y+bh, x:x+bw]
        cv2.imwrite(str(crop_filename), crop)

        object_entry = {
            "id": idx,
            "mask_path": str(mask_filename),
            "crop_path": str(crop_filename),
            "bbox_2d": [x, y, bw, bh],
            "area": float(mask_info["area"]),
            "centroid_2d": [x + bw // 2, y + bh // 2]
        }
        segmented_objects.append(object_entry)

    # Save summary metadata
    meta_path = output_dir / "segmentation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(segmented_objects, f, indent=2)

    print(f"[+] Successfully isolated {len(segmented_objects)} objects into {output_dir}")
    return segmented_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment objects from 2D scene image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input scene image")
    parser.add_argument("--out_dir", type=str, default=str(MASKS_DIR), help="Output directory")
    args = parser.parse_args()

    segment_scene_objects(Path(args.image), Path(args.out_dir))

