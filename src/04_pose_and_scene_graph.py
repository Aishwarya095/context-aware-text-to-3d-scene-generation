"""
Module 4: 6D Pose Translation & Scene Graph Generation
Maps 2D pixel coordinates to 3D world space, optimizes for collision avoidance, and outputs scene_graph.json.
"""

import argparse
import json
import math
from pathlib import Path
import numpy as np

from config import OUTPUTS_DIR


def estimate_room_bounds(room_type: str = "bathroom") -> dict:
    """
    Returns standard canonical room dimensions in meters (width, depth, height).
    """
    if "bath" in room_type.lower():
        return {"width": 3.0, "depth": 3.0, "height": 2.6}
    elif "kitchen" in room_type.lower():
        return {"width": 4.5, "depth": 4.0, "height": 2.8}
    else:
        return {"width": 4.0, "depth": 4.0, "height": 2.6}


def check_and_resolve_collisions(objects_3d: list, min_distance: float = 0.3) -> list:
    """
    Applies simple 2D horizontal repulsive force relaxation to prevent overlapping objects.
    """
    n = len(objects_3d)
    positions = np.array([obj["transform"]["position"] for obj in objects_3d])

    for iteration in range(10):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i][:2] - positions[j][:2]
                dist = np.linalg.norm(diff)
                if dist < min_distance and dist > 1e-4:
                    overlap = (min_distance - dist) * 0.5
                    direction = diff / dist
                    positions[i][:2] += direction * overlap
                    positions[j][:2] -= direction * overlap
                    moved = True
        if not moved:
            break

    for idx, obj in enumerate(objects_3d):
        obj["transform"]["position"] = [round(float(v), 3) for v in positions[idx]]

    return objects_3d


def build_scene_graph(
    assets_meta_path: Path,
    image_width: int = 1024,
    image_height: int = 1024,
    room_type: str = "bathroom",
    output_json_path: Path = None
) -> dict:
    """
    Constructs a complete 3D scene graph with translation, rotation, scale, and room boundaries.
    """
    if output_json_path is None:
        output_json_path = OUTPUTS_DIR / "scene_graph.json"

    with open(assets_meta_path, "r") as f:
        assets = json.load(f)

    room_dim = estimate_room_bounds(room_type)
    objects_3d = []

    print(f"[*] Building 3D scene graph for room type: {room_type}")

    for asset in assets:
        # Map 2D pixel coordinates to 3D ground plane (X: left-right, Y: floor-ceiling, Z: depth)
        cx_2d, cy_2d = asset["centroid_2d"]
        bbox_w, bbox_h = asset["bbox_2d"][2], asset["bbox_2d"][3]

        norm_x = (cx_2d / float(image_width) - 0.5) * room_dim["width"]
        norm_z = (cy_2d / float(image_height) - 0.5) * room_dim["depth"]

        # Approximate physical object scale based on 2D bounding box
        scale_x = max(0.25, (bbox_w / float(image_width)) * room_dim["width"] * 1.2)
        scale_y = max(0.25, (bbox_h / float(image_height)) * room_dim["height"] * 1.2)
        scale_z = max(0.25, min(scale_x, scale_y) * 0.8)

        # Place object on the floor (Y = scale_y / 2) unless it's a wall fixture
        pos_y = scale_y / 2.0
        if cy_2d < image_height * 0.35:  # Wall-mounted object heuristic (mirrors, lights, cabinets)
            pos_y = room_dim["height"] * 0.65

        obj_entry = {
            "id": asset["id"],
            "mesh_file": asset["mesh_path"],
            "category": f"object_{asset['id']:02d}",
            "transform": {
                "position": [round(norm_x, 3), round(pos_y, 3), round(norm_z, 3)],
                "rotation_euler_deg": [0.0, 0.0, 0.0],
                "scale": [round(scale_x, 3), round(scale_y, 3), round(scale_z, 3)]
            }
        }
        objects_3d.append(obj_entry)

    # Enforce collision-free placement
    resolved_objects = check_and_resolve_collisions(objects_3d)

    scene_graph = {
        "scene_name": f"ContextAware_3D_{room_type.capitalize()}",
        "room_dimensions": room_dim,
        "environment": {
            "floor_material": "materials/floor_pbr",
            "wall_material": "materials/wall_pbr"
        },
        "camera_preset": {
            "type": "isometric",
            "fov": 45,
            "position": [4.5, 4.0, 4.5],
            "target": [0.0, 0.5, 0.0]
        },
        "objects": resolved_objects
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(scene_graph, f, indent=2)

    print(f"[+] Scene graph saved to: {output_json_path}")
    return scene_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D scene graph from asset metadata.")
    parser.add_argument("--assets_meta", type=str, required=True, help="Path to assets_metadata.json")
    parser.add_argument("--room_type", type=str, default="bathroom", help="Room category (bathroom, kitchen)")
    parser.add_argument("--out", type=str, default=None, help="Output scene_graph.json path")
    args = parser.parse_args()

    out_p = Path(args.out) if args.out else None
    build_scene_graph(Path(args.assets_meta), room_type=args.room_type, output_json_path=out_p)

