"""
End-to-End Orchestrator Pipeline for Context-Aware Text-to-3D Scene Generation
Executes the cascaded multi-stage pipeline:
  Stage 1: Text-to-Image Generation (FLUX)
  Stage 2: Object Segmentation (SAM)
  Stage 3: 3D Asset Reconstruction (SAM-3D / Monocular Lifting)
  Stage 4: Pose Translation & Scene Graph (Collision-free layout)
  Stage 5: Material Generation (PBR Textures)
  Stage 6: Scene Composition (JSON Scene Graph export)
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_SCENE_PROMPT, OUTPUTS_DIR, MASKS_DIR, ASSETS_DIR, MATERIALS_DIR
import importlib

text_to_image = importlib.import_module("01_text_to_image")
segmentation = importlib.import_module("02_segmentation")
asset_reconstruction = importlib.import_module("03_asset_reconstruction")
pose_scene_graph = importlib.import_module("04_pose_and_scene_graph")
material_gen = importlib.import_module("05_material_generation")


def run_full_pipeline(
    prompt: str = DEFAULT_SCENE_PROMPT,
    room_type: str = "bathroom",
    output_dir: Path = OUTPUTS_DIR
):
    start_time = time.time()
    print("=" * 70)
    print(" Context-Aware Text-to-3D Scene Generation Pipeline ")
    print("=" * 70)
    print(f"Input Prompt: '{prompt}'")
    print(f"Room Category: '{room_type}'")
    print(f"Output Root: {output_dir}\n")

    # Stage 1: Text-to-Image
    print("\n--- [Stage 1/6] Text-to-Image Generation ---")
    scene_image_path = output_dir / "01_scene_layout.png"
    text_to_image.generate_scene_image(prompt=prompt, output_path=scene_image_path)

    # Stage 2: Segmentation
    print("\n--- [Stage 2/6] Object Segmentation (SAM) ---")
    segmented_objects = segmentation.segment_scene_objects(
        image_path=scene_image_path,
        output_dir=MASKS_DIR
    )

    # Stage 3: 3D Mesh Reconstruction
    print("\n--- [Stage 3/6] 3D Asset Reconstruction (2D-to-3D Lifting) ---")
    meta_path = MASKS_DIR / "segmentation_metadata.json"
    reconstructed_assets = asset_reconstruction.reconstruct_all_assets(
        metadata_json=meta_path,
        output_dir=ASSETS_DIR
    )

    # Stage 4: 6D Pose Translation & Scene Graph
    print("\n--- [Stage 4/6] Pose Translation & Scene Graph Generation ---")
    assets_meta_path = ASSETS_DIR / "assets_metadata.json"
    scene_graph_path = output_dir / "scene_graph.json"
    scene_graph = pose_scene_graph.build_scene_graph(
        assets_meta_path=assets_meta_path,
        room_type=room_type,
        output_json_path=scene_graph_path
    )

    # Stage 5: Environment Material Generation
    print("\n--- [Stage 5/6] Environment PBR Material Generation ---")
    material_gen.create_default_scene_materials(output_dir=MATERIALS_DIR)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[+] Pipeline execution complete in {elapsed:.2f} seconds!")
    print(f"[+] Reconstructed Assets: {len(reconstructed_assets)}")
    print(f"[+] Scene Graph JSON: {scene_graph_path}")
    print(f"[+] Next Step: Render in Blender with:")
    print(f"    blender --background --python src/06_blender_compose.py -- --scene_json {scene_graph_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the text-to-3D scene generation pipeline.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_SCENE_PROMPT, help="Natural language scene description")
    parser.add_argument("--room_type", type=str, default="bathroom", choices=["bathroom", "kitchen", "bedroom", "living_room"])
    parser.add_argument("--out_dir", type=str, default=str(OUTPUTS_DIR), help="Root outputs directory")
    args = parser.parse_args()

    run_full_pipeline(prompt=args.prompt, room_type=args.room_type, output_dir=Path(args.out_dir))

