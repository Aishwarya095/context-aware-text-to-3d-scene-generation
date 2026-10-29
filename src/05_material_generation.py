"""
Module 5: Environment Texturing & PBR Material Generation
Generates physically based rendering (PBR) texture maps (Albedo, Normal, Roughness, ORM) for walls and floors.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from config import MATERIALS_DIR


def generate_normal_map_from_albedo(albedo_rgb: np.ndarray, strength: float = 2.0) -> np.ndarray:
    """
    Computes a tangent space normal map using Sobel image gradients.
    """
    gray = cv2.cvtColor(albedo_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    normal_x = -sobel_x * strength
    normal_y = -sobel_y * strength
    normal_z = np.ones_like(gray)

    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-6
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    # Map from [-1, 1] to [0, 255]
    r = ((normal_x * 0.5 + 0.5) * 255).astype(np.uint8)
    g = ((normal_y * 0.5 + 0.5) * 255).astype(np.uint8)
    b = ((normal_z * 0.5 + 0.5) * 255).astype(np.uint8)

    return np.dstack([r, g, b])


def generate_pbr_material_set(
    base_color: tuple,
    material_name: str,
    output_dir: Path = MATERIALS_DIR,
    texture_size: int = 512,
    roughness_val: int = 140
) -> dict:
    """
    Generates a full PBR map set (Albedo, Normal, Roughness, ORM) for a surface material.
    """
    mat_folder = output_dir / material_name
    mat_folder.mkdir(parents=True, exist_ok=True)

    # 1. Albedo Map
    albedo = np.full((texture_size, texture_size, 3), base_color, dtype=np.uint8)
    # Add subtle organic noise texture
    noise = np.random.normal(0, 4, (texture_size, texture_size, 3)).astype(np.int16)
    albedo = np.clip(albedo.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Normal Map
    normal = generate_normal_map_from_albedo(albedo)

    # 3. Roughness Map
    roughness = np.full((texture_size, texture_size), roughness_val, dtype=np.uint8)

    # 4. ORM Map (Occlusion, Roughness, Metallic packed into RGB)
    occlusion = np.full((texture_size, texture_size), 255, dtype=np.uint8)
    metallic = np.full((texture_size, texture_size), 10, dtype=np.uint8)
    orm = np.dstack([occlusion, roughness, metallic])

    paths = {
        "albedo": str(mat_folder / "albedo.png"),
        "normal": str(mat_folder / "normal.png"),
        "roughness": str(mat_folder / "roughness.png"),
        "orm": str(mat_folder / "orm.png")
    }

    cv2.imwrite(paths["albedo"], cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["normal"], cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
    cv2.imwrite(paths["roughness"], roughness)
    cv2.imwrite(paths["orm"], cv2.cvtColor(orm, cv2.COLOR_RGB2BGR))

    print(f"[+] Generated PBR material set: '{material_name}' in {mat_folder}")
    return paths


def create_default_scene_materials(output_dir: Path = MATERIALS_DIR):
    """
    Generates standard wall and floor materials matching the aesthetic described in the project.
    """
    # Pastel Teal Wall Material
    generate_pbr_material_set(
        base_color=(125, 185, 180),
        material_name="wall_teal",
        output_dir=output_dir,
        roughness_val=160
    )
    # White Ceramic Tile Floor Material
    generate_pbr_material_set(
        base_color=(235, 235, 240),
        material_name="floor_tile",
        output_dir=output_dir,
        roughness_val=80
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PBR material textures.")
    parser.add_argument("--out_dir", type=str, default=str(MATERIALS_DIR), help="Output directory")
    args = parser.parse_args()

    create_default_scene_materials(Path(args.out_dir))

