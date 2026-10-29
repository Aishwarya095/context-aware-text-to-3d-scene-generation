"""
Module 6: Scene Composition & Rendering (Blender Python - bpy)
Reads scene_graph.json, sets up walls/floors, imports 3D meshes, arranges objects, and renders the isometric 3D scene.

Usage (Headless Blender):
    blender --background --python src/06_blender_compose.py -- --scene_json outputs/scene_graph.json --out outputs/final_render.png
"""

import sys
import json
from pathlib import Path

# Note: bpy is available when executed inside Blender
try:
    import bpy
    import mathutils
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False


def reset_blender_scene():
    """Removes all default objects (cubes, cameras, lights)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_room_enclosure(width=3.0, depth=3.0, height=2.6):
    """
    Creates an open isometric room (floor + back wall + left wall).
    """
    # 1. Floor
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (width, depth, 1.0)

    # 2. Back Wall (along X)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, depth / 2.0, height / 2.0))
    back_wall = bpy.context.active_object
    back_wall.name = "Back_Wall"
    back_wall.rotation_euler = (1.5708, 0, 0)
    back_wall.scale = (width, height, 1.0)

    # 3. Left Wall (along Y)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-width / 2.0, 0, height / 2.0))
    left_wall = bpy.context.active_object
    left_wall.name = "Left_Wall"
    left_wall.rotation_euler = (0, 1.5708, 0)
    left_wall.scale = (height, depth, 1.0)


def setup_isometric_camera_and_lighting(width=3.0, depth=3.0, height=2.6):
    """
    Sets up an isometric orthographic camera and soft studio lighting.
    """
    # Isometric Camera
    cam_data = bpy.data.cameras.new("IsometricCamera")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max(width, depth) * 1.6
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Position camera diagonally for true isometric projection
    cam_obj.location = (5.0, -5.0, 4.5)
    cam_obj.rotation_euler = (0.9553, 0.0, 0.7854)

    # Key Sun / Directional Light
    light_data = bpy.data.lights.new(name="SunLight", type='SUN')
    light_data.energy = 3.5
    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (4.0, -3.0, 6.0)
    light_obj.rotation_euler = (0.8, 0.2, 0.6)

    # Fill Ambient Light
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.9, 0.95, 1.0, 1.0)
        bg.inputs[1].default_value = 0.8


def compose_scene_from_json(scene_json_path: str, render_output_path: str = None):
    """
    Parses scene graph and places each object.
    """
    if not BPY_AVAILABLE:
        print("[!] bpy not available. Please run this script with Blender: blender --background --python ...")
        return

    reset_blender_scene()

    with open(scene_json_path, "r") as f:
        scene_data = json.load(f)

    room = scene_data.get("room_dimensions", {"width": 3.0, "depth": 3.0, "height": 2.6})
    build_room_enclosure(room["width"], room["depth"], room["height"])
    setup_isometric_camera_and_lighting(room["width"], room["depth"], room["height"])

    # Import objects
    for item in scene_data.get("objects", []):
        mesh_path = item["mesh_file"]
        transform = item["transform"]

        if not Path(mesh_path).exists():
            print(f"[!] Warning: Mesh file {mesh_path} not found. Skipping.")
            continue

        if mesh_path.endswith(".obj"):
            bpy.ops.wm.obj_import(filepath=mesh_path)
        elif mesh_path.endswith(".glb") or mesh_path.endswith(".gltf"):
            bpy.ops.import_scene.gltf(filepath=mesh_path)

        obj = bpy.context.selected_objects[0]
        obj.name = item.get("category", "SceneObject")

        # Apply position, rotation, scale
        obj.location = transform["position"]
        obj.rotation_euler = [r * (3.14159 / 180.0) for r in transform["rotation_euler_deg"]]
        obj.scale = transform["scale"]

    # Render settings
    if render_output_path:
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = render_output_path
        bpy.context.scene.render.resolution_x = 1024
        bpy.context.scene.render.resolution_y = 1024
        bpy.ops.render.render(write_still=True)
        print(f"[+] Final isometric 3D render saved to: {render_output_path}")


if __name__ == "__main__":
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        scene_json = "outputs/scene_graph.json"
        out_render = "outputs/final_render.png"

        for i, arg in enumerate(args):
            if arg == "--scene_json" and i + 1 < len(args):
                scene_json = args[i + 1]
            elif arg == "--out" and i + 1 < len(args):
                out_render = args[i + 1]

        compose_scene_from_json(scene_json, out_render)
    else:
        print("[*] Run inside Blender with: blender --background --python src/06_blender_compose.py -- --scene_json ...")

