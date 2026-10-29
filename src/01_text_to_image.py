"""
Module 1: Text-to-Image Generation (FLUX Diffusion Model)
Synthesizes a high-fidelity 2D intermediate scene layout from a natural language prompt.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import torch

try:
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from config import OUTPUTS_DIR, DEVICE, DTYPE, FLUX_MODEL_ID, DEFAULT_SCENE_PROMPT


def generate_scene_image(
    prompt: str = DEFAULT_SCENE_PROMPT,
    output_path: Path = None,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42
) -> Path:
    """
    Generates a 2D scene layout from a natural language description.
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "generated_scene.png"

    print(f"[*] Generating 2D scene layout with prompt:\n    '{prompt}'")
    print(f"[*] Device: {DEVICE}, Model: {FLUX_MODEL_ID}")

    if not DIFFUSERS_AVAILABLE or DEVICE == "cpu":
        print("[!] Note: Running in lightweight/CPU mode or diffusers missing.")
        print(f"[*] Creating baseline intermediate scene image at: {output_path}")
        img = Image.new("RGB", (width, height), color=(220, 235, 235))
        img.save(output_path)
        return output_path

    # Load FLUX Pipeline
    pipeline = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=DTYPE
    )
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale
    ).images[0]

    image.save(output_path)
    print(f"[+] Scene image successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D scene image from prompt.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_SCENE_PROMPT, help="Scene text description")
    parser.add_argument("--out", type=str, default=None, help="Output image file path")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    args = parser.parse_args()

    out_file = Path(args.out) if args.out else None
    generate_scene_image(prompt=args.prompt, output_path=out_file, num_inference_steps=args.steps)

