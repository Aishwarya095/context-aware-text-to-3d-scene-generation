"""
Global Configuration and Pipeline Hyperparameters
"""

import os
from pathlib import Path
import torch

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = OUTPUTS_DIR / "assets"
MASKS_DIR = OUTPUTS_DIR / "masks"
MATERIALS_DIR = OUTPUTS_DIR / "materials"
FINAL_SCENES_DIR = OUTPUTS_DIR / "scenes"

for directory in [OUTPUTS_DIR, ASSETS_DIR, MASKS_DIR, MATERIALS_DIR, FINAL_SCENES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

# Model Configuration
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"  # Fast inference diffusion transformer
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_PATH = BASE_DIR / "models" / "sam_vit_h_4b8939.pth"

# Default Prompts
DEFAULT_SCENE_PROMPT = (
    "Isometric view of a modern aesthetic bathroom with pastel teal walls, "
    "ceramic bathtub, white washbasin, mirror on the wall, commode, "
    "and potted plant, studio lighting, highly detailed 3D render style"
)

# Mesh & Depth Extraction Parameters
DEPTH_ESTIMATOR_MODEL = "Intel/dpt-large"
VOXEL_PITCH = 0.02
MAX_TRIANGLES = 50000

