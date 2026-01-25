"""
Train LFADS on MC_Maze_Small dataset
Week 1 - ALPHA Track

Run with: conda activate lfads && python train_mc_maze_small.py
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

# Change to lfads-torch directory for config resolution
os.chdir(Path(__file__).parent / "lfads-torch")

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-week1"
DATASET_STR = "nlb_mc_maze_small"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_mc_maze_small"
RUN_DIR = Path("runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)

# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)

print(f"Training LFADS on {DATASET_STR}")
print(f"Output directory: {RUN_DIR.absolute()}")
print("=" * 50)

run_model(
    overrides={
        "datamodule": DATASET_STR,
        "model": DATASET_STR,
        "trainer.max_epochs": 50,  # Reduced for quick test
    },
    config_path="../configs/single.yaml",
)
