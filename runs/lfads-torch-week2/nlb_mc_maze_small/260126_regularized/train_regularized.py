"""
Train LFADS on MC_Maze_Small with Proper Regularization
Week 2 - ALPHA Track

The Week 1 run used zero KL/L2 regularization (all scales = 0.0 because
they were marked "sampled" for PBT). This resulted in the model memorizing
training data without learning meaningful latent structure.

This script sets reasonable regularization values:
- kl_ic_scale: penalizes initial condition posterior diverging from prior
- kl_co_scale: penalizes controller output posterior
- l2_gen_scale: L2 on generator GRU weights
- l2_con_scale: L2 on controller GRU weights

Regularization ramps linearly from epoch l2/kl_start_epoch to
l2/kl_increase_epoch, so we need enough epochs to get past the ramp.

Run with: conda activate lfads && python train_regularized.py
"""
import os
import shutil
from datetime import datetime
from pathlib import Path

# Change to lfads-torch directory for config resolution
os.chdir(Path(__file__).parent / "lfads-torch")

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-week2"
DATASET_STR = "nlb_mc_maze_small"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_regularized"
RUN_DIR = Path("runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True

# Regularization values (literature-informed defaults for NLB datasets)
KL_IC_SCALE = 1e-4       # Initial condition KL penalty
KL_CO_SCALE = 1e-4       # Controller output KL penalty
L2_GEN_SCALE = 1e-4      # Generator L2 weight decay
L2_CON_SCALE = 1e-4      # Controller L2 weight decay
KL_INCREASE_EPOCH = 80   # Ramp KL from 0 to full over this many epochs
L2_INCREASE_EPOCH = 80   # Ramp L2 from 0 to full over this many epochs

# Training length - need 200+ epochs to get well past regularization ramp
MAX_EPOCHS = 250
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)

# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)

print(f"Training LFADS on {DATASET_STR} (WITH regularization)")
print(f"Output directory: {RUN_DIR.absolute()}")
print("=" * 60)
print(f"Regularization settings:")
print(f"  kl_ic_scale:      {KL_IC_SCALE}")
print(f"  kl_co_scale:      {KL_CO_SCALE}")
print(f"  l2_gen_scale:     {L2_GEN_SCALE}")
print(f"  l2_con_scale:     {L2_CON_SCALE}")
print(f"  kl_increase_epoch: {KL_INCREASE_EPOCH}")
print(f"  l2_increase_epoch: {L2_INCREASE_EPOCH}")
print(f"  max_epochs:       {MAX_EPOCHS}")
print("=" * 60)

run_model(
    overrides={
        "datamodule": DATASET_STR,
        "model": DATASET_STR,
        "trainer.max_epochs": MAX_EPOCHS,
        # Regularization
        "model.kl_ic_scale": KL_IC_SCALE,
        "model.kl_co_scale": KL_CO_SCALE,
        "model.l2_gen_scale": L2_GEN_SCALE,
        "model.l2_con_scale": L2_CON_SCALE,
        "model.kl_increase_epoch": KL_INCREASE_EPOCH,
        "model.l2_increase_epoch": L2_INCREASE_EPOCH,
    },
    config_path="../configs/single.yaml",
)
