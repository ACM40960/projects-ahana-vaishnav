
from pathlib import Path
import torch
import torch.nn as nn

BASE_FOLDER = Path(__file__).resolve().parent.parent
VIT_WEIGHTS = BASE_FOLDER / "models" / "vit_transfer_learning" / "final_vit_transfer.pt"
OUT_PATH = BASE_FOLDER / "deployment" / "artifacts" / "vit_head_weights.pt"

full_state = torch.load(VIT_WEIGHTS, map_location="cpu")

# Only keep the classifier head keys (everything else is frozen backbone)
head_state = {k: v for k, v in full_state.items() if k.startswith("classifier.")}

torch.save(head_state, OUT_PATH)

total_params = sum(v.numel() for v in head_state.values())
size_kb = OUT_PATH.stat().st_size / 1024

print(f"Head parameters: {total_params:,}")
print(f"Saved to: {OUT_PATH}")
print(f"File size: {size_kb:.1f} KB  (vs ~330MB for full weights)")