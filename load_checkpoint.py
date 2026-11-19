import torch
import os

from freqnet import FreqNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

checkpoint_path = os.path.join(
    BASE_DIR,
    "checkpoints",
    "4-classes-freqnet-v2.pth"
)

model = FreqNet()

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)
model.eval()

print("Checkpoint loaded successfully!")
