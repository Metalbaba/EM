import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.sft_model import load_sft_model

model = load_sft_model("gpt2", "cpu")

model.load_state_dict(torch.load("../../models/projected_gpt2_joint.pth", map_location="cpu"))

model.eval()
print("Model loaded and ready for inference")