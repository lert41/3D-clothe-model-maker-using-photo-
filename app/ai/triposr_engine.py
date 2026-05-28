import torch
import numpy as np
from PIL import Image
import os

# импорт из TripoSR
from triposr.models import construct_model_from_config
from triposr.utils import remove_background


class TripoEngine:

    def __init__(self, model_path="TripoSR/checkpoints/model.ckpt"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading TripoSR on", self.device)

        self.model = construct_model_from_config(
            "TripoSR/configs/config.yaml"
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.to(self.device)
        self.model.eval()

    def generate(self, image_path, output_path):

        print("AI INPUT:", image_path)

        image = Image.open(image_path).convert("RGB")

        # remove background (очень важно)
        image = remove_background(image)

        image = np.array(image)

        with torch.no_grad():

            mesh = self.model.infer(image)

        mesh.export(output_path)

        return output_path