import torch.nn as nn
from transformers import AutoModel

class ImageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.cfg.img_model_name)
        self.clf = nn.Linear(self.cfg.latent_dim, self.cfg.num_class)

    def forward(self, inputs):
        enc = self.model(inputs)
        x = enc.pooler_output
        outputs = self.clf(x)
        return outputs