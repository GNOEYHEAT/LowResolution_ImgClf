import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from sklearn.metrics import f1_score

class ImageClassifier(pl.LightningModule):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.num_class = cfg.num_class

    def forward(self, x):
        outputs = self.backbone(x)
        return outputs

    def step(self, batch):
        x = batch["pixel_values"]
        y = batch["labels"]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss_ce, y, y_hat = self.step(batch)
        loss_cos = nn.CosineEmbeddingLoss()(
            y_hat, y, torch.Tensor([1]).to(self.device)
        )
        loss = loss_ce + loss_cos
        f1 = f1_score(y_hat.max(dim=1)[1].cpu().numpy(), y.max(dim=1)[1].cpu().numpy(), average='macro')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_ce, y, y_hat = self.step(batch)
        loss_cos = nn.CosineEmbeddingLoss()(
            y_hat, F.one_hot(y.long(), self.num_class), torch.Tensor([1]).to(self.device)
        )
        loss = loss_ce + loss_cos
        f1 = f1_score(y_hat.max(dim=1)[1].cpu().numpy(), y.cpu().numpy(), average='macro')
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        f1 = f1_score(y_hat.max(dim=1)[1].cpu().numpy(), y.cpu().numpy(), average='macro')
        self.log("test_f1", f1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["pixel_values"]
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        if self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.learning_rate, momentum=0.9)
        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        if self.cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.scheduler == "none":
            return optimizer
        if self.cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.cfg.epochs // 2,
                eta_min=self.cfg.learning_rate // 10,
            )
            return [optimizer], [scheduler]