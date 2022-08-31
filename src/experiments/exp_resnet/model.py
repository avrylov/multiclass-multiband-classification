import torch
import torch.nn.functional as f
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from models.model_resnet import ResNet


class TorchLightNet(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.model = ResNet()

    def forward(self, bands_13):
        out = self.model(bands_13)
        return f.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        idx, inputs, labels = batch
        logits = self(inputs.float())
        loss = f.nll_loss(logits, labels)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        idx, inputs, labels = batch
        logits = self(inputs.float())
        loss = f.nll_loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=5e-2,
                patience=10,
                verbose=True
            ),
            "monitor": "val_loss"
        }
        d = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
        return d
