import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model import TransformerAlignerMel

# ----------------------------------------------------------------
# LightningModule for Mel-to-Mel Alignment without greedy_decode
# ----------------------------------------------------------------
class MelAlignTransformerSystem(pl.LightningModule):
    def __init__(
        self,
        lr: float = 2e-4,
        input_dim_mel: int = 80,
        input_dim_pitch: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
        diag_w: float = 1.0,
        ce_w: float = 1.0,
        # free_run parameters retained for compatibility but unused
        free_run_interval: int = 100,
        free_run_weight: float = 1.0,
        free_run_segment: int = 200
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerAlignerMel(
            input_dim_mel   = self.hparams.input_dim_mel,
            input_dim_pitch = self.hparams.input_dim_pitch,
            d_model         = self.hparams.d_model,
            nhead           = self.hparams.nhead,
            num_layers      = self.hparams.num_layers,
            dim_feedforward = self.hparams.dim_ff,
            dropout         = self.hparams.dropout,
            nu              = 0.3,
            diag_weight     = self.hparams.diag_w,
            ce_weight       = self.hparams.ce_w
        )
        self.train_losses = []

    def forward(self, src_mel, src_pitch, tgt_mel, tgt_pitch):
        return self.model(src_mel, src_pitch, tgt_mel, tgt_pitch)

    def training_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p = batch
        #src_m.requires_grad_()
        #src_p.requires_grad_()        
        # teacher-forcing
        loss_tf, metrics = self(src_m, src_p, tgt_m, tgt_p)
        # log and return
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.train_losses.append(loss_tf.detach())
        return loss_tf

    def validation_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p = batch
        # teacher-forcing only
        loss_tf, metrics = self(src_m, src_p, tgt_m, tgt_p)
        self.log('val_loss', loss_tf, on_step=False, on_epoch=True, prog_bar=True)
        return loss_tf

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 10.0
        )
        return {
            'optimizer': opt,
            'lr_scheduler': sched,
        }

    def on_train_epoch_end(self):
        if self.train_losses:
            avg = torch.stack(self.train_losses).mean()
            self.log('train_loss', avg, prog_bar=True, on_epoch=True)
            self.train_losses.clear()
