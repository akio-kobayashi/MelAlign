import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model import TransformerAlignerMel

# ----------------------------------------------------------------
# LightningModule for Mel-to-Mel Alignment
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
        free_run_interval: int = 100,
        free_run_weight: float = 1.0,
        free_run_segment: int = 200
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerAlignerMel(
            input_dim_mel    = self.hparams.input_dim_mel,
            input_dim_pitch  = self.hparams.input_dim_pitch,
            d_model          = self.hparams.d_model,
            nhead            = self.hparams.nhead,
            num_layers       = self.hparams.num_layers,
            dim_feedforward  = self.hparams.dim_ff,
            dropout          = self.hparams.dropout,
            nu               = 0.3,
            diag_weight      = self.hparams.diag_w,
            ce_weight        = self.hparams.ce_w
        )
        self.train_losses = []
        self.val_losses   = []

    def forward(self, src_mel, src_pitch, tgt_mel, tgt_pitch):
        return self.model(src_mel, src_pitch, tgt_mel, tgt_pitch)

    def training_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p = batch
        # teacher-forcing
        loss_tf, metrics = self(src_m, src_p, tgt_m, tgt_p)
        loss = loss_tf
        # intermittent free-run
        if batch_idx % self.hparams.free_run_interval == 0 and self.hparams.free_run_weight > 0:
            idx = torch.randint(0, src_m.size(0), (1,), device=self.device)
            seg = min(self.hparams.free_run_segment, tgt_m.size(1))
            with torch.no_grad():
                ph, pp = self.model.greedy_decode(
                    src_m[idx, :seg], src_p[idx, :seg], max_len=seg
                )
            # pad/trunc
            if ph.size(1) < seg:
                pad = seg - ph.size(1)
                ph = torch.cat([ph, ph.new_zeros(1, pad, ph.size(2))], dim=1)
                pp = torch.cat([pp, pp.new_zeros(1, pad)], dim=1)
            else:
                ph, pp = ph[:, :seg], pp[:, :seg]
            fr_h = F.l1_loss(ph, tgt_m[idx, :seg])
            fr_p = F.l1_loss(pp, tgt_p[idx, :seg])
            loss += self.hparams.free_run_weight * (fr_h + fr_p)
            metrics['fr_l1_mel'] = fr_h
            metrics['fr_l1_pitch'] = fr_p
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.train_losses.append(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p = batch
        loss_tf, _ = self(src_m, src_p, tgt_m, tgt_p)
        loss_fr = torch.tensor(0.0, device=self.device)
        if batch_idx % self.hparams.free_run_interval == 0:
            idx = torch.randint(0, src_m.size(0), (1,), device=self.device)
            seg = min(self.hparams.free_run_segment, tgt_m.size(1))
            with torch.no_grad():
                ph, pp = self.model.greedy_decode(
                    src_m[idx, :seg], src_p[idx, :seg], max_len=seg
                )
            if ph.size(1) < seg:
                pad = seg - ph.size(1)
                ph = torch.cat([ph, ph.new_zeros(1, pad, ph.size(2))], dim=1)
                pp = torch.cat([pp, pp.new_zeros(1, pad)], dim=1)
            else:
                ph, pp = ph[:, :seg], pp[:, :seg]
            loss_fr = F.l1_loss(ph, tgt_m[idx, :seg]) + F.l1_loss(pp, tgt_p[idx, :seg])
        self.log('val_loss_tf', loss_tf, on_epoch=True)
        self.log('val_loss_fr', loss_fr, on_epoch=True)
        return {'val_loss_tf': loss_tf, 'val_loss_fr': loss_fr}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.hparams.lr/10.0
        )
        return {'optimizer': opt, 'lr_scheduler': sched, 'gradient_clip_val': 1.0}

    def on_train_epoch_end(self):
        avg = torch.stack(self.train_losses).mean()
        self.log('train_loss', avg, prog_bar=True, on_epoch=True)
        self.train_losses.clear()

    def on_validation_epoch_end(self):
        if not self.val_losses:
            return
        avg = torch.stack(self.val_losses).mean()
        self.log('val_loss', avg, prog_bar=True, on_epoch=True)
        self.val_losses.clear()
