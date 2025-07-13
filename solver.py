# dataset.py（または collate 定義ファイル）
import torch
from torch.utils.data import Dataset

def collate_m2m(batch):
    src_m_list, src_p_list, tgt_m_list, tgt_p_list = zip(*batch)
    B = len(batch)
    src_lengths = torch.tensor([sm.size(0) for sm in src_m_list], dtype=torch.long)
    tgt_lengths = torch.tensor([tm.size(0) for tm in tgt_m_list], dtype=torch.long)

    T_src_max = src_lengths.max().item()
    T_tgt_max = tgt_lengths.max().item()
    D = src_m_list[0].size(1)

    src_mel_pad   = torch.zeros(B, T_src_max, D)
    src_pitch_pad = torch.zeros(B, T_src_max)
    tgt_mel_pad   = torch.zeros(B, T_tgt_max, D)
    tgt_pitch_pad = torch.zeros(B, T_tgt_max)

    for i, (sm, sp, tm, tp) in enumerate(batch):
        Ls, Lt = sm.size(0), tm.size(0)
        src_mel_pad[i, :Ls]   = sm
        src_pitch_pad[i, :Ls] = sp
        tgt_mel_pad[i, :Lt]   = tm
        tgt_pitch_pad[i, :Lt] = tp

    return src_mel_pad, src_pitch_pad, tgt_mel_pad, tgt_pitch_pad, src_lengths, tgt_lengths


# lightning_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model import TransformerAlignerMel

class MelAlignTransformerSystem(pl.LightningModule):
    def __init__(self,
                 lr: float = 2e-4,
                 input_dim_mel: int = 80,
                 input_dim_pitch: int = 1,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_ff: int = 512,
                 dropout: float = 0.1,
                 diag_w: float = 1.0,
                 ce_w: float = 1.0):
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

    def forward(self,
                src_mel, src_pitch, src_lengths,
                tgt_mel, tgt_pitch, tgt_lengths):
        # src_lengths, tgt_lengths を内部で attention mask／loss mask に利用
        return self.model(src_mel, src_pitch, src_lengths,
                          tgt_mel, tgt_pitch, tgt_lengths)

    def training_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch
        loss_tf, metrics = self(src_m, src_p, src_len, tgt_m, tgt_p, tgt_len)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.train_losses.append(loss_tf.detach())
        return loss_tf

    def validation_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch
        loss_tf, metrics = self(src_m, src_p, src_len, tgt_m, tgt_p, tgt_len)
        self.log('val_loss', loss_tf, on_step=False, on_epoch=True, prog_bar=True)
        return loss_tf

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return opt
        #sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    opt,
        #    T_max=self.trainer.max_epochs,
        #    eta_min=self.hparams.lr / 10.0
        #)
        #return {'optimizer': opt, 'lr_scheduler': sched}

    def on_train_epoch_end(self):
        if self.train_losses:
            avg = torch.stack(self.train_losses).mean()
            self.log('train_loss', avg, prog_bar=True, on_epoch=True)
            self.train_losses.clear()
