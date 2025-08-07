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
#from model import TransformerAlignerMel
from model2 import MelPitchAligner
from typing import List, Dict, Optional, Tuple

class MelAlignTransformerSystem(pl.LightningModule):
    def __init__(self,
                 lr: float = 2e-4,
                 weight_decay: float = 0.5,
                 input_dim_mel: int = 80,
                 input_dim_pitch: int = 1,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_ff: int = 512,
                 dropout: float = 0.1,
                 diag_w: float = 1.0,
                 ce_w: float = 1.0,
                 ga_w: float=2.0,
                 free_run_steps: int = 1,
                 free_run_w: float = 0.1,
                 free_run_steps_schedule: Optional[dict] = None,
                 use_f0: bool = True,
                 # ----- adaptive curriculum hyper-params -----
                 w_min: float = 0.05,
                 w_max: float = 1.0,
                 inc_factor: float = 1.15,
                 dec_factor: float = 0.85,
                 patience: int = 3,
                 delta: float = 0.03,
                 mono_w: float = 0.1,
                 nu: float = 0.3):                 
       
        super().__init__()
        # 初期値として free_run_steps, free_run_w を設定し、manual schedule のみ使用
        self.save_hyperparameters()
        self.free_run_steps_schedule = free_run_steps_schedule or []

        # ──────────────────────────────────────────────
        # (1) Target-SNR curriculum  *******************
        #    * list of tuples (epoch_threshold, SNR_dB)
        #    * SNR = None  →  ノイズを入れない
        # ──────────────────────────────────────────────
        self.snr_schedule: List[Tuple[int, Optional[float]]] = [
            (0,  None),       # Stabilize      (clean)
            (10, 20.0),       # Light noise    (20 dB)
            (20, 10.0),       # Mid noise      (10 dB)
            (30,  5.0),       # Heavy noise    ( 5 dB)
        ]
        self._current_snr_db: Optional[float] = None

        self.model = MelPitchAligner(
            input_dim_mel   = self.hparams.input_dim_mel,
            input_dim_pitch = self.hparams.input_dim_pitch,
            d_model         = self.hparams.d_model,
            nhead           = self.hparams.nhead,
            num_layers      = self.hparams.num_layers,
            dim_feedforward = self.hparams.dim_ff,
            dropout         = self.hparams.dropout,
            nu              = self.hparams.nu,
            diag_w          = self.hparams.diag_w,
            ce_w            = self.hparams.ce_w,
            ga_w            = self.hparams.ga_w,
            use_f0          = self.hparams.use_f0,
            mono_w          = self.hparams.mono_w
        )

        # ----- state for adaptive curriculum -----
        self.best_free_mel: Optional[float] = None
        self.no_improve_epochs: int = 0        
        # --- internal flag : True の epoch は adaptive w 更新をスキップ ---
        self._manual_w_this_epoch: bool = False

    def forward(
        self,
        src_mel:   torch.Tensor,
        tgt_mel:   torch.Tensor,
        src_pitch: torch.Tensor,
        tgt_pitch: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
        epoch: int,
        batch_idx: int,
        save_dir: str = None,
        nu: float     = None,
        weight: float = None,
        save_maps: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LightningModule → MelPitchAligner への薄いラッパー。
        * weight 以降は **必ずキーワード渡し** にすること!
        """
        return self.model.forward(
            src_mel, tgt_mel, src_pitch, tgt_pitch,
            src_lengths, tgt_lengths,
            nu=nu,
            weight=weight,
            epoch=epoch,
            batch_idx=batch_idx,
            save_dir=save_dir,
            keep_maps=save_maps,   # ← MelPitchAligner 側の引数名に合わせる
        )        

    def training_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch

        # ──────────────────────────────────────────────
        # (2) Dynamic Mix-Sampling λ  ******************
        #     * Uniform(0.0, 0.5)  を各ステップで再サンプル
        # ──────────────────────────────────────────────
        self.model.mix_lam = torch.rand(1, device=self.device).item() * 0.5

        # ──────────────────────────────────────────────
        # (1) Spectral-noise injection  ****************
        # ──────────────────────────────────────────────
        if self._current_snr_db is not None:
            # パワー計算（mean of power over (B,T,D)）
            power = (tgt_m ** 2).mean(dim=(1, 2), keepdim=True)  # (B,1,1)
            noise_power = power / (10.0 ** (self._current_snr_db / 10.0))

            # 1/f ノイズ近似: 周波数次元に重み 1/(f+1)
            D = tgt_m.size(-1)
            freqs = torch.arange(D, device=self.device).float() + 1.0
            pink_weight = 1.0 / freqs.unsqueeze(0).unsqueeze(0)  # (1,1,D)

            gauss = torch.randn_like(tgt_m)
            gauss = gauss * pink_weight                          # Pink-ish
            gauss = gauss / gauss.std(dim=(1, 2), keepdim=True) # σ=1 に正規化

            noise = torch.sqrt(noise_power) * gauss
            tgt_m = tgt_m + noise

        # teacher-forcing only
        pred, loss_tf, mel_loss, diag_loss, ga_loss = self(
            src_m, tgt_m, src_p, tgt_p, src_len, tgt_len,
            self.current_epoch, batch_idx, "./attn_maps/",
            self.hparams.nu, self.hparams.ce_w, save_maps=True
            
        )
        # ログ出力
        self.log('train_loss', loss_tf,    on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mel_l1', mel_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_diag', diag_loss,  on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_ga', ga_loss,  on_step=False, on_epoch=True, prog_bar=False)
        return loss_tf
    
    def validation_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch
        # teacher-forcing only
        pred, loss_tf, mel_loss, diag_loss, ga_loss = self(
            src_m, tgt_m, src_p, tgt_p, src_len, tgt_len,
            epoch        = self.current_epoch,   # ★必須
            batch_idx    = batch_idx,            # ★必須
            save_dir     = None,
            nu           = self.hparams.nu,
            weight       = self.hparams.ce_w,
            save_maps    = False
         )        
        val_loss = loss_tf
        self.log('val_mel_l1', mel_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss_tf

    def configure_optimizers(self):
        # Adam optimizer with weight decay
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        # Reduce LR on plateau of val_loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=1e-4,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }    
    
    def on_train_epoch_end(self):
        pass

    def on_train_epoch_start(self):
        pass
