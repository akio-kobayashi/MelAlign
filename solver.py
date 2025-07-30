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
from typing import List, Dict, Optional

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
                 scheduler: dict = None,
                 free_run_steps: int = 10,
                 free_run_w: float = 0.1,
                 free_run_steps_schedule: List[Dict[str,int]] = None,
                 use_f0: bool = True):
        
        super().__init__()
        self.free_run_steps_schedule = free_run_steps_schedule or []       
        self.save_hyperparameters()
        self.model = MelPitchAligner(
            input_dim_mel   = self.hparams.input_dim_mel,
            input_dim_pitch = self.hparams.input_dim_pitch,
            d_model         = self.hparams.d_model,
            nhead           = self.hparams.nhead,
            num_layers      = self.hparams.num_layers,
            dim_feedforward = self.hparams.dim_ff,
            dropout         = self.hparams.dropout,
            nu              = 0.3,
            diag_w          = self.hparams.diag_w,
            ce_w            = self.hparams.ce_w,
            free_run_steps  = self.hparams.free_run_steps,
            free_run_w      = self.hparams.free_run_w,
            use_f0          = self.hparams.use_f0,
        )

    def forward(self,
                src_mel, src_pitch, src_lengths,
                tgt_mel, tgt_pitch, tgt_lengths):
        # src_lengths, tgt_lengths を内部で attention mask／loss mask に利用
        return self.model(src_mel, src_pitch, src_lengths,
                          tgt_mel, tgt_pitch, tgt_lengths)

    def training_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch
        loss_tf, metrics = self(src_m, src_p, src_len, tgt_m, tgt_p, tgt_len)
        # サブ・ロスをステップごとにログ
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False)
        # 全体のトータル・ロスをエポックごとにプログレスバー表示
        self.log('train_loss', loss_tf, on_step=False, on_epoch=True, prog_bar=True)
        # メル損失(mel_l1)をエポックごとにプログレスバー表示
        self.log('train_mel_l1', metrics['mel_l1'], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_tf
    
    def validation_step(self, batch, batch_idx):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch
        # —— teacher-forced の損失とサブメトリクス
        loss_tf, metrics = self(src_m, src_p, src_len, tgt_m, tgt_p, tgt_len)
        val_loss = loss_tf
        self.log('val_mel_l1', metrics['mel_l1'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # —— free_run 損失をバッチ全体で計算してログ
        if self.hparams.free_run_w > 0.0:
            # バッチ中の最小ターゲット長を取得
            min_len = tgt_len.min().item()
            # L は free_run_steps と min_len の小さいほう
            L = min(self.hparams.free_run_steps, min_len)
            if L > 0:
                # 安全な開始位置 p をランダム選択
                p = torch.randint(0, min_len - L + 1, (1,)).item()
                # エンコード済みメモリをバッチ全体で detach
                memory_detach = self.model.encode(src_m, src_p)
                # 部分系列で自己回帰デコード
                mel_free, f0_free = self.model.decode_free_run(memory_detach, p, L)
                # 教師信号を対応箇所から切り出し
                mel_gt     = tgt_m[:, p:p+L]
                loss_free_m = F.l1_loss(mel_free, mel_gt)
                self.log('val_free_mel', loss_free_m, on_step=False, on_epoch=True, prog_bar=True)
                # f0 損失も同様にバッチ全体で
                if self.hparams.use_f0:
                    f0_gt       = tgt_p[:, p:p+L]
                    loss_free_f = F.l1_loss(f0_free, f0_gt)
                    self.log('val_free_f0', loss_free_f, on_step=False, on_epoch=True, prog_bar=True)
                    val_free_total = loss_free_m + loss_free_f
                else:
                    val_free_total = loss_free_m
                self.log('val_free_total', val_free_total, on_step=False, on_epoch=True, prog_bar=True)        
        '''
        # —— free_run 損失は別指標としてログ（モデル選択に使う場合は monitor="val_free_mel" 等に）
        if self.hparams.free_run_w > 0.0:
            idx = torch.randint(0, src_m.size(0), (1,)).item()
            memory = self.model.encode(src_m[idx:idx+1], src_p[idx:idx+1])
            L = min(self.hparams.free_run_steps, tgt_m.size(1))
            p = torch.randint(0, tgt_m.size(1) - L + 1, (1,)).item()
            mel_free, f0_free = self.model.decode_free_run(memory, p, L)
            mel_gt = tgt_m[idx:idx+1, p:p+L]
            loss_free_m = F.l1_loss(mel_free, mel_gt)
            self.log('val_free_mel', loss_free_m, on_step=False, on_epoch=True, prog_bar=True)
            val_free_total = loss_free_m
            if self.hparams.use_f0:
                f0_gt  = tgt_p[idx:idx+1, p:p+L]
                loss_free_f = F.l1_loss(f0_free, f0_gt)
                # モニタリング用
                self.log('val_free_f0',  loss_free_f, on_step=False, on_epoch=True, prog_bar=True)

                val_free_total += loss_free_f
            self.log('val_free_total', val_free_total, on_epoch=True, prog_bar=True)
        ''' 
        return loss_tf        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        def lr_lambda(step):
            warmup_steps = 4000
            return min(step / warmup_steps, 1.0)
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }    
    
    def on_train_epoch_end(self):
        pass

    def on_train_epoch_start(self):
        # 現在のエポック数に応じて free_run_steps を更新
        for item in self.free_run_steps_schedule:
            if self.current_epoch >= item["epoch"]:
                self.model.free_run_steps = item["steps"]
        # いくつかログに出しておくと安心
        self.log("free_run_steps", self.model.free_run_steps, prog_bar=True)
