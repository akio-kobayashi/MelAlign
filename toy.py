import argparse
from pathlib import Path
import yaml
import warnings

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
import torch.nn as nn

# local modules for Mel-to-Mel alignment
torch.set_float32_matmul_precision("high")
from dataset import Mel2MelDataset, collate_m2m
from solver import MelAlignTransformerSystem
from model import FixedPositionalEncoding
#from greedy_eval import GreedyEvalCallbackMel

# suppress warnings
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message="The number of training batches.*")
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*"
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniCrossOverfit(nn.Module):
    def __init__(self, n_mels, d_model=128, nhead=4):
        super().__init__()
        # input → d_model
        self.input_proj  = nn.Linear(n_mels, d_model)
        # positional encoding
        self.posenc      = FixedPositionalEncoding(d_model, max_len=1000)
        # self-attn & cross-attn
        self.self_attn   = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # output proj back to mel, zero-init
        self.output_proj = nn.Linear(d_model, n_mels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        x: (1, T, F)
        """
        # 1) embed + pos
        h = self.input_proj(x)           # (1, T, D)
        h = self.posenc(h)               # add positional
        # 2) self-attn
        h2, _ = self.self_attn(h, h, h)  # (1, T, D)
        # 3) cross-attn using same as "memory"
        y, _  = self.cross_attn(h2, h2, h2) 
        # 4) residual + zero‐init out
        return x + self.output_proj(y)   # (1, T, F)
    
class MiniTransformerOverfit(nn.Module):
    def __init__(self, n_mels, d_model=128, nhead=4, dim_ff=256, dropout=0.0):
        super().__init__()
        # 1) メル→埋め込み
        self.input_proj = nn.Linear(n_mels, d_model)
        # 2) 位置エンコーダ
        self.posenc = FixedPositionalEncoding(d_model, max_len=1000)
        # 3) 単層 Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        # 4) 出力投影 back to mel
        self.output_proj = nn.Linear(d_model, n_mels)
        # 出力層をゼロ初期化→初期状態で out=0
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)        

    def forward(self, x):
        # x: (1, T, F)
        h = self.input_proj(x)            # (1, T, d_model)
        h = self.posenc(h)                # add positional encoding
        h = self.transformer(h)           # (1, T, d_model)
        return x + self.output_proj(h)

class ResidualMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 1) 前半は普通の線形→ReLU
        self.lin1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        # 2) 後半はゼロ初期化しておく（学習開始時は出力がゼロになる）
        self.lin2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 恒等マッピング + 学習可能な微修正
        return x + self.lin2(self.relu(self.lin1(x)))

def train(cfg: dict):
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)

    # ─── ToyAEC overfit test ────────────────────────────────
    if cfg.get("overfit_test", False):
        from torch import nn, optim
        # ─── データ取得 ───────────────────────────────────────────
        tmp_ds = Mel2MelDataset(cfg["train_csv"], stats, map_location=cfg.get("map_location","cpu"))
        src_mel, src_pitch, tgt_mel, tgt_pitch = tmp_ds[0]
        device = torch.device(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))
        src = src_mel.unsqueeze(0).to(device)   # (1, T, F)

        # ─── モデル／最適化準備 ───────────────────────────────────
        model_x = MiniCrossOverfit(src_mel.size(1)).to(device)
        opt     = torch.optim.Adam(model_x.parameters(), lr=1e-3)

        # ─── Overfit ループ ───────────────────────────────────────
        print("=== MiniCrossOverfit Test Start ===")
        for step in range(300):
            opt.zero_grad()
            out = model_x(src)               # src は (1, T, F)
            loss = F.l1_loss(out, src)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"[Step {step:3d}] L1 = {loss.item():.6f}")
            if loss.item() < 1e-5:
                print(f"Converged at step {step}, L1={loss.item():.6f}")
                break
        print("=== MiniCrossOverfit Test End ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Mel-to-Mel alignment model"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML config for mel2mel training"
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
