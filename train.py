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
from pytorch_lightning.callbacks import EarlyStopping

# local modules for Mel-to-Mel alignment
torch.set_float32_matmul_precision("high")
from dataset import Mel2MelDataset, collate_m2m
from solver import MelAlignTransformerSystem
#from greedy_eval import GreedyEvalCallbackMel

# suppress warnings
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue.*")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message="The number of training batches.*")
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*"
)

def train(cfg: dict):
    # オプションのチェックポイントパス
    ckpt_path = cfg.get("ckpt_path", None)
    
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)
    # ─── dataset ───────────────────────────────────────────
    sort_by_len=cfg.get("sort_by_len", False)
    train_ds = Mel2MelDataset(
        cfg["train_csv"],
        stats, 
        map_location=cfg.get("map_location", "cpu"),
        sort_by_len=sort_by_len,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=not sort_by_len,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_m2m,
        pin_memory=True,
    )
    valid_ds = Mel2MelDataset(
        cfg["valid_csv"],
        stats,
        map_location=cfg.get("map_location", "cpu"),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_m2m,
        pin_memory=True,
    )

    # ─── model ─────────────────────────────────────────────
    model = MelAlignTransformerSystem(
        lr=cfg.get("lr", 2e-4),
        weight_decay=cfg.get("weight_decay", 0.5),
        input_dim_mel=cfg.get("input_dim_mel", 80),
        input_dim_pitch=cfg.get("input_dim_pitch", 1),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 3),
        dim_ff=cfg.get("dim_ff", 512),
        dropout=cfg.get("dropout", 0.1),
        diag_w=cfg.get("diag_w", 1.0),
        ce_w=cfg.get("ce_w", 1.0),
        ga_w=cfg.get("ga_w", 2.0),
        use_f0=cfg.get("use_f0", True),
        mono_w=cfg.get("mono_w", 0.1),
        nu=cfg.get("nu", 0.3)
    )    

    # ─── callbacks / logger ─────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg["ckpt_dir"],
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",    
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=30,         # 改善が20エポック続かなければ停止
        min_delta=1e-4,      # 0.001以上の改善で更新とみなす
        verbose=True
    )    
    from pathlib import Path

    # ckpt_path があれば同じ version に追記する
    if ckpt_path:
        # 例: logs/m2m_align/version_0/checkpoints/last.ckpt
        version = Path(ckpt_path).parents[1].name  # 'version_0'
        tb_logger = TensorBoardLogger(
            save_dir=cfg["log_dir"],
            name="m2m_align",
            version=version,
        )
    else:
        tb_logger = TensorBoardLogger(
            save_dir=cfg["log_dir"],
            name="m2m_align",
        )

    # ─── trainer ────────────────────────────────────────────
    trainer = pl.Trainer(
        num_sanity_val_steps=0,        
        max_epochs=cfg.get("max_epochs", 100),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.get("gpus", 1),
        precision=cfg.get("precision", "16-mixed"),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm="norm",
        default_root_dir=cfg["work_dir"],
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_monitor, early_stop],
        profiler="simple",
        check_val_every_n_epoch=1,
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 4),
    )

    
    # ─── fit ────────────────────────────────────────────────
    # ckpt_path を指定すると，そのチェックポイントから学習を再開する
    trainer.fit(model, train_dl, valid_dl, ckpt_path=ckpt_path)
    
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser(
        description="Train Mel-to-Mel alignment model"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML config for mel2mel training"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="(optional) Path to a pretrained .ckpt file to resume from"
    )
    
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    # コマンドライン引数があれば優先して設定に登録
    if args.ckpt_path is not None:
        cfg["ckpt_path"] = args.ckpt_path
    train(cfg)
