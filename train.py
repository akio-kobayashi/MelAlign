import argparse
from pathlib import Path
import yaml
import warnings

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

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
    stats = torch.load(cfg["stats_tensor"], map_location="cpu", weights_only=True)
    # ─── dataset ───────────────────────────────────────────
    train_ds = Mel2MelDataset(
        cfg["train_csv"],
        stats, 
        map_location=cfg.get("map_location", "cpu"),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
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
        input_dim_mel=cfg.get("input_dim_mel", 80),
        input_dim_pitch=cfg.get("input_dim_pitch", 1),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 3),
        dim_ff=cfg.get("dim_ff", 512),
        dropout=cfg.get("dropout", 0.1),
        diag_w=cfg.get("diag_w", 1.0),
        ce_w=cfg.get("ce_w", 1.0),
        #free_run_interval=cfg.get("free_run_interval", 100),
        #free_run_weight=cfg.get("free_run_weight", 1.0),
        #free_run_segment=cfg.get("free_run_segment", 200),
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
    '''
    greedy_cb = GreedyEvalCallbackMel(
        every_n_epochs=cfg.get("greedy_every_n_epochs", 5),
        max_len=cfg.get("greedy_max_len", 1000),
        num_batches=cfg.get("greedy_num_batches", 50),
    )
    '''
    lr_monitor = LearningRateMonitor(logging_interval="step")
    tb_logger = TensorBoardLogger(save_dir=cfg["log_dir"], name="m2m_align")

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
        callbacks=[ckpt_cb, lr_monitor],
        profiler="simple",
        check_val_every_n_epoch=1,
    )

    # ─── fit ────────────────────────────────────────────────
    trainer.fit(model, train_dl, valid_dl)

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
