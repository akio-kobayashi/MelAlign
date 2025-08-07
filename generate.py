#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader
import yaml

# suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# local imports
from dataset import Mel2MelDataset, collate_m2m
from solver import MelAlignTransformerSystem


def main():
    parser = argparse.ArgumentParser(description="Infer Mel-to-Mel alignment with greedy decode (batch=1)")
    parser.add_argument('--ckpt',      required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--stats',     required=True, help='Path to pitch stats .pt (pitch_mean, pitch_std)')
    parser.add_argument('--csv',       required=True, help='CSV for dataset with source/target mel and log_f0')
    parser.add_argument('--out_dir',   default='m2m_out', help='Directory to save outputs')
    parser.add_argument('--device',    default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_len',   type=int, default=200, help='Max decode length')
    parser.add_argument('--config',    default=None, help='Optional YAML config to override defaults')
    args = parser.parse_args()

    # prepare output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # load config if provided
    cfg = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # load pitch stats
    stats = torch.load(args.stats, map_location='cpu')

    # load model
    model = MelAlignTransformerSystem.load_from_checkpoint(
        args.ckpt,
        map_location=device
    )
    model = model.to(device).eval()

    # dataset & loader (batch_size=1)
    ds = Mel2MelDataset(
        args.csv,
        stats,
        map_location='cpu'
    )
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_m2m,
        pin_memory=True
    )

    for idx, batch in enumerate(dl):
        src_m, src_p, tgt_m, tgt_p, src_len, tgt_len = batch        
        # move to device
        src_m = src_m.to(device)
        src_p = src_p.to(device)
        # greedy_decode には Python の int を渡す
        src_len = src_len.to(device)[0].item()
       
        with torch.no_grad():
            # greedy_decode は mel_pred のみ返す
            mel_pred = model.model.greedy_decode(
                src_m,           # (1, T_src, D)
                src_p,           # (1, T_src)
                src_len,         # int
                max_len=args.max_len
            )
        # バッチ次元を落としてから denormalize
        mel_pred = mel_pred.squeeze(0).cpu()   # (T, F)
        mel_pred = ds.denormalize(mel_pred)
            
        print(torch.mean(mel_pred), torch.max(mel_pred), torch.min(mel_pred))
        mel_src = ds.denormalize(src_m.squeeze(0).cpu())
        print(torch.mean(mel_src), torch.max(mel_src), torch.min(mel_src))        
        
        # save
        # enumerate で得られた idx を使って DataFrame の該当行を取得
        row = ds.df.iloc[idx]
        key = Path(row['source']).stem        
        
        out_mel_path   = os.path.join(args.out_dir, f"{key}.pt")
        torch.save(mel_pred, out_mel_path)
        print(f"Saved: {out_mel_path}")

if __name__ == '__main__':
    main()
