from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
import torch
import feature_pipeline as F
from pathlib import Path

def main(args):

    mel_list, f0_list = [], []
    for dir in args.dir:
        for idx, filepath in enumerate(sorted(glob.glob(os.path.join(dir, '**/*.pt'))), start=1):
            data = torch.load(filepath, map_location="cpu")
            f0_list.append(data['log_f0'])
            mel = data["mel"].float().squeeze()
            if mel.size(0) == 80:          # stored as (80, T)
                mel = mel.transpose(0, 1)
            if mel.size(1) != 80:
                raise ValueError(f"Unexpected mel shape: {mel.shape}")
            mel_list.append(mel)
            
    f0_cat = torch.cat(f0_list)
    mel_cat = torch.cat(mel_list, dim=0)
    pitch_mean = f0_cat.mean().item()
    pitch_std  = f0_cat.std(unbiased=False).item() + 1e-9
    mel_mean   = mel_cat.mean(dim=0)
    mel_std    = mel_cat.std(dim=0) + 1e-9
            
    output_stats = {"mel_mean": mel_mean, "mel_std": mel_std, "pitch_mean": pitch_mean, "pitch_std": pitch_std}
    torch.save(output_stats, args.out)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', nargs='*')
    parser.add_argument('--out', default="stats.pt")

    args=parser.parse_args()
       
    main(args)
