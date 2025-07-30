import csv
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

def get_max_len(args):
    max_len = 0
    df = pd.read_csv(args.csv)

    def _check_mel(mel):
        # ensure mel is (T, 80)
        if mel.ndim != 2:
            raise ValueError(f"mel tensor must be 2‑D, got {mel.shape}")
        if mel.size(0) == 80 and mel.size(1) != 80:
            mel = mel.transpose(0, 1)
        if mel.size(1) != 80:
            raise ValueError(f"Unexpected mel shape after transpose: {mel.shape}")
        return mel

    for idx, row in df.iterrows():
        src = torch.load(row['source'])
        src_mel = _check_mel(src['mel'])
        tgt = torch.load(row['target'])
        tgt_mel = _check_mel(tgt['mel'])
        if src_mel.shape[0] > max_len:
            max_len = src_mel.shape[0]
        if tgt_mel.shape[0] > max_len:
            max_len = tgt_mel.shape[0]
    return max_len

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, required=True,
    )
    args = parser.parse_args()
    max_len = get_max_len(args)
    print(max_len)
    
