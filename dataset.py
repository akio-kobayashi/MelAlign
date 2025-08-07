import csv
import pandas as pd
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

def collate_m2m(batch):
    """
    Collate function to pad a batch of (src_mel, src_pitch, tgt_mel, tgt_pitch).
    Returns:
      src_mel_pad: (B, T_src_max, D)
      src_pitch_pad: (B, T_src_max)
      tgt_mel_pad: (B, T_tgt_max, D)
      tgt_pitch_pad: (B, T_tgt_max)
      src_lengths:  (B,) 実フレーム長
      tgt_lengths:  (B,) 実フレーム長
    """
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

class Mel2MelDataset(Dataset):
    """
    Loads paired log-mel spectrograms and pitch from .pt files.
    CSV must have columns 'source' and 'target', each pointing to a .pt file containing:
      - 'mel': Tensor of shape (T, n_mels)
      - 'log_f0': Tensor of shape (T,)
    Returns tuples (src_mel, src_pitch, tgt_mel, tgt_pitch).
    """
    def __init__(self, csv_path: str or Path, stats: dict, map_location: str = 'cpu', sort_by_len: bool = False) -> None:
        self.map_location = map_location
        #self.rows: List[dict] = []
        csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        
        # 各サンプルのメル長を取得し、短い順に並べ替え
        if sort_by_len:
            print(f'use sorted-by-length dataset')
            lengths = []
            for idx, row in self.df.iterrows():
                # ファイル読み込み mel のフレーム長を取得
                data = torch.load(row['source'], map_location=self.map_location)
                mel  = self._check_mel(data['mel'])
                lengths.append(mel.size(0))
            # ソート用インデックスを計算し df を再構築
            sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
            self.df = self.df.iloc[sorted_idx].reset_index(drop=True)

        self.mean_mel = stats['mel_mean']
        self.std_mel = stats['mel_std']
        self.mean_f0 = stats['pitch_mean']
        self.std_f0  = stats['pitch_std']
        #print(f'{self.mean_mel.shape} {self.std_mel.shape} {self.mean_f0} {self.std_f0}')
    def __len__(self) -> int:
        return len(self.df)

    def _check_mel(self, mel):
        # ensure mel is (T, 80)
        if mel.ndim != 2:
            raise ValueError(f"mel tensor must be 2‑D, got {mel.shape}")
        if mel.size(0) == 80 and mel.size(1) != 80:
            mel = mel.transpose(0, 1)
        if mel.size(1) != 80:
            raise ValueError(f"Unexpected mel shape after transpose: {mel.shape}")
        return mel
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        #row = self.rows[idx]
        src = torch.load(row['source'], map_location=self.map_location)
        src_mel = self._check_mel(src['mel'])
        tgt = torch.load(row['target'], map_location=self.map_location)
        tgt_mel = self._check_mel(tgt['mel'])

        src_mel   = (src_mel.float() - self.mean_mel) / (self.std_mel + 1.e-9)
        src_pitch = (src['log_f0'].float() - self.mean_f0) / (self.std_f0 + 1.e-9)
        tgt_mel   = (tgt_mel.float() - self.mean_mel) / (self.std_mel + 1.e-9)
        tgt_pitch = (tgt['log_f0'].float() - self.mean_f0) / (self.std_f0 + 1.e-9)
        
        return src_mel, src_pitch, tgt_mel, tgt_pitch

    def denormalize(self, mel):
        return (mel * self.std_mel) + self.mean_mel
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    ds = Mel2MelDataset('path/to/pairs.csv')
    dl = DataLoader(ds, batch_size=4, collate_fn=collate_m2m)
    src_mel_pad, src_pitch_pad, tgt_mel_pad, tgt_pitch_pad = next(iter(dl))
    print('Shapes:', src_mel_pad.shape, src_pitch_pad.shape, tgt_mel_pad.shape, tgt_pitch_pad.shape)
