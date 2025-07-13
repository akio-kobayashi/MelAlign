from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
from pathlib import Path

def main(args):
    # 例:
    # df1: key, tensor
    # df2: key, tensor

    # tensor列をリネームしてからマージ
    df1 = pd.read_csv(args.source)
    df1_renamed = df1.rename(columns={'tensor': 'source'})
    df2 = pd.read_csv(args.target)
    df2_renamed = df2.rename(columns={'tensor': 'target'})

    # keyを軸に内部結合（inner join）
    merged_df = pd.merge(df1_renamed, df2_renamed, on='key', how='inner')

    filtered_df = merged_df.dropna(subset=['source', 'target'])

    filtered_df.to_csv(args.output_csv, index=False)
         
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--output_csv', type=str)
    args=parser.parse_args()
       
    main(args)
