from argparse import ArgumentParser
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import glob
import os, sys
from pathlib import Path

def main(args):
    keys, tensors=[], []
    
    for dir in args.dir:
        for idx, filepath in enumerate(sorted(glob.glob(os.path.join(dir, '*.pt'))), start=1):
            path = Path(filepath)
            key = path.stem.replace('_fake', '').replace('_DT', '')
            keys.append(key)
            tensors.append(path)
    output_df = pd.DataFrame()
    output_df['key'] = keys
    output_df['tensor'] = tensors
    output_df.to_csv(args.output_csv, index=False)
         
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', nargs='*', type=str, required=True)
    parser.add_argument('--output_csv', type=str)

    args=parser.parse_args()
       
    main(args)
