import os
import torch
import pandas as pd
import warnings

from src.dift import SDFeaturizer
from src.initialization import initialize
from src.optimization import optimize

from main import get_opts

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    args = get_opts(is_main=False)
    samples = pd.read_csv(args.samples_df_path)

    if args.shuffle:
        print("Shuffling order of items")
        samples = samples.sample(len(samples)).reset_index(drop=True)

    if args.n_rows is not None:
        print(f"Only using top {args.n_rows} items")
        samples = samples.head(args.n_rows)


    available = [i.split("_")[0] for i in os.listdir(args.con_dir)]
    torch.manual_seed(args.seed)
    dift = SDFeaturizer(sd_id=args.sd_checkpoint)

    for i, row in samples.iterrows():
        hex_code = row['hex']
        if hex_code not in available:
            continue
        args.prompt = row['name']
        args.target_image_path = row['file_path']
        for j in range(args.repeats):
            rep = None if args.repeats == 1 else j
            desc = f"({i+1}/{len(samples)}) {row['fn'][:10]}"
            try:
                data, H = initialize(args, dift, rep=rep)
                optimize(args, data, desc=desc, initial_transform=H, rep=rep)
            except Exception as e:
                if args.ignore_errors:
                    continue
                else:
                    raise e


if __name__ == '__main__':
    main()

