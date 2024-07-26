import torch
from argparse import ArgumentParser
import warnings

from src.dift import SDFeaturizer
from src.initialization import initialize
from src.optimization import optimize

warnings.filterwarnings("ignore", category=FutureWarning)


def get_opts(is_main=True):
    parser = ArgumentParser()
    if is_main:
        parser.add_argument('prompt', type=str)
        parser.add_argument('--target_image_path', type=str, default="target_images")
        parser.add_argument('--output_folder', type=str, default=None)
    else:
        parser.add_argument('--samples_df_path', type=str, default='test_set/metadata.csv')
        parser.add_argument('--ignore_errors', action='store_true', default=False)
        parser.add_argument('--n_rows', type=int, default=None)
        parser.add_argument('--shuffle', action='store_true', default=False)
        parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--iter_num', type=int, default=100)
    parser.add_argument('--viz_every', type=int, default=10)
    parser.add_argument('--ransac_k', type=int, default=5)
    parser.add_argument('--ransac_threshold', type=float, default=50.)
    parser.add_argument('--init_tries', '-t', type=int, default=8)
    parser.add_argument('--initial_temp', type=float, default=100.0)
    parser.add_argument('--temp_multiplier', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--lambda_dift', type=float, default=1.0)
    parser.add_argument('--lambda_reg_global', type=float, default=100)
    parser.add_argument('--lambda_reg', type=float, default=1e-4)
    parser.add_argument('--lambda_oob', type=float, default=1e-4)
    parser.add_argument('--lambda_mask', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--n_interpolations', '-ni', type=int, default=8)
    parser.add_argument('--line_weighting', '-lw', type=float, default=0.0)
    parser.add_argument('--df_filename', type=str, default='prototypes/metadata.csv')
    parser.add_argument('--sd_checkpoint', type=str, default='weights/SD_with_prompt')
    parser.add_argument('--con_dir', type=str, default='skeletons/Santakku')
    parser.add_argument('--font_dir', type=str, default='prototypes/Santakku')
    parser.add_argument('--suffix', type=str, default='results')
    parser.add_argument('--intermediate_factor', type=float, default=40.)
    parser.add_argument('--init_scatter_size', type=int, default=8)
    parser.add_argument('--scatter_size', type=int, default=20)
    parser.add_argument('--line_alpha', type=float, default=0.8)
    parser.add_argument('--save_losses', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = get_opts()

    torch.manual_seed(args.seed)
    dift = SDFeaturizer(sd_id=args.sd_checkpoint)

    data, H = initialize(args, dift)
    optimize(args, data, initial_transform=H)


if __name__ == '__main__':
    main()
