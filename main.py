import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from scripts.train import train
from scripts.submission import submission

def get_args():
    parser = argparse.ArgumentParser(description='Run files [scripts/train.py, scripts/submission.py]')

    parser.add_argument('--cfg_path', default=f"{ROOT_DIR}/configs/dinov2-large.yaml", type=str,
                        help="Path to the config file in yaml format")
    parser.add_argument('--save_path', default="results/dinov2-large", type=str,
                        help="Path to the saved models")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of "
                        "workers for each data loader")
    parser.add_argument('--device', nargs='+', default=[0], type=int,
                        help="GPU indices")
    return parser.parse_args()

def main(args):
    train(args)
    submission(args)

if __name__ == "__main__":
    args = get_args()
    args.dir_path = args.save_path
    main(args)