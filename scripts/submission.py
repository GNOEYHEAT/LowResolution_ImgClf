import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))+'/../')
sys.path.append(ROOT_DIR)
from utils.config import load_cfg_from_cfg_file

def get_args():
    parser = argparse.ArgumentParser(description='Create submission file')
    parser.add_argument('--dir_path', default="results/dinov2-large", type=str,
                        help="Path where model pred numpy files are stored")
    return parser.parse_args()

def submission(args):

    # load label_dict
    with open(os.path.join(args.dir_path, 'label_dict.json'), 'r') as f:
        label_dict = json.load(f)

    # find paths of np files
    pred_paths = [f for f in os.listdir(args.dir_path) if f.endswith('.npy')]

    # preds_lists
    preds_list = []
    for pred_path in pred_paths:
        full_path = os.path.join(args.dir_path, pred_path)
        pred = np.load(full_path)
        preds_list.append(pred)

    # averaging
    y_pred = np.mean(preds_list, axis=0)
    preds = y_pred.argmax(axis=1)

    # decoding
    label_decoder = {val: key for key, val in label_dict.items()}
    result = [label_decoder[pred] for pred in preds]

    # load sample_submission csv file
    base_cfg = load_cfg_from_cfg_file(f'{ROOT_DIR}/configs/base.yaml')

    data_dir = os.path.abspath(Path(base_cfg.train_csv_path).parent)
    submit = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    submit["label"] = result
    save_path = os.path.join(args.dir_path, 'submission.csv')
    submit.to_csv(save_path, index=False)
    print(f"The submission file was saved successfully : {os.path.abspath(save_path)}")

if __name__ == '__main__':
    args = get_args()
    submission(args)
