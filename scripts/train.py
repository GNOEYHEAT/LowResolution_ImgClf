import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoImageProcessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))+'/../')
sys.path.append(ROOT_DIR)

from dataset.base import ImageDataset
from dataset.utils import Collator, get_label_dict, process_dataframe
from models.torch_model import ImageModel
from models.pl_module import ImageClassifier
from utils.utils import set_seeds
from utils.config import load_cfg_from_cfg_file, merge_cfg_from_dict

def get_args():
    parser = argparse.ArgumentParser(description='Run files [train.py, test.py]')

    parser.add_argument('--cfg_path', default=f"{ROOT_DIR}/configs/dinov2-large.yaml", type=str,
                        help="Path to the config file in yaml format")
    parser.add_argument('--save_path', default="results/dinov2-large", type=str,
                        help="Path to the saved models")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of "
                        "workers for each data loader")
    # parser.add_argument('--device', default='0', type=str,
    #                     help="GPU indices ""comma separated, e.g. '0,1' ")
    parser.add_argument('--device', nargs='+', default=[0], type=int,
                        help="GPU indices")
    return parser.parse_args()

def train_one_fold(i, train_df, val_df, test_df, processor, cfg):

    # setup dataset
    data_dir = os.path.abspath(Path(cfg.train_csv_path).parent)
    train_ds_low = ImageDataset(train_df, data_dir, "img_path", processor=processor,  is_test=False)
    train_ds_high = ImageDataset(train_df, data_dir, "upscale_img_path", processor=processor, is_test=False)
    train_ds = train_ds_low + train_ds_high
    val_ds = ImageDataset(val_df, data_dir,"img_path", processor=processor, is_test=False)
    test_ds = ImageDataset(test_df, data_dir,"img_path", processor=processor, is_test=True)

    # setup dataloaders

    train_dataloader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=Collator(augment_mode=cfg.aug_mode, num_class=cfg.num_class)
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_dataloader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = ImageClassifier(ImageModel(cfg), cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=cfg.save_path, filename=f"{cfg.img_model_name.split(os.sep)[-1]}_{i}",
            monitor="val_f1", mode="max"
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.epochs, accelerator="auto", callbacks=callbacks,
        precision=cfg.mixed_precision,
        devices=cfg.device,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    ckpt = torch.load(os.path.join(cfg.save_path, f'{cfg.img_model_name.split(os.sep)[-1]}_{i}.ckpt'), map_location=torch.device('cuda'))
    model.load_state_dict(ckpt['state_dict'])

    # validate using val_df.
    eval_dict = trainer.validate(model, dataloaders=val_dataloader)[0]

    # predict using test_df
    y_preds = trainer.predict(model, dataloaders=test_dataloader)
    y_pred = np.vstack(y_preds)
    val_score = eval_dict['val_f1']
    np.save(f'{cfg.save_path}/{cfg.img_model_name.split(os.sep)[-1]}_{i}_{val_score}', y_pred)

    return val_score


def train(args):
    # setup configuration
    base_cfg = load_cfg_from_cfg_file(f'{ROOT_DIR}/configs/base.yaml')
    cfg = load_cfg_from_cfg_file(args.cfg_path)

    cfg = merge_cfg_from_dict(cfg, base_cfg)
    cfg = merge_cfg_from_dict(cfg, args.__dict__)

    # setup seed
    set_seeds(cfg.seed)

    # setup dataframe
    label_dict = get_label_dict(cfg.train_csv_path)
    train_df = process_dataframe(cfg.train_csv_path, label_dict, is_train=True)
    test_df = process_dataframe(cfg.test_csv_path, is_train=False)
    cfg.num_class = len(label_dict)

    # setup processor
    processor = AutoImageProcessor.from_pretrained(cfg.img_model_name)

    # train all fold
    os.makedirs(cfg.save_path, exist_ok=True)
    skf = StratifiedKFold(n_splits=cfg.cv, shuffle=True, random_state=cfg.seed)

    val_f1_list = []

    for i, (train_index, val_index) in enumerate(skf.split(train_df, train_df["label"])):
        temp_train_df = train_df.iloc[train_index]
        temp_val_df = train_df.iloc[val_index]

        ## train one fold and predict
        val_f1 = train_one_fold(i, temp_train_df, temp_val_df, test_df, processor, cfg)

        val_f1_list.append(val_f1)

    val_f1_mean = np.mean(val_f1_list)
    print(f"val_f1_mean: {val_f1_mean}")


if __name__ == '__main__':
    args = get_args()
    train(args)
