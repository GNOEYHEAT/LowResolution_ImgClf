
from torchvision.transforms import v2
def get_augment(augment_mode:str, num_class:int):
    if augment_mode=='cutmix':
        return v2.CutMix(num_classes=num_class)
    elif augment_mode=='mixup':
        return v2.MixUp(num_classes=num_class)
    elif augment_mode=='cutmix_or_mixup':
        cutmix = v2.CutMix(num_classes=num_class)
        mixup = v2.MixUp(num_classes=num_class)
        return v2.RandomChoice([cutmix, mixup])

from torch.utils.data import default_collate

class Collator(object):
    def __init__(self, **params):
        self.params = params

    def __call__(self, batch):
        data = default_collate(batch)
        augment = get_augment(self.params['augment_mode'], self.params['num_class'])
        data = augment(data['pixel_values'], data['labels'])
        data_dict = {
            'pixel_values': data[0],
            'labels': data[1],
        }
        return data_dict

import pandas as pd
def process_dataframe(csv_path:str, label_dict=None ,is_train=False):
    df = pd.read_csv(csv_path)
    df["img_path"] = df["img_path"].apply(lambda x: "data" + x[1:])

    if is_train:
        df["upscale_img_path"] = df["upscale_img_path"].apply(lambda x: "data" + x[1:])
        df['label'] = df['label'].apply(lambda x : label_dict[x])
    return df

import numpy as np
def get_label_dict(csv_path:str):

    df = pd.read_csv(csv_path)
    label_unique = sorted(np.unique(df['label']))
    label_dict = {key: value for key, value in zip(label_unique, range(len(label_unique)))}
    return label_dict

