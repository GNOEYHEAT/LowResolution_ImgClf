import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, df, data_dir, img_path, processor, is_test=False, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.img_path = img_path
        self.processor = processor
        self.is_test = is_test
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row[self.img_path]
        img_path = os.path.join(self.data_dir, Path(img_path).relative_to("data"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.is_test:

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            encoding = self.processor(
                images=image,
                return_tensors="pt"
            )

            encoding["labels"] = torch.tensor(row['label'], dtype=torch.long)

            for k, v in encoding.items():
                encoding[k] = v.squeeze()

            return encoding

        encoding = self.processor(
            images=image,
            return_tensors="pt"
        )

        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding
