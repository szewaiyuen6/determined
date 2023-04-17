import boto3
import logging
import math
import numpy as np
import os
import pandas as pd
import random
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from pathlib import Path
import PIL

import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import precomputed as P


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)


def get_test_loader(
    root_dir, s3_bucket=None, s3_additional_path=None, read_from_s3=False, exclude_leak=False
):
    test_dataset = CellularDataset(
        root_dir,
        "test",
    )
    return DataLoader(
        test_dataset, 24, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn  # SWY
    )


def get_all_loader(
    root_dir, s3_bucket=None, s3_additional_path=None, read_from_s3=False, exclude_leak=False
):
    test_dataset = CellularDataset(
        root_dir,
        "all",
    )
    return DataLoader(
        test_dataset, 24, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn  # SWY
    )


class CellularDataset(Dataset):
    treatment_classes = 1108

    def __init__(
        self,
        root_dir,
        mode,
    ):

        super().__init__()

        files_list = []
        if mode == "test" or mode == "all":
            full_dir = "/run/determined/workdir/shared_fs/recursion-data/test"
            for root, dirs, files in os.walk(full_dir):
                for file in files:
                    if file.endswith(".png"):
                        files_list.append(os.path.join(root, file))
        if mode == "train" or mode == "all":
            full_dir = "/run/determined/workdir/shared_fs/recursion-data/train"
            for root, dirs, files in os.walk(full_dir):
                for file in files:
                    if file.endswith(".png"):
                        files_list.append(os.path.join(root, file))
        self.data = files_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_path = self.data[i]

        image = PIL.Image.open(image_path).convert("RGB")

        tensor = F.to_tensor(image)

        r = [tensor, image_path]

        return tuple(r)
