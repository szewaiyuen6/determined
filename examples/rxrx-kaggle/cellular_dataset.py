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


def tta(images):
    """Augment all images in a batch and return list of augmented batches"""
    tta_size = 1
    tta = 3
    ret = []
    n1 = math.ceil(tta ** 0.5)
    n2 = math.ceil(tta / n1)
    k = 0
    for i in range(n1):
        for j in range(n2):
            if k >= tta:
                break

            dw = round(tta_size * images.size(2))
            dh = round(tta_size * images.size(3))
            w = i * (images.size(2) - dw) // max(n1 - 1, 1)
            h = j * (images.size(3) - dh) // max(n2 - 1, 1)

            imgs = images[:, :, w : w + dw, h : h + dh]
            if k & 1:
                imgs = imgs.flip(3)
            if k & 2:
                imgs = imgs.flip(2)
            if k & 4:
                imgs = imgs.transpose(2, 3)

            ret.append(nn.functional.interpolate(imgs, images.size()[2:], mode="nearest"))
            k += 1

    return ret


def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 10 ** 9) + worker_id)


def get_train_val_loader(data, predict=False):
    scale_aug = 0.5

    def train_transform1(image):
        if random.random() < 0.5:
            image = image[:, ::-1, :]
        if random.random() < 0.5:
            image = image[::-1, :, :]
        if random.random() < 0.5:
            image = image.transpose([1, 0, 2])
        image = np.ascontiguousarray(image)

        # if scale_aug != 1:
        #     size = random.randint(round(512 * scale_aug), 512)
        #     x = random.randint(0, 512 - size)
        #     y = random.randint(0, 512 - size)
        #     image = image[x:x + size, y:y + size]
        #     image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)

        return image

    def train_transform2(image):
        pw_aug = (0.1, 0.1)
        a, b = np.random.normal(1, pw_aug[0], (6, 1, 1)), np.random.normal(0, pw_aug[1], (6, 1, 1))
        a, b = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
        return image * a + b

    if not predict:
        train_dataset = CellularDataset(
            data,
            "train_all_controls",
            transform=(train_transform1, train_transform2),
            cv_number=0,
            split_seed=0,
            normalization="sample",
        )
        train = DataLoader(
            train_dataset,
            24,
            shuffle=True,
            drop_last=True,
            num_workers=10,
            worker_init_fn=worker_init_fn,
        )

    for i in range(1 if not predict else 2):
        dataset = CellularDataset(
            data, "val" if i == 0 else "train", cv_number=0, split_seed=0, normalization="sample"
        )
        loader = DataLoader(
            dataset, 24, shuffle=False, num_workers=10, worker_init_fn=worker_init_fn
        )
        if i == 0:
            val = loader
        else:
            train = loader

    assert len(set(train.dataset.data).intersection(set(val.dataset.data))) == 0
    return train, val


def get_test_loader(
    root_dir, s3_bucket=None, s3_additional_path=None, read_from_s3=False, exclude_leak=False
):
    test_dataset = CellularDataset(
        root_dir,
        "test",
        s3_bucket=s3_bucket,
        s3_additional_path=s3_additional_path,
        read_from_s3=read_from_s3,
        normalization="sample",
    )
    return DataLoader(
        test_dataset, 24, shuffle=False, num_workers=1, worker_init_fn=worker_init_fn  # SWY
    )


class CellularDataset(Dataset):
    treatment_classes = 1108

    def __init__(
        self,
        root_dir,
        mode,
        s3_bucket=None,
        s3_additional_path=None,
        read_from_s3=False,
        split_seed=0,
        cv_number=0,
        transform=None,
        normalization="global",
    ):
        """
        :param split_seed: seed for train/val split of labeled experiments and HUVEC-18
        :param mode: possible choices:
                        train -- dataset containing only non-control images from training set
                        train_controls -- dataset containing non-control and control images from training set
                        train_all_controls -- dataset containing non-control and control images from training set and
                                              control images from validation and test set
                        val -- dataset containing only non-control images from validation set
                        test -- dataset containing only non-control images from test set
                        test_noleak -- dataset containing only non-control images from test set excluding HUVEC-18
        :param transform: tuple of 2 functions for image transformation. First is called right after loading with image
                          in numpy format. Second is called after normalization and converting to tensor
        """

        super().__init__()

        self.root = Path(root_dir)

        self.s3_bucket = s3_bucket

        self.read_from_s3 = read_from_s3

        self.transform = transform

        self.s3_additional_path = s3_additional_path

        self.s3_client = boto3.client("s3")

        self.data_indices = None

        self.mode = mode

        assert normalization in ["global", "experiment", "sample"]
        self.normalization = normalization

        if mode == "train_controls":
            mode = "train"
            move_controls = True
            all_controls = False
        elif mode == "train_all_controls":
            mode = "train"
            move_controls = True
            all_controls = True
        else:
            move_controls = False
            all_controls = False

        assert mode in ["train", "val", "test"]
        self.mode = mode

        csv_file_name = "train.csv" if mode in ["train", "val"] else "test.csv"
        csv = self._read_csv_to_pd(csv_file_name)
        csv_controls_file_name = (
            "train_controls.csv" if mode in ["train", "val"] else "test_controls.csv"
        )
        csv_controls = self._read_csv_to_pd(csv_controls_file_name)

        if all_controls:
            csv_controls_test = self._read_csv_to_pd("test_controls.csv")
        self.data = []  # (experiment, plate, well, site, cell_type, sirna or None)
        experiments = {}
        for row in chain(
            csv.iterrows(),
            csv_controls.iterrows(),
            *([csv_controls_test.iterrows()] if all_controls else []),
        ):
            r = row[1]
            if r.experiment == "HUVEC-18":
                # problematic experiment that requires additional processing
                continue
            typ = r.experiment[: r.experiment.find("-")]
            self.data.append(
                (r.experiment, r.plate, r.well, 1, typ, r.sirna if hasattr(r, "sirna") else None)
            )
            self.data.append(
                (r.experiment, r.plate, r.well, 2, typ, r.sirna if hasattr(r, "sirna") else None)
            )
            if typ not in experiments:
                experiments[typ] = set()
            experiments[typ].add(r.experiment)
        if mode in ["train", "val"]:
            data_dict = {(e, p, w): sir for e, p, w, s, typ, sir in self.data}
            for row in self._read_csv_to_pd("test.csv").iterrows():
                r = row[1]
                typ = r.experiment[: r.experiment.find("-")]

            if not all_controls:
                for row in self._read_csv_to_pd("test_controls.csv").iterrows():
                    r = row[1]
                    typ = r.experiment[: r.experiment.find("-")]

        self.cell_types = sorted(experiments.keys())

        all_data = self.data.copy()

        assert len(set(self.data)) == len(self.data)
        assert len(set(all_data)) == len(all_data)

        logging.info("{} dataset size: data: {}".format(mode, len(self.data)))

    def filter(self, func=None):
        """
        Filter dataset by given function. If function is not specified, it will clear current filter
        :param func: func((index, (experiment, plate, well, site, cell_type, sirna or None))) -> bool
        """
        if func is None:
            self.data_indices = None
        else:
            self.data_indices = list(filter(lambda i: func(i, self.data[i]), range(len(self.data))))

    def _read_csv_to_pd(self, file_identifier):
        if not self.read_from_s3:
            return pd.read_csv(self.root / file_identifier)

        # S3 path
        # make sure s3fs is installed for this code path
        if self.s3_additional_path is None:
            return pd.read_csv(
                f"s3://{self.s3_bucket}/{file_identifier}",
            )
        else:
            return pd.read_csv(
                f"s3://{self.s3_bucket}/{self.s3_additional_path}/{file_identifier}",
            )

    def __len__(self):
        return len(self.data_indices if self.data_indices is not None else self.data)

    def __getitem__(self, i):
        i = self.data_indices[i] if self.data_indices is not None else i
        d = self.data[i]

        images = []
        for channel in range(1, 7):

            dir = self.mode
            local_path_to_dir = self.root / dir / d[0] / "Plate{}".format(d[1])
            local_path_to_file = (
                self.root
                / dir
                / d[0]
                / "Plate{}".format(d[1])
                / "{}_s{}_w{}.png".format(d[2], d[3], channel)
            )

            if not local_path_to_file.exists() and self.read_from_s3:
                if self.s3_additional_path is None:
                    file_identifier = (
                        dir
                        + "/"
                        + str(d[0])
                        + "/"
                        + "Plate{}".format(d[1])
                        + "/"
                        + "{}_s{}_w{}.png".format(d[2], d[3], channel)
                    )
                else:
                    file_identifier = (
                        self.s3_additional_path
                        + "/"
                        + dir
                        + "/"
                        + str(d[0])
                        + "/"
                        + "Plate{}".format(d[1])
                        + "/"
                        + "{}_s{}_w{}.png".format(d[2], d[3], channel)
                    )
                    s3_path = f"s3://{self.root}/{self.s3_additional_path}/{file_identifier}"

                if not os.path.exists(local_path_to_dir):
                    os.makedirs(local_path_to_dir)
                print("Download from")
                print(s3_path)
                self.s3_client.download_file(self.s3_bucket, file_identifier, local_path_to_file)

            image = PIL.Image.open(local_path_to_file)

            # applying grayscale method
            image = PIL.ImageOps.grayscale(image)
            images.append(image)

            assert images[-1] is not None

        image = np.stack(images, axis=-1)

        if self.transform is not None:
            image = self.transform[0](image)

        image = F.to_tensor(image)

        if self.normalization == "experiment":
            pixel_mean = torch.tensor(P.pixel_stats[d[0]][0]) / 255
            pixel_std = torch.tensor(P.pixel_stats[d[0]][1]) / 255
        elif self.normalization == "global":
            pixel_mean = (
                torch.tensor(list(map(lambda x: x[0], P.pixel_stats.values()))).mean(0) / 255
            )
            pixel_std = (
                torch.tensor(list(map(lambda x: x[1], P.pixel_stats.values()))).mean(0) / 255
            )
        elif self.normalization == "sample":
            pixel_mean = image.mean([1, 2])
            pixel_std = image.std([1, 2]) + 1e-8
        else:
            assert 0

        image = (image - pixel_mean.reshape(-1, 1, 1)) / pixel_std.reshape(-1, 1, 1)

        if self.transform is not None:
            image = self.transform[1](image)

        cell_type = nn.functional.one_hot(
            torch.tensor(self.cell_types.index(d[-2]), dtype=torch.long), len(self.cell_types)
        ).float()

        r = [image, cell_type, torch.tensor(i, dtype=torch.long)]
        return tuple(r)
