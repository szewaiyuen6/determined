import boto3
import botocore
import os
import tempfile

from cellular_dataset import CellularDataset
from model import ModelAndLoss

import determined as det
from determined import core

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import time

from typing import Optional

import json

import logging

S3_BUCKET = "det-swy-benchmark-us-west-2-573932760021"


def get_data_loader(
    root_dir,
    batch_size,
    total_worker,
    rank,
    read_from_s3=False,
    s3_bucket=None,
    s3_additional_path=None,
):
    inference_data = CellularDataset(
        root_dir,
        "test",
        s3_bucket=s3_bucket,
        s3_additional_path=s3_additional_path,
        read_from_s3=read_from_s3,
        normalization="sample",
    )
    sampler = DistributedSampler(inference_data, total_worker, rank, shuffle=False)
    sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    return torch.utils.data.DataLoader(inference_data, batch_sampler=sampler)


def run_inference(core_context, rank, data_loader, latest_checkpoint):
    model = ModelAndLoss()
    model.eval()
    batch = 0
    with torch.no_grad():
        for i, (X, S, I) in enumerate(data_loader):
            print(f"batch id is {i}")
            model.eval_forward(X, S)
            batch = batch + 1

    if core_context.distributed.rank == 0:
        total_batch_completed = sum(core_context.distributed.gather(batch))
        return total_batch_completed
    else:
        core_context.distributed.gather(batch)
        return 0


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except botocore.exceptions.ClientError as e:
        logging.error(e)
        return False
    return True


def main(core_context):
    batch_size = 50
    info = det.get_cluster_info()
    total_worker = len(info.container_addrs)
    rank = info.container_rank
    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_loader = get_data_loader(
            root_dir=tmpdirname,
            batch_size=batch_size,
            total_worker=total_worker,
            rank=rank,
            s3_bucket="det-swy-benchmark-us-west-2-573932760021",
            s3_additional_path="recursion-data",
            read_from_s3=True,
        )
        tic = time.time()
        total_batch_completed = run_inference(core_context, rank, data_loader, latest_checkpoint)
        toc = time.time()
        if rank == 0:
            file_name = f"benchmark_{trial_id}.json"

            full_filename = os.path.join(tmpdirname, f"{file_name}")

            with open(
                full_filename,
                "w",
            ) as f:
                json.dump(
                    {
                        "tic": tic,
                        "toc": toc,
                        "trial_id": trial_id,
                        "flavor": "det_native",
                        "master": info.master_url,
                        "download_from_s3": True,
                        "dataset": "test",
                        "total_worker": total_worker,
                        "total_batch_completed": total_batch_completed,
                        "batch_size": batch_size,
                        "sync_point_count": 1,
                    },
                    f,
                )

            upload_file(full_filename, S3_BUCKET, f"benchmark-output/{file_name}")


def _initialize_distributed_backend() -> Optional[core.DistributedContext]:
    # Pytorch specific initialization
    if torch.cuda.is_available():
        dist.init_process_group(
            backend="nccl",
        )  # type: ignore
        return core.DistributedContext.from_torch_distributed()
    else:
        dist.init_process_group(backend="gloo")  # type: ignore
    return core.DistributedContext.from_torch_distributed()


if __name__ == "__main__":
    distributed = _initialize_distributed_backend()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
