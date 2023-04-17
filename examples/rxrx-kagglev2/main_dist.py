import boto3
import botocore
import os
import pandas as pd

from cellular_dataset import CellularDataset

import determined as det
from determined import core

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import time

from typing import Optional

import json

import logging

S3_BUCKET = "det-swy-benchmark3-us-west-2-573932760021"
STORE_INTERVAL = 100


def get_data_loader(
    root_dir,
    batch_size,
    total_worker,
    rank,
):
    inference_data = CellularDataset(
        root_dir,
        "test",
    )
    sampler = DistributedSampler(inference_data, total_worker, rank, shuffle=False)
    sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    return torch.utils.data.DataLoader(inference_data, batch_sampler=sampler)


def run_inference(core_context, rank, data_loader, trial_id, root_dir):
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
    model.eval()
    batch = 0

    store_interval = STORE_INTERVAL
    output = []
    with torch.no_grad():
        for i, (X, P) in enumerate(data_loader):
            print(f"batch {i}")
            batch = batch + 1
            model(X)

            output.append(P)
            if (i + 1) % store_interval == 0:
                print("upload block")
                filename = f"pred_out_rank_trial_{trial_id}_rank_{rank}_batch_{batch - store_interval + 1}_{batch}"
                save_pred_output(output, root_dir, filename)
                output = []

        if len(output) > 0:
            print("upload last block")
            filename = f"pred_out_rank_trial_{trial_id}_rank_{rank}_batch_last"
            save_pred_output(output, root_dir, filename)

    if core_context.distributed.rank == 0:
        start_wait = time.time()
        total_batch_completed = sum(core_context.distributed.gather(batch))
        end_wait = time.time()
        return [total_batch_completed, start_wait, end_wait]
    else:
        start_wait = time.time()
        core_context.distributed.gather(batch)
        end_wait = time.time()
        return [0, start_wait, end_wait]


def save_pred_output(output, root_dir, filename):
    import csv

    with open(os.path.join(root_dir, filename), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["path"])
        for tuple in output:
            for path in tuple:
                csv_out.writerow([path])

    upload_file(os.path.join(root_dir, filename), S3_BUCKET, f"trial_output/{filename}")

    os.remove(os.path.join(root_dir, filename))


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


def main_from_fs(core_context):
    batch_size = 300
    info = det.get_cluster_info()
    total_worker = len(info.container_addrs)
    rank = info.container_rank
    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id
    read_from_s3 = False
    root_dir_data = "/run/determined/workdir/shared_fs/recursion-data/"
    root_dir = "/run/determined/workdir/shared_fs/"

    data_loader = get_data_loader(
        root_dir=root_dir_data,
        batch_size=batch_size,
        total_worker=total_worker,
        rank=rank,
    )
    tic = time.time()
    [total_batch_completed, start_wait, end_wait] = run_inference(
        core_context, rank, data_loader, trial_id, root_dir
    )
    toc = time.time()
    if rank == 0:
        file_name = f"benchmark_trial_{trial_id}.json"

        full_filename = os.path.join(root_dir, f"{file_name}")
        print(tic)
        print(toc)
        with open(
            full_filename,
            "w",
        ) as f:
            json.dump(
                {
                    "job_start_time": tic,
                    "job_end_time": toc,
                    "trial_id": trial_id,
                    "flavor": "det_native",
                    "download_from_s3": read_from_s3,
                    "dataset": "test",
                    "total_worker": total_worker,
                    "total_batch_completed": total_batch_completed,
                    "batch_size": batch_size,
                    "sync_point_count": 1,
                    "note": "per image run",
                    "store_interval": STORE_INTERVAL,
                },
                f,
            )

        upload_file(full_filename, S3_BUCKET, f"benchmark-output/{file_name}")
    wait_time_file_name = f"benchmark_wait_time_agent_{rank}_trial_{trial_id}.json"
    wait_time_full_file_name = os.path.join(root_dir, f"{wait_time_file_name}")
    with open(
        wait_time_full_file_name,
        "w",
    ) as f:
        json.dump(
            {
                "agent_wait_time": start_wait,
                "agent_end_time": end_wait,
                "rank": rank,
                "trial_id": trial_id,
                "flavor": "det_native",
                "download_from_s3": read_from_s3,
                "dataset": "test",
                "total_worker": total_worker,
                "total_batch_completed": total_batch_completed,
                "batch_size": batch_size,
                "sync_point_count": 1,
                "note": "per image run",
                "store_interval": STORE_INTERVAL,
            },
            f,
        )
    upload_file(wait_time_full_file_name, S3_BUCKET, f"benchmark-output/{wait_time_file_name}")


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
        main_from_fs(core_context)
