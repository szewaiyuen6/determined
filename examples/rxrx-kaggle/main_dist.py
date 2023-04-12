from cellular_dataset import CellularDataset
from model import ModelAndLoss

import determined as det
from determined import core

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from typing import Optional


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


def run_inference(core_context, rank, data_loader):

    i = 0
    model = ModelAndLoss()
    model.eval()
    for i, (X, S, I) in enumerate(data_loader):
        print(I)
        try:
            out = model.eval_forward(X, S)
            print(out)
        except Exception as e:
            print(e)

        print(out.size())
        print("out")
        print(out)

        i = i + 1
        if i > 10:
            break


def main(core_context):
    batch_size = 50
    info = det.get_cluster_info()
    total_worker = len(info.container_addrs)
    rank = info.container_rank
    # latest_checkpoint = info.latest_checkpoint

    data_loader = get_data_loader(
        root_dir="/run/determined/workdir/shared_fs/recursion-data",
        batch_size=batch_size,
        total_worker=total_worker,
        rank=rank,
        s3_bucket="det-swy-benchmark-us-west-2-573932760021",
        s3_additional_path="recursion-data",
        read_from_s3=True,
    )
    run_inference(core_context, rank, data_loader)


def main2(core_context):
    print("hahaha")


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
