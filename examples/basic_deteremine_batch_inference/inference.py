from __future__ import print_function

import torch
import torch.distributed as dist

import torchvision as tv
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler

from typing import Optional

import determined as det

from determined.horovod import hvd
from determined import core, gpu, horovod, profiler, pytorch

from model.model import get_model

import json
import os
import pathlib


def load_state(checkpoint_directory):
    checkpoint_directory = pathlib.Path(checkpoint_directory)
    with checkpoint_directory.joinpath("metadata.json").open("r") as f:
        metadata = json.load(f)
        return metadata


def run_inference(model, data_loader, core_context, rank, latest_checkpoint) -> int:

    model.eval()
    steps_completed = 0

    # Look for checkpoint of the inference job
    # If checkpoint exist, get number of steps completed
    if latest_checkpoint is not None:
        print("Checkpoint is not none")
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            metadata = load_state(path)
            steps_completed = metadata["steps_completed"]
            print(f"Steps completed {steps_completed}")

    with torch.no_grad():
        for batch, (data, target) in enumerate(data_loader):
            # Do not recompute finished batches
            if batch < steps_completed:
                print(f"skipping batch")
                continue

            print(f"Working on batch is {batch}")

            preds = []

            for data_point in data:
                output = model(data_point)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                preds.append(pred[0][0].item())

            file_name = f"inference_out_{rank}_{batch}.json"

            with open(
                os.path.join("/run/determined/workdir/shared_fs/inference_out/", f"{file_name}"),
                "w",
            ) as f:
                json.dump({"predictions": preds}, f)

            # After each batch, synchronize and update number of catches completed
            if core_context.distributed.rank == 0:
                work_completed_this_round = sum(core_context.distributed.gather(len(data)))
                checkpoint_metadata = {
                    "steps_completed": batch,
                    "record_processed": work_completed_this_round,
                }
                with core_context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    with open(os.path.join(path, "batch_completed.json"), "w") as file_obj:
                        json.dump({"batch_completed": batch}, file_obj)
            else:
                core_context.distributed.gather(len(data))

            if core_context.preempt.should_preempt():
                return


def get_data_loader(batch_size, total_worker, rank):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    inference_data = tv.datasets.CIFAR10(
        root=".\\data", train=False, download=True, transform=transform
    )

    sampler = DistributedSampler(inference_data, total_worker, rank, shuffle=False)
    sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    return torch.utils.data.DataLoader(inference_data, batch_sampler=sampler)


def main(core_context):
    batch_size = 200
    info = det.get_cluster_info()
    total_worker = len(info.container_addrs)
    rank = info.container_rank
    latest_checkpoint = info.latest_checkpoint

    data_loader = get_data_loader(batch_size, total_worker, rank)
    run_inference(get_model(), data_loader, core_context, rank, latest_checkpoint)


def _initialize_distributed_backend() -> Optional[core.DistributedContext]:
    info = det.get_cluster_info()

    distributed_backend = det._DistributedBackend()
    if distributed_backend.use_horovod():
        hvd.require_horovod_type("torch", "PyTorchTrial is in use.")
        hvd.init()
        return core.DistributedContext.from_horovod(horovod.hvd)
    elif distributed_backend.use_torch():
        if torch.cuda.is_available():
            dist.init_process_group(
                backend="nccl",
            )  # type: ignore
        else:
            dist.init_process_group(backend="gloo")  # type: ignore
        return core.DistributedContext.from_torch_distributed()
    elif info and (len(info.container_addrs) > 1 or len(info.slot_ids) > 1):
        raise ValueError(
            "In multi-slot managed cluster training, you must wrap your training script with a "
            "distributed launch layer such as determined.launch.torch_distributed or "
            "determined.launch.horovod."
        )
    return None


if __name__ == "__main__":
    distributed = _initialize_distributed_backend()
    with det.core.init(distributed=distributed) as core_context:
        main(core_context)
