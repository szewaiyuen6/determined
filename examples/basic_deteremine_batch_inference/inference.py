from __future__ import print_function

import torch
import torch.distributed as dist

import torchvision as tv
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler

from typing import Optional

import determined as det

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
    records_processed = 0

    # I set up my AWS cluster with file system, this is where the fs is mounted to my container
    inference_output_dir = "/run/determined/workdir/shared_fs/inference_out/"
    # The first worker will create it, and exist_ok option makes sure subsequent workers
    # do not run into error
    pathlib.Path.mkdir(pathlib.Path(inference_output_dir), parents=True, exist_ok=True)

    # Look for checkpoint of the inference job
    # If checkpoint exist, get number of steps completed
    if latest_checkpoint is not None:
        print("Checkpoint is not none")
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            metadata = load_state(path)
            steps_completed = metadata["steps_completed"]
            records_processed = metadata["records_processed"]
            print(f"Steps completed {steps_completed}")

    with torch.no_grad():
        for batch, (data, target) in enumerate(data_loader):
            # Do not recompute finished batches
            if batch < steps_completed:
                print(f"skipping batch")
                continue

            print(f"Working on batch is {batch}")

            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)

            file_name = f"inference_out_{rank}_{batch}.json"

            output = []

            for pred in preds:
                output.append(pred[0].item())

            with open(
                os.path.join(inference_output_dir, f"{file_name}"),
                "w",
            ) as f:
                json.dump({"predictions": output}, f)

            # After each batch, synchronize and update number of catches completed
            if core_context.distributed.rank == 0:
                work_completed_this_round = sum(core_context.distributed.gather(len(data)))
                records_processed += work_completed_this_round
                checkpoint_metadata = {
                    "steps_completed": batch,
                    "records_processed": records_processed,
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
