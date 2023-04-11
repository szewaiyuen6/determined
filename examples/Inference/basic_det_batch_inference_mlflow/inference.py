from __future__ import print_function

import torch
import torch.distributed as dist

import torchvision as tv
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler

from typing import Optional

import determined as det

from determined import core, gpu, horovod, profiler, pytorch

import json
import os
import pathlib

import mlflow

import boto3


def load_state(checkpoint_directory):
    checkpoint_directory = pathlib.Path(checkpoint_directory)
    with checkpoint_directory.joinpath("metadata.json").open("r") as f:
        metadata = json.load(f)
        return metadata


def run_inference(
    model,
    data_loader,
    core_context,
    rank,
    latest_checkpoint,
    mlflow_model_name,
    mlflow_model_version,
) -> int:

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

            full_filename = os.path.join(inference_output_dir, f"{file_name}")
            for pred in preds:
                output.append(pred[0].item())

            with open(
                full_filename,
                "w",
            ) as f:
                json.dump({"predictions": output}, f)

            upload_file(
                full_filename,
                "det-detraydemo-us-west-2-573932760021",
                f"det_native_pred_out/{file_name}",
            )

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

    # [OPTIONAL] Add tag to MLFlow (e.g. evluation metrics)
    if rank == 0:
        client = mlflow.MlflowClient()
        client.set_model_version_tag(
            name=mlflow_model_name,
            version=mlflow_model_version,
            key="prediction_performance_det_native",
            value="0.99",
        )


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
    except ClientError as e:
        logging.error(e)
        return False
    return True


def main(core_context):
    batch_size = 200
    info = det.get_cluster_info()
    total_worker = len(info.container_addrs)
    rank = info.container_rank
    latest_checkpoint = info.latest_checkpoint

    data_loader = get_data_loader(batch_size, total_worker, rank)

    # Get model from MLFlow
    MLFLOW_MODEL_NAME = "cifar10_model"
    MLFLOW_MODEL_VERSION = "2"
    mlflow.set_tracking_uri("http://ec2-35-161-33-234.us-west-2.compute.amazonaws.com")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "<MLFLOW_USERNAME>"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "<MLFLOW_PASSWORD>"
    loaded_model = mlflow.pytorch.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")

    run_inference(
        loaded_model,
        data_loader,
        core_context,
        rank,
        latest_checkpoint,
        MLFLOW_MODEL_NAME,
        MLFLOW_MODEL_VERSION,
    )


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
