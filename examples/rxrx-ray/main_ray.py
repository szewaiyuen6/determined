import ray
import torch
import numpy as np
import time
import json
import determined as det
import os
import boto3
import botocore
import logging

S3_BUCKET = "det-swy-benchmark3-us-west-2-573932760021"


class CallableCls:
    def __init__(self):
        self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        self.model.eval()

    def __call__(self, batch):
        image = batch.loc[:, "image"]
        image = image.to_numpy()
        image_stacked = np.stack(image, axis=-1)
        row = image_stacked.shape[3]
        channel = image_stacked.shape[2]
        w = image_stacked.shape[1]
        h = image_stacked.shape[0]
        image_stacked = image_stacked.reshape(row, channel, w, h)

        tensor = torch.from_numpy(image_stacked)
        tensor = tensor / 255
        with torch.no_grad():
            self.model(tensor)

        path = batch.loc[:, "path"]
        df = path.to_frame()
        return df


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


def main():
    print("inside main")
    batch_size = 100
    info = det.get_cluster_info()
    total_worker = len(info.container_addrs)

    trial_id = info.trial.trial_id
    root_dir = "/run/determined/workdir/shared_fs/"

    ray.init()

    tic = time.time()
    path = "/run/determined/workdir/shared_fs/recursion-data/test"
    ds = ray.data.read_images(path, include_paths=True, mode="RGB")

    out = ds.map_batches(
        CallableCls, batch_size=batch_size, compute=ray.data.ActorPoolStrategy(1, 8)
    )
    out.write_csv("s3://" + S3_BUCKET + "/" + "ray_out")
    toc = time.time()

    print("tic")
    print(tic)
    print("toc")
    print(toc)
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
                "flavor": "ray",
                "download_from_s3": False,
                "dataset": "test",
                "total_worker": total_worker,
                "total_batch_completed": "automatic_by_ray",
                "batch_size": batch_size,
                "sync_point_count": 0,
                "note": "per image run",
                "store_interval": "-1",
            },
            f,
        )

    upload_file(full_filename, S3_BUCKET, f"benchmark-output/{file_name}")


if __name__ == "__main__":
    main()
