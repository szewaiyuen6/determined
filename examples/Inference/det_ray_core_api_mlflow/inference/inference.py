from ray.cluster_utils import Cluster
import ray
import os


import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchCheckpoint, TorchPredictor

import os

from determined.experimental import client

import determined as det

from ray.air import session

import json

import mlflow

distributed = det.core.DummyDistributedContext()

MLFLOW_MODEL_NAME = "mnist_model"
MLFLOW_MODEL_VERSION = "2"

with det.core.init(distributed=distributed) as core_context:

    # init ray, should get ray cluster address from environment variable
    res = ray.init()

    # Set-up MLFlow and load model form it
    mlflow.set_tracking_uri("<MLFLOW_TRACKING_URI>")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "<MLFLOW_USERNAME>"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "<MLFLOW_PASSWORD>"

    loaded_model = mlflow.pytorch.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")

    # Put model loaded from MLFlow into a Ray checkpoint
    checkpoint = TorchCheckpoint.from_model(model=loaded_model)
    batch_predictor = BatchPredictor(checkpoint, TorchPredictor)

    # Reading data in
    # Note that ray.data.from_torch does not support parallel reads
    # This should only be used for small data sets like MNIST
    # For production use case, other ways to read data into ray
    # such as from file systems or from distributed data processing framework
    # should be used.
    # See: https://docs.ray.io/en/latest/data/creating-datasets.html#creating-datasets

    dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    ray_data = ray.data.from_torch(dataset)

    from typing import Dict, Tuple
    import numpy as np
    from PIL.Image import Image
    import torch

    def convert_batch_to_numpy(batch: Tuple[Image, int]) -> Dict[str, np.ndarray]:
        images = np.stack([np.array(image) for image, _ in batch])
        labels = np.array([label for _, label in batch])
        return {"img": images, "label": labels}

    ray_data = ray_data.map_batches(convert_batch_to_numpy).fully_executed()

    # Run batch prediction with two workers
    predictions = batch_predictor.predict(
        ray_data,
        feature_columns=["img"],
        keep_columns=["label"],
        batch_size=64,
        max_scoring_workers=2,
        num_gpus_per_worker=1,
    )

    # Store prediction to s3
    predictions.write_json(
        f"s3://det-detraydemo-us-west-2-573932760021/{MLFLOW_MODEL_NAME}_{MLFLOW_MODEL_VERSION}"
    )

    # [OPTIONAL] Add tag to MLFlow (e.g. evluation metrics)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(
        name=MLFLOW_MODEL_NAME,
        version=MLFLOW_MODEL_VERSION,
        key="prediction_performance",
        value="0.99",
    )

    # [OPTIONAL] Store informatin to inference job determined checkpoint
    checkpoint_metadata = {
        "steps_completed": 1000,
    }
    with core_context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
        with open(os.path.join(path, "metrics.json"), "w") as file_obj:
            json.dump({"metrics_1": 0.256}, file_obj)
