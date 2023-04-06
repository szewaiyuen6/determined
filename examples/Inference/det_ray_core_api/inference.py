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

distributed = det.core.DummyDistributedContext()

with det.core.init(distributed=distributed) as core_context:

    # init ray, should get ray cluster address from environment variable
    res = ray.init()

    # This UUID is from determined UI
    specific_checkpoint = client.get_checkpoint(uuid="408e6e0a-b00e-4e93-9cbf-36d057d2fb63")
    checkpoint_path = specific_checkpoint.download()

    # Right now, there is no standardized way to get model definition from checkpoint
    # Latest discussion re: trainer API also requires user to instantiate a trial / model
    # to make use of checkpointed state. We are following the same paradigm here
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork()
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "state_dict.pth")))

    # Put instantiated model into a Ray checkpoint
    checkpoint = TorchCheckpoint.from_model(model=model)
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

    # Store prediction
    predictions.write_json("s3://det-detraydemo-us-west-2-573932760021/prediction_out_10")

    # Save data to determined checkpoint
    checkpoint_metadata = {
        "steps_completed": 1000,
    }
    with core_context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
        with open(os.path.join(path, "metrics.json"), "w") as file_obj:
            json.dump({"metrics_1": 0.256}, file_obj)
