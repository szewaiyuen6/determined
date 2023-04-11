import os

import torch.nn as nn
import torch

from determined.experimental import client
from typing import Dict, Sequence, Union

# Constants about the data set.
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

from typing import Any


class Flatten(nn.Module):
    def forward(self, *args: TorchData, **kwargs: Any) -> torch.Tensor:
        assert len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        return x.contiguous().view(x.size(0), -1)


model = nn.Sequential(
    nn.Conv2d(NUM_CHANNELS, IMAGE_SIZE, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Dropout2d(0.25),
    nn.Conv2d(32, 64, (3, 3), padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Dropout2d(0.25),
    Flatten(),
    nn.Linear(2304, 512),
    nn.ReLU(),
    nn.Dropout2d(0.5),
    nn.Linear(512, NUM_CLASSES),
)


def get_model():
    # This UUID is from determined UI
    specific_checkpoint = client.get_checkpoint(uuid="6a5a641a-fe9e-49fe-9f96-2b5d6109709b")
    checkpoint_path = specific_checkpoint.download()

    model.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "state_dict.pth"))["models_state_dict"][0]
    )
    return model
