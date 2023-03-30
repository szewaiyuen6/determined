import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import determined as det

import os

# Download training dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test dataset
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Pass data into data loader
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()

# NEW: given a checkpoint_directory of type pathlib.Path, save our state to a file.
# You can save multiple files, and use any file names or directory structures.
# All files nested under `checkpoint_directory` path will be included into the checkpoint.
def save_state(x, steps_completed, trial_id, checkpoint_directory):
    with checkpoint_directory.joinpath("state").open("w") as f:
        f.write(f"{x},{steps_completed},{trial_id}")


# NEW: given a checkpoint_directory, load our state from a file.
def load_state(trial_id, checkpoint_directory):
    checkpoint_directory = pathlib.Path(checkpoint_directory)
    with checkpoint_directory.joinpath("state").open("r") as f:
        x, steps_completed, ckpt_trial_id = [field for field in f.read().split(",")]
    if ckpt_trial_id == trial_id:
        return x, steps_completed
    else:
        # This is a new trial; load the "model weight" but not the batch count.
        return x, 0


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)


# Training function
def train(
    dataloader, model, loss_fn, optimizer, core_context, epoch, trial_id, latest_checkpoint=None
):

    if latest_checkpoint is not None:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            model, starting_batch = load_state(trial_id, path)
        print(f"Starting batch is {starting_batch}")

    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    core_context.train.report_training_metrics(
        steps_completed=epoch, metrics={"train_loss": train_loss, "correct": correct}
    )

    # Save states
    checkpoint_metadata = {"steps_completed": epoch}
    with core_context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
        save_state(model.state_dict(), epoch, trial_id, path)
        torch.save(model.state_dict(), os.path.join(path, "state_dict.pth"))


# Testing function
def test(dataloader, model, loss_fn, core_context, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    core_context.train.report_validation_metrics(
        steps_completed=epoch, metrics={"test_loss": test_loss, "correct": correct}
    )
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_runner(core_context, latest_checkpoint, trial_id, lr):
    # Training the model
    epochs = 5

    # Loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            core_context,
            t,
            trial_id,
            latest_checkpoint,
        )
        test(test_dataloader, model, loss_fn, core_context, t)
    print("Done!")


if __name__ == "__main__":
    # NEW: use the ClusterInfo API to access information about the current running task.  We
    # choose to extract the information we need from the ClusterInfo API here and pass it into
    # main() so that you could eventually write your main() to run on- or off-cluster.
    info = det.get_cluster_info()
    assert info is not None, "this example only runs on-cluster"
    print(f"info is {info}")
    print("master_url", info.master_url)
    print("task_id", info.task_id)
    print("allocation_id", info.allocation_id)
    print("session_token", info.session_token)

    if info.task_type == "TRIAL":
        print("trial.id", info.trial.trial_id)
        print("trial.hparams", info.trial.hparams)

    latest_checkpoint = info.latest_checkpoint
    print("latest check point", latest_checkpoint)
    trial_id = info.trial.trial_id

    # NEW: get the hyperaparameter values chosen for this trial.
    hparams = info.trial.hparams

    with det.core.init() as core_context:
        train_runner(
            core_context=core_context,
            latest_checkpoint=latest_checkpoint,
            trial_id=trial_id,
            lr=hparams["lr"],
        )
