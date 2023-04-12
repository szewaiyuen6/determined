from cellular_dataset import CellularDataset
from model import ModelAndLoss

import torch
from torch.utils.data.distributed import DistributedSampler


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


if __name__ == "__main__":
    model = ModelAndLoss()
    test_loader = get_data_loader(
        root_dir="/run/determined/workdir/shared_fs/recursion-data",
        batch_size=24,
        total_worker=2,
        rank=0,
        s3_bucket="det-swy-benchmark-us-west-2-573932760021",
        s3_additional_path="recursion-data",
        read_from_s3=True,
    )
    i = 0
    model = ModelAndLoss()
    for i, (X, S, I) in enumerate(test_loader):
        out = model.eval_forward(X, S)
        print(out.size())
        print("out")
        print(out)
        print(I)
        i = i + 1
        if i > 0:
            break
