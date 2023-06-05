import logging
import os
import pathlib
import time
import uuid

import pinecone
import torch
from _torch_batch_process_grid import TorchBatchProcessor, torch_batch_process
from accelerate import init_empty_weights
from datasets import Dataset, load_dataset
from torch.profiler import ProfilerActivity
from transformers import AutoConfig, AutoModelForCausalLM, BertModel, BertTokenizer

import determined as det

PINCONE_ENV = "<PINECONE_ENV>"
API_KEY = "<PINECONE_API_KEY>"
pinecone.init(api_key=API_KEY, environment=PINCONE_ENV)


class MyProcessor(TorchBatchProcessor):
    def __init__(self, context):
        """
        Initialize model
        """
        checkpoint = "EleutherAI/gpt-j-6B"
        config = AutoConfig.from_pretrained(checkpoint)
        config.output_hidden_states = True
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
            model.tie_weights()

        from accelerate import load_checkpoint_and_dispatch

        self.model = load_checkpoint_and_dispatch(
            model,
            "shared_fs/test_inference_api/test_hf_accelerate/sharded-gpt-j-6B",
            device_map="auto",
            no_split_module_classes=["GPTJBlock"],
        )

        print(self.model.hf_device_map)
        """
        Initialize tokenizer
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        """
        Initialize other variables
        """
        self.output_list = []
        self.last_index = 0
        self.context = context
        self.rank = self.context.get_distributed_context().get_rank()
        self.output_dir = f"/run/determined/workdir/shared_fs/gpt_6b_output/worker_{self.rank}"
        self.index = pinecone.Index("swy-test")

    def process_batch(self, batch, batch_idx) -> None:
        with torch.no_grad():
            start = time.time()
            for idx, X in enumerate(batch["line_text"]):
                inputs = self.tokenizer(X, return_tensors="pt")
                inputs = inputs["input_ids"].to(0)

                output = self.model(inputs)

                embeddings = output.hidden_states[-1]
                embeddings = torch.mean(embeddings, dim=1)
                print(embeddings.shape)
                embeddings = embeddings.to("cpu")
                input_id = batch["id"][idx]
                record = {"embedding": embeddings, "id": f"office_{input_id}", "text": X}
                self.output_list.append(record)

            end = time.time()
            through_put = len(batch) / (end - start)
            print(f"Throughput is {through_put}")

        self.last_index = batch_idx

    def on_checkpoint_start(self):
        if len(self.output_list) == 0:
            return
        file_name = f"prediction_output_{self.last_index}"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        file_path = pathlib.PosixPath(self.output_dir, file_name)
        torch.save(self.output_list, file_path)
        self.output_list = []

    def on_finish(self):
        vector = []
        for filename in os.listdir(self.output_dir):
            file_path = pathlib.PosixPath(self.output_dir, filename)

            batches = torch.load(file_path)

            for batch in batches:
                id = batch["id"]
                embeddings = batch["embedding"][0].tolist()

                vector.append(
                    (
                        id,  # Vector ID
                        embeddings,  # Dense vector values
                        {"text": batch["text"]},  # Vector metadata
                    )
                )

                if len(vector) > 20:
                    upsert_response = self.index.upsert(vectors=vector, namespace="testing_1")
                    print(upsert_response)
                    vector = []

        if len(vector) > 0:
            upsert_response = self.index.upsert(vectors=vector, namespace="testing_1")
            print(upsert_response)


if __name__ == "__main__":
    dataset = load_dataset("jxm/the_office_lines")
    dataset = dataset["test"]
    dataset = dataset.filter(lambda x: x["deleted"] is False)
    dataset = dataset.sort(["season", "episode", "scene"])
    dataset = dataset.map(lambda x: {"text": f'{x["speaker"]}: {x["line_text"]}'})
    dataset = dataset.select(list(range(500)))
    info = det.get_cluster_info()

    hparams = info.trial.hparams
    num_replica = hparams["num_replica"]
    rank = hparams["rank"]

    torch_batch_process(
        MyProcessor,
        dataset,
        batch_size=10,
        checkpoint_interval=5,
        num_replica=num_replica,
        data_shard_rank=rank,
    )
