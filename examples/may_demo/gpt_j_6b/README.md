# LLM Inference Demo With Torch Batch Processing API
## Context
This example illustrate using torch batch processing API with HF accelerate to run inference with model not fitting into
one GPU. 

## Use case
The use case we are demoing here is a model hub workflow.

We run inference with an LLM to generate embeddings, then we upload the embeddings onto a vector database.

## Caveat
- This demo only work on top of PR: https://github.com/determined-ai/determined/pull/6807
- Setting CUDA_VISIBLE_DEVICES at the top is crucial as it limits which cuda accelerate uses on each worker.

