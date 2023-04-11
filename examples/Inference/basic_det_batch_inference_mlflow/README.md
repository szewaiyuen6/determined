# Determined Native Batch Inference Demo + basic mlflow integration
## Context
This demo showcases performing batch inference using Determined's Core API.

We are able to leverage Determined's existing primitives and build a demo that is
- Fault tolerant
- Can be pre-empted and can continue
- Can be monitored on the Determined UI

## Important Prerequisite
- The dataloader has to be deterministic for fault tolerance to work. In that way simply by keeping track of batch 
index, we can avoid recomputing already processed records.
- This demo assumes you to have a standalone MLflow tracking server with models stored. You can also choose to download
the model from a determined checkpoint like the example under the `basic_determine_batch_inference` folder.

## To run
`det e create inference_config.yaml`