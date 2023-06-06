# Run inference API with grid search
## tl;dr
This is a proof of concept of running each model replica as an individual trial. This set up is not ideal as we are
passing rank and number of replicas to the trial as hyperparameters, when they are clearly not.

The goal is to showcase limitation of current system to support this paradigm.

Caveat: This example relies on a variation of [this PR](https://github.com/determined-ai/determined/pull/6807) in the `_torch_batch_process_grid.py` file. 

## Use case
- Run a LLM (larger than one GPU if running on AWS g4dn12xlarge) with model parallel and cpu offloading with HF 
  Accelerate
- The LLM generates embeddings which are persisted to disk during checkpointing
- After embedding generation completes, upload all embeddings to Pinecone vector DB

## How to run the example
- Sign up for Pincone vector DB
- Replace Pinecone credential placeholders with actual credentials in `grid_new_apu.py`
- run det e `create grid_new_api_config.yaml`

## Notes

### Why do we want to run each model replica as its own trial / job?
- Better fault tolerance as one replica failure would not affect other replicas
- More compatible with model parallel paradigm. We do not need complex accounting of which GPU belongs to which model
  replicas. 

### What are some challenges of running each model replica as its own trial
- Currently, our metrics aggregation support(e.g. metric reducers) only work within a single trial. We would need
  support for cross trial metric aggregation
  - We use the term trial here because it is currently the only contruct we have to run a job. Ideally, we will have
    more generic task / actor construct that better fits what we are actually trying to do here.
