# Determined + Ray + Core API + MLFlow
## tl;dr
This demo runs a Ray cluster within a Determined experiment. Integration with Core API and MLFlow allows:
- loading model from either determined checkpoint or from a standalone Mlflow tracking server 
  - the code loads from
  Mlflow but since the example is integrated with CoreAPI, you can also load from a det checkpoint easily if you would like
- Storing metrics in either MLflow model registry (tied to the model that we use for inference) or in a determined checkpoint

The inference output is stored in a s3 bucket in this case.

## Prerequisite:
You should have a standalone MLFlow tracking server set up (note that the MLFlow tracker server needs to be using a SQL 
db as `--backend-store-uri` to enable model registry functionality).

Reference for how to set up the standalone server:
- (set up Mlflow server on ec2) https://medium.com/analytics-vidhya/setup-mlflow-on-aws-ec2-94b8e473618f
- (set up postgres as backend store for mlflow) https://pedro-munoz.tech/how-to-setup-mlflow-in-production/

## Step 1 (optional):
Run the command in training_job_submitter.ipynb in training_job directory to train and upload model to your mlflow server.
Note that you will have to fill in the tracking uri as well as mlflow user name and password.

## Step 2: 
Fill in the tracking uri as well as mlflow username and password in the inference.py under inference directory.

You can simply run `det e create det-expconfig.yaml .`to create an experiment for the inference job (or just run
the same command from inference_job_submitter.ipynb).