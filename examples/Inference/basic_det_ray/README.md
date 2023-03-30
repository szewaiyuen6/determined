# Basic Demo of running inference with Ray inside of Determined
## Set up
This code is run in a Determined native Jupyter notebook on AWS. 

On AWS, it is recommended to start your determined cluster with the option `--deployment-type efs` or `fsx`
so that the notebook is automatically mounted on a shared file system (`/mnt/efs` on the host, and `/run/determined/workdir/shared_fs` (i.e. `$CWD/shared_fs`) in the container)

On GCP, it may just work out of the box (but I have not tried)

## Step 1 [optional]: kick off a determined training job
If you already have access to a determined checkpoint you can skip this step.

For completeness, this demo start with running the `training_job_sumitter.ipynb` under `training_job` folder to create an experiment and subsequent checkpoint.

## Step 2: Spin up a ray cluster
Note that you can really use any tech with this set up. This demo makes use of Ray.

Go to `launch_ray` folder. 

Then run the command in the `ray_launcher.ipynb`. There is also instructions on how to connect to the Ray dashboard in that notebook.

This is borrowed from Ilia's port proxy example, thanks Ilia!

## Step 3: Batch inference
Navigate to `batch_inference` folder. Then you can submit the inference job using `inference_job_submitter.ipynb`.

`mnist_1.py` contains the logic to read checkpoint from determined, get inference data, run batch inference, and output to s3

You will be able to see status of this submission from the Ray dashboard from step 2.

