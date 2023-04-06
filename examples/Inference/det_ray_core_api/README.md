# Determined Ray Inference Demo
## tl;dr
This demo makes use of Liam's Ray launcher and does not rely on port proxy feature. A ray inference job launched this 
way is shown as individual experiment within determined UI.

We also added basic determined core API integration, which allows us to store data such as metrics to a determined 
checkpoint.

## How to run
You can simply run `det e create det-expconfig.yaml .` in this directory to submit the experiment to a determined 
cluster.

If you prefer to run this in a determined notbook, you can run the command in the control_panel.ipynb file.

## Explanation
- This experiment would first run `ray_launcher_2.py` to set up a ray cluster (det chief worker becomes the head ray node
and others would be worker ray nodes). 
- Then it would run `inference.py` on the det chief worker. This script initialize ray and attach to the cluster created
  by running `ray_launcher_2.py`
- Ray manages allocation of work to other ray workers (e.g. when we call `ray_data.map_batches` or 
  `batch_predictor.predict` in `inference.py`)
- As `inference.py` is wrapped in a det core context, we can also report checkpoint, metrics, etc. 


## Note
Also included a requirements.txt file from when I run this created from pip freeze. If you try to run this and it errors
out, you can use this file as a reference on the dependency snapshot.

It is not actually used in setting up the experiment, just as a reference.

