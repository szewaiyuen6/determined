# Specifying gpustat as the next minor version run into issue
# We should think about environment management for this
pip install gpustat==1.0.0
pip install ray
pip install -U "ray[air]"
