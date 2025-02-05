# Neumann Maps

This repository contains implementations and examples of Neumann Maps, a landmarking method for accelerating and improving diffusion maps. Our implementation of Nmap subclasses Banisch, Theide, and Trstanova's `PyDiffmap` [library](https://pydiffmap.readthedocs.io/en/master/readme.html)

## Python Environment

To set up the Python environment, you need the following packages:

- numpy
- scipy
- matplotlib
- scikit-learn
- jupyter

You can install these packages using pip:

```bash
pip install numpy scipy matplotlib scikit-learn jupyter
```

## Numerical experiments

The experiments benchmarking Nmaps vs Dmap for butane data can be found [here](https://github.com/ShashankSule/Neumann_maps/blob/pub/butane.ipynb). The UCI digits example, with NMI + ACC calculations [here](https://github.com/ShashankSule/Neumann_maps/blob/pub/UCI_benchmark.ipynb). We also performed an experiment with spectral clustering (not described in the paper) [here](https://github.com/ShashankSule/Neumann_maps/blob/pub/two_clusters.ipynb). 

