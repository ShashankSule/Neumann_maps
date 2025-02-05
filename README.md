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

## Example usage 

```python
# Generate 100 points from each Gaussian cluster
cluster1 = np.random.multivariate_normal(np.array([0,0]), 1.5*np.eye(2), 1000)
cluster2 = np.random.multivariate_normal(np.array([4,0]), 0.1*np.eye(2), 100)
dataset = np.vstack((cluster1, cluster2))

# Select 20 random indices or landmarks 
marked_points = np.random.choice(1100, 20, replace=False)
boolean_vector = np.ones(1100, dtype=bool)
boolean_vector[marked_points] = False

# compute nmap
nmap = diffusion_map.NeumannMap(alpha=0.0,epsilon=1.0,num_evecs=15, n_neigh=20)
nmap.construct_generator(dataset.T, boolean_vector)
L_nmap = nmap.get_generator()
nmap, nevecs3, nevals3 = nmap._construct_diffusion_coords(L_nmap)
```

## Numerical experiments

The experiments benchmarking Nmaps vs Dmap for butane data can be found [here](https://github.com/ShashankSule/Neumann_maps/blob/pub/butane.ipynb). The UCI digits example, with NMI + ACC calculations [here](https://github.com/ShashankSule/Neumann_maps/blob/pub/UCI_benchmark.ipynb). We also performed an experiment with spectral clustering (not described in the paper) [here](https://github.com/ShashankSule/Neumann_maps/blob/pub/two_clusters.ipynb). 

