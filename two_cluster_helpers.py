import numpy as np 
import sklearn
import matplotlib.pyplot as plt
import os
import copy
import sys 
sys.path.append("..")
import scipy
import src.model_systems as model_systems
import src.helpers as helpers
import src.potentials as potentials
import src.diffusion_map as diffusion_map
from tqdm import tqdm
from scipy.stats import entropy

def gen_data():
    # Parameters for the Gaussian clusters
    mean1 = np.random.rand(2) * 10  # Random mean for the first cluster
    cov1 = 0.4*np.eye(2)  # Identity matrix as covariance for the first cluster

    mean2 = np.random.rand(2) * 10  # Random mean for the second cluster
    cov2 = 0.1*np.eye(2)  # Identity matrix as covariance for the second cluster

    mean1 = np.array([0,0])
    mean2 = np.array([4,0])
    cov1 = 1.5*np.eye(2)
    cov2 = 0.1*np.eye(2)

    # Generate 100 points from each Gaussian cluster
    cluster1 = np.random.multivariate_normal(mean1, cov1, 1000)
    cluster2 = np.random.multivariate_normal(mean2, cov2, 100)

    # Combine the clusters to form the dataset
    dataset = np.vstack((cluster1, cluster2))

    # Optionally, create labels for the clusters
    labels = np.array([0]*1000 + [1]*100)
    # Select 20 random indices from the dataset
    marked_points = np.random.choice(1100, 20, replace=False)
    # Create a boolean vector of size 200 with all True values
    boolean_vector = np.ones(1100, dtype=bool)
    # Set the selected indices to False
    boolean_vector[marked_points] = False

    # Fiedler vector: 
    dmap = diffusion_map.DiffusionMap(alpha=0.0,epsilon=1.0,num_evecs=15, n_neigh=20)
    dmap.construct_generator(dataset[boolean_vector].T)
    L_dmap = dmap.get_generator()
    dmap, devecs3, devals3 = dmap._construct_diffusion_coords(L_dmap)
    spectral_labels = dmap[:,0] > np.mean(dmap[:,0])

    # Fiedler vector: 
    nmap = diffusion_map.NeumannMap(alpha=0.0,epsilon=1.0,num_evecs=15, delta=0.8, n_neigh=20)
    nmap.construct_generator(dataset.T, boolean_vector)
    L_nmap = nmap.get_generator()
    nmap, nevecs3, nevals3 = nmap._construct_diffusion_coords(L_nmap)
    # save experiment 
    np.savez('clusters_experiment.npz', dataset=dataset, labels=labels, boolean_vector=boolean_vector, \
            dmap=dmap, devecs3=devecs3, devals3=devals3, nmap=nmap, nevecs3=nevecs3, nevals3=nevals3, \
                alpha=0.0, epsilon=1.0, num_evecs=10)

def random_labels_experiment(dataset, n_marks):
    # Select 20 random indices from the dataset
    marked_points = np.random.choice(1100, n_marks, replace=False)
    boolean_vector = np.ones(1100, dtype=bool)
    # Set the selected indices to False
    boolean_vector[marked_points] = False
    # Fiedler vector:
    dmap = diffusion_map.DiffusionMap(alpha=0.0,epsilon=1.0,num_evecs=1, n_neigh=20)
    dmap.construct_generator(dataset[boolean_vector].T)
    L_dmap = dmap.get_generator()
    dmap, _, _ = dmap._construct_diffusion_coords(L_dmap)
    # Fiedler vector: 
    nmap = diffusion_map.NeumannMap(alpha=0.0,epsilon=1.0,num_evecs=1, delta=0.5, n_neigh=20)
    nmap.construct_generator(dataset.T, boolean_vector)
    L_nmap = nmap.get_generator()
    nmap, _, _ = nmap._construct_diffusion_coords(L_nmap)
    return dmap, nmap, boolean_vector

def get_embeddings(n_marks):
    embeddings = []
    for _ in tqdm(range(100)): 
        dmap, nmap, boolean_vector = random_labels_experiment(dataset, n_marks)
        embeddings.append((dmap, nmap, boolean_vector))
    return embeddings

def compute_cluster_separator(embeddings, labels):
    dmap_separators = []
    nmap_separators = []
    for dmap, nmap, boolean_vector in embeddings:
        data_labels = labels[boolean_vector]
        dmap_separators.append(np.abs(np.mean(dmap[data_labels==0]) - np.mean(dmap[data_labels==1])))
        nmap_separators.append(np.abs(np.mean(nmap[data_labels==0]) - np.mean(nmap[data_labels==1])))
    return dmap_separators, nmap_separators

def compute_cluster_concentration(embeddings, labels, cluster_label, setting="entropy"):
    dmap_concentration = []
    nmap_concentration = []
    for dmap, nmap, boolean_vector in embeddings:
        data_labels = labels[boolean_vector]
        if setting == "entropy":
            dmap_distribution, _ = np.histogram(dmap[data_labels==cluster_label], density=True)
            nmap_distribution, _ = np.histogram(nmap[data_labels==cluster_label], density=True)
            dmap_concentration.append(entropy(dmap_distribution))
            nmap_concentration.append(entropy(nmap_distribution))
        else: 
            dmap_concentration.append(np.std(dmap[data_labels==cluster_label]))
            nmap_concentration.append(np.std(nmap[data_labels==cluster_label]))
    return dmap_concentration, nmap_concentration

def gen_plot(data_list, n_marks):
    # Plot histograms
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    nmap_cluster_separation, dmap_cluster_separation, nmap_cluster_concentration_0, dmap_cluster_concentration_0, nmap_cluster_concentration_1, dmap_cluster_concentration_1 = data_list
    axs[0].hist(nmap_cluster_separation, bins=10, alpha=0.5, label='Nmap', density=True)
    axs[0].hist(dmap_cluster_separation, bins=10, alpha=0.5, label='Dmap', density=True)

    # Add labels and legend
    axs[0].set_xlabel('Distance between cluster means', fontsize=16)
    axs[0].set_ylabel('Density', fontsize=16)
    axs[0].set_title(f"Cluster mean separation, marking rate = {n_marks/11:.2f}%", fontsize=16)
    axs[0].legend()

    axs[1].hist(nmap_cluster_concentration_0, bins=10, alpha=0.5, label='Nmap', density=True)
    axs[1].hist(dmap_cluster_concentration_1, bins=10, alpha=0.5, label='Dmap', density=True)

    # Add labels and legend
    axs[1].set_xlabel('Cluster entropy', fontsize=16)
    axs[1].set_ylabel('Density', fontsize=16)
    axs[1].set_title("Big cluster", fontsize=16)
    axs[1].legend()

    axs[2].hist(nmap_cluster_concentration_0, bins=10, alpha=0.5, label='Nmap', density=True)
    axs[2].hist(dmap_cluster_concentration_1, bins=10, alpha=0.5, label='Dmap', density=True)

    # Add labels and legend
    axs[2].set_xlabel('Cluster entropy', fontsize=16)
    axs[2].set_ylabel('Density', fontsize=16)
    axs[2].set_title("Big cluster", fontsize=16)
    axs[2].legend()

    plt.subplots_adjust(hspace=0.25)  # Add padding between subplots
    plt.savefig(f"random_labels_{n_marks}.png")