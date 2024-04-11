import pandas as pd
import numpy as np
import scanpy as sc
from GraphST.utils import *


def prepGene_HPC(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def clust_HPC(adata, n_clusters=7, radius=50, key='emb_pca', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:
        if adata.obs['batch'] is None:
            new_type = refine_label(adata, radius, key='domain')
            adata.obs['domain_ref'] = new_type 
        else:
            new_type = np.zeros(adata.shape[0], dtype=np.float32)
            for i in adata.obs['batch'].unique():
                adata_i = adata[adata.obs['batch']==i]
                batch_i = adata.obs['batch'] == i
                new_type[batch_i] = refine_label(adata_i, radius, key='domain')
            adata.obs['domain_ref'] = new_type
            adata.obs['domain_ref'] = adata.obs['domain_ref'].astype(int).astype('category')
 

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type


# def find_resolution():


import harmonypy as hm
def run_harmony_recursive(data, obs, factors, lambdas):
    """
    Run harmony correction recursively on factors.
    """
    # Base condition for recursion: If no factors left to correct for
    if not factors:
        print("no factor left, data shape ", data.shape)
        return data
    
    factor = factors[0]
    lamb = lambdas[0]
    
    # Apply Harmony correction for the current factor across the entire dataset
    corrected_data = hm.run_harmony(data, obs, factor, max_iter_harmony=50, theta=2, lamb=lamb).Z_corr.T
    
    # If this is the last factor in the list, return the corrected data
    if len(factors) == 1:
        print("corrected data shape ", corrected_data.shape)
        return corrected_data

    # Placeholder for further corrected data
    further_corrected_data = np.zeros_like(corrected_data)
    
    # Get unique values for the current factor
    unique_values = obs[factor].unique()

    # Loop through each unique value of the factor
    for value in unique_values:
        subset_meta = obs[obs[factor] == value]
        subset_meta = subset_meta.copy()
        subset_meta[factors[1]] = subset_meta[factors[1]].astype(str).astype('category')
        
        indices = np.where(obs[factor] == value)[0]
        subset_data = corrected_data[indices]

        # Recursively correct the data for the remaining factors
        further_corrected_data[indices] = run_harmony_recursive(subset_data, subset_meta, factors[1:], lambdas[1:])
    print("further corrected data shape ", further_corrected_data.shape)
    return further_corrected_data





