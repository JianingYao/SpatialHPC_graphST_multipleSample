import os
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import anndata as ad
import harmonypy as hm
from sklearn.decomposition import PCA
import seaborn as sns
import colorcet as cc
import math

from GraphST import GraphST
from helper_HPC import clust_HPC, run_harmony_recursive

# Run device, by default, the package is implemented on 'cpu', but recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm.
os.environ['R_HOME'] = '/users/jyao/.conda/envs/GraphST_hpc/lib/R'
# os.environ['R_HOME'] = '/jhpce/shared/jhpce/core/R/4.0.3/lib64/R/'

######### load samples and paths
full_list = pd.read_csv('../HPC_GraphST_full.csv')
# if using selected platforms
cohort = 'HE' # 'HE'; 'VSPG'
full_list = full_list[full_list['Platform'] == cohort]

full_list = full_list.reset_index(drop=True)
column_names = full_list.columns.tolist()
donor_list = full_list['Donor']
captureArea_list = full_list['CaptureArea']
out_paths = full_list['Spaceranger output']
platforms = full_list['Platform']

clusters = list(range(16, 19))

######### Load each sample and create annData
adatas = []
for i in range(len(donor_list)):
    print(f'Donor: {donor_list[i]}') # for checking
    file_fold = out_paths[i] + '/outs'
    # change file name to align with sc.read_visium
    old_name = os.path.join(file_fold, 'spatial', 'tissue_positions.csv')
    new_name = os.path.join(file_fold, 'spatial', 'tissue_positions_list.csv')
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        os.rename(new_name, old_name)
    else: 
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    print(f'  Capture Area: {captureArea_list[i]}') # for checking
    adata.var_names_make_unique()
    adata.obs['donor'] = donor_list[i]
    adata.obs['captureArea'] = captureArea_list[i]
    adata.obs['platform'] = platforms[i]
    adata
    adatas.append(adata.copy())

######### Concatenate all samples
adata_all = sc.AnnData.concatenate(*adatas)
adata_all.obs['donor'] = adata_all.obs['donor'].astype('category')
adata_all.obs['captureArea'] = adata_all.obs['captureArea'].astype('category')
adata_all.obs['platform'] = adata_all.obs['platform'].astype('category')
adata_all.obs['barcode'] = adata_all.obs.index.str.split('-').str[:-1].str.join('-')
adata_all.obsm['spatial'] = adata_all.obsm['spatial'].astype(int)
adata_all.obs['in_tissue'] = adata_all.obs['in_tissue'].astype(str).astype(int)
adata_all.obs['array_row'] = adata_all.obs['array_row'].astype(str).astype(int)
adata_all.obs['array_col'] = adata_all.obs['array_col'].astype(str).astype(int)
adata_all.var_names_make_unique()
adata_all

######### Subset spots after QC
# read in spot index after filtering and QC
qc_spots = pd.read_csv('../index.csv')
qc_spots['spot_area'] = qc_spots['barcode'] + '-' + qc_spots['sampleid']
barcodes = adata_all.obs['barcode']
sampleids = adata_all.obs['captureArea'].astype(str)
all_spots = pd.DataFrame({
    'barcode': barcodes,
    'sampleid': sampleids
})
all_spots['spot_area'] = all_spots['barcode'] + '-' + all_spots['sampleid']
# create a boolean mask for rows where both 'barcode' and 'captureArea' match with index file
mask = (
    all_spots['spot_area'].isin(qc_spots['spot_area'])
)
adata_all = adata_all[mask]

######### Preprocessing
# https://stlearn.readthedocs.io/en/latest/tutorials/Integration_multiple_datasets.html 
# filter genes
sc.pp.filter_genes(adata_all, min_cells=3)
# normalize data (default)
sc.pp.normalize_total(adata_all, target_sum=1e4)
# log transformation
sc.pp.log1p(adata_all)
# store raw data
adata_all.raw = adata_all
# extract top highly variable genes
sc.pp.highly_variable_genes(adata_all, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_all = adata_all[:, adata_all.var.highly_variable]
# scale data
sc.pp.scale(adata_all, max_value=10)

######### Perform GraphST on individual slices to get reconstructed gene expression
adatas2 = []
emb_all = np.zeros((adata_all.shape[0], adata_all.shape[1]), dtype=np.float32)
for i in adata_all.obs['batch'].unique():
    adata = adata_all[adata_all.obs['batch']==i]
    model = GraphST.GraphST(adata, device=device)
    adata = model.train()
    adata
    batch_i = adata_all.obs['batch'] == i
    emb_all[batch_i] = adata.obsm['emb'].copy()
    adatas2.append(adata.copy())
adata_all.obsm['emb'] = emb_all

adata_all_path = '/fastscratch/myscratch/jyao/hpc/adata_' + cohort + '_default.h5ad'
adata_all.write(adata_all_path)
adata_all = sc.read(adata_all_path)

######### Run dimensionality reduction
pc = 30
pca = PCA(n_components=pc, random_state=619) 
embedding = pca.fit_transform(adata_all.obsm['emb'].copy())
adata_all.obsm['rec_pca'] = embedding

######### Visualizing from variability from different sources before BEC
result_path = '../HPC_result'
directory_path = os.path.join(result_path, f"{cohort}_default_pc30")
if not os.path.exists(directory_path):
    # Create the directory if it doesn't exist
    os.makedirs(directory_path)
umap_path = directory_path + '/umap_' + f"{cohort}_before-BEC.png"
sc.pp.neighbors(adata_all, use_rep='rec_pca')
sc.tl.umap(adata_all)
sc.pl.umap(adata_all, color=['captureArea','donor', 'platform'])
plt.savefig(umap_path)
plt.close()
# Br3942 & Br8325
selected_donors = ["Br3942", "Br8325"]
adata_2donors = adata_all[adata_all.obs['donor'].isin(selected_donors)].copy()
umap_path = directory_path + '/umap_' + f"{cohort}_2donors_before-BEC.png"
sc.pp.neighbors(adata_2donors, use_rep='rec_pca')
sc.tl.umap(adata_2donors)
sc.pl.umap(adata_2donors, color=['captureArea','donor', 'platform'])
plt.savefig(umap_path)
plt.close()

######### Run integration with harmony
# prepare metadata and PCA
meta_data = adata_all.obs
data_mat = adata_all.obsm['rec_pca']
# Define the order of factors
if cohort == 'HE' or cohort == 'VSPG':
    factors = ['donor', 'captureArea'] 
else:
    factors = ['platform', 'donor', 'captureArea']

# harmony alternative 0. Default harmony
ho = hm.run_harmony(data_mat, meta_data, factors, max_iter_harmony=50)
# mapping back the result to the adata object
adata_all.obsm['rec_pca_bec'] = ho.Z_corr.T

######### Visualizing from variability from different sources after BEC
umap_path = directory_path + '/umap_' + f"{cohort}_after-BEC.png"
sc.pp.neighbors(adata_all, use_rep='rec_pca_bec')
sc.tl.umap(adata_all)
sc.pl.umap(adata_all, color=['captureArea','donor', 'platform'])
plt.savefig(umap_path)
plt.close()
# Br3942 & Br8325
selected_donors = ["Br3942", "Br8325"]
adata_2donors = adata_all[adata_all.obs['donor'].isin(selected_donors)].copy()
umap_path = directory_path + '/umap_' + f"{cohort}_2donors_after-BEC.png"
sc.pp.neighbors(adata_2donors, use_rep='rec_pca_bec')
sc.tl.umap(adata_2donors)
sc.pl.umap(adata_2donors, color=['captureArea','donor', 'platform'])
plt.savefig(umap_path)
plt.close()

######### Perform clustering
for n_clusters in clusters: 
    radius = 6 # refinement
    key = 'rec_pca_bec'
    tool = 'mclust'  # mclust, leiden, and louvain
    if tool == 'mclust':
        clust_HPC(adata_all, n_clusters, radius=radius, key=key, method=tool, refinement=True) 
    elif tool in ['leiden', 'louvain']:
        clust_HPC(adata_all, n_clusters, radius=radius, key=key, method=tool, start=0.1, end=2.0, increment=0.01, refinement=True)
    cluster_col = f'k{n_clusters}'
    ref_cluster_col = f'k{n_clusters}-ref'
    adata_all.obs[cluster_col] = adata_all.obs['domain']
    adata_all.obs[ref_cluster_col] = adata_all.obs['domain_ref']

adata_all.obs.drop(columns=['domain'], inplace=True)
adata_all.obs.drop(columns=['domain_ref'], inplace=True)
adata_all.obs.drop(columns=['mclust'], inplace=True)

adata_all_path = '/fastscratch/myscratch/jyao/hpc/adata_' + cohort + f"_default_clusters.h5ad"
adata_all.write(adata_all_path)
adata_all = sc.read(adata_all_path)

######### Plot multiple slides within a donor
adata_all.obsm['spatial'][:,1] = -1*adata_all.obsm['spatial'][:,1]
for n_clusters in clusters: 
    cluster_col = f'k{n_clusters}'
    # rgb_values = sns.color_palette("tab20", len(adata_all.obs['cluster_col'].unique()))
    # color_fine = dict(zip(list(adata_all.obs['cluster_col'].unique()), rgb_values))
    palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)
    color_fine = dict(zip(list(range(1, n_clusters+1)), palette))
    pdf_path = directory_path + '/' + f"{cohort}_k{n_clusters}" + '.pdf'
    pdf = PdfPages(pdf_path)
    # plot a donor at a time
    for donor in list(set(donor_list)):
        adata_donor = adata_all[adata_all.obs['donor'] == donor]
        unique_capture_areas = adata_donor.obs['captureArea'].unique()
        num_subplots = len(unique_capture_areas)
        num_rows = math.ceil(math.sqrt(num_subplots))
        # Calculate the number of rows needed for the subplot grid
        num_cols = math.ceil(num_subplots / num_rows)
        # Create a subplot grid for the current donor
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*5, num_rows*5))
        # Flatten the axes array to make it easier to index
        axes = axes.flatten()
        # Iterate over each captureArea and plot in the corresponding subplot
        for i, captureArea in enumerate(unique_capture_areas):
            adata = adata_donor[adata_donor.obs['captureArea'] == captureArea]
            # Plot raw predictions in the current subplot with larger dots (size=20)
            sc.pl.embedding(adata,
                            basis="spatial",
                            color=cluster_col,
                            palette=color_fine,
                            size=50,  
                            show=False,
                            title=captureArea,
                            ax=axes[i])
            # Hide axis for empty subplots
            if i >= num_subplots:
                axes[i].axis('off')
        fig.suptitle(f'Donor: {donor}', fontsize=16)
        # Adjust layout for better visualization
        plt.tight_layout()
        # Save the current page with donor plots to the PDF file
        pdf.savefig(fig)
        plt.close()
    # Close the PDF file
    pdf.close()


# convert to cvs file
spot_id = adata_all.obs_names
barcode = adata_all.obs['barcode']
captureArea = adata_all.obs['captureArea']
donor_id = adata_all.obs['donor']
cluster_k16 = adata_all.obs['k16']
cluster_k17 = adata_all.obs['k17']
cluster_k18 = adata_all.obs['k18']


# domain_ref = adata_all.obs['domain_ref']
result_combined = pd.DataFrame({
    'spot_id': spot_id,
    'barcode': barcode,
    'captureArea': captureArea,
    'donor_id': donor_id,
    'cluster_k16': cluster_k16,
    'cluster_k17': cluster_k17,
    'cluster_k18': cluster_k18
})

# Save the DataFrame to a CSV file
result_path = '../HPC_result/HE_default_pc30/hpc_allSamples_graphst_clusters.csv'
result_combined.to_csv(result_path, index=False)

