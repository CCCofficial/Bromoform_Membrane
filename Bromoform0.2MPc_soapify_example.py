#------------------------
# Information
#------------------------
#> This example script contains the source code that does the following:
#> 1) Convert xyz coordinates to SOAP vectors (i.e. SOAPify step )
#> 2) Dimensionality reduction of SOAP vectors using PCA
#> 3) Perform clustering on principal components from PCA.

import pytraj as pt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pathlib
import pytraj_analysis
import MDAnalysis as mda
import MDAnalysis

import SOAPify.HDF5er as HDF5er
from MDAnalysis import Universe

import SOAPify
import SOAPify.HDF5er as HDF5er
from MDAnalysis import Universe
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import cdist
import hdbscan
import numpy
import h5py
from time import perf_counter

from sklearn.decomposition import PCA
import SOAPify
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import numpy
import nglview as nv

# Custom library contained in './applysoappackage' containing helper functions
from applysoappackage import soappca, soaphdbscan, soapplotclusters


# Starting Parameters

# In[2]:

conc = '0.2'
run = '014'
# Trajectory was prepared to contain only bromoform carbon atoms and stored in "./00_post_process_traj"
traj_dir = "./00_post_process_traj"

system = f'Bromoform{conc}MPc'
traj_base_name = f'{system}_copy_01_run_{run}_soap'

# Multiple cutoffs for SOAP analysis can be used
# SOAPrcut_list = [5,6,7,8,9]
SOAPrcut_list = [5.5]

PCAdim = 8


# Convert mda.Universe {xyz} coordinates to hdf5

# In[3]:


traj_name = f'{traj_base_name}.dcd'
top_name = f'{traj_base_name}.psf'


# In[4]:


dcdFile = f'{traj_dir}/{traj_name}'
#topologyFile = 'CBR_Only_1000FramesBromoform0.6MWater_copy_01_run_024.pdb'
topologyFile = f'{traj_dir}/{top_name}'


# In[6]:

bromo_u = mda.Universe(topologyFile, dcdFile)


# In[7]:

# Check if dimensions and atom types are properly stored in the universe object
print(bromo_u.dimensions)
print(bromo_u.atoms.types)


# In[8]:


n_atoms = len(bromo_u.atoms)


# In[9]:

# Ensure each atom is assigned as type 'C' so that SOAP library detects it as a carbon.
bromo_u.atoms.types = ['C'] * n_atoms


# In[10]:


xyz_hdf5_fname = f'{traj_base_name}.hdf5'
outFile = f'{traj_dir}/{xyz_hdf5_fname}'
# Name of group you want to save to
trajGroupName = traj_base_name
HDF5er.MDA2HDF5(bromo_u, outFile, trajGroupName,
                trajChunkSize=100, attrs=None, override=True)


# In[20]:


with h5py.File(outFile, "r") as f:
    print(f['/Trajectories'].keys())
    print(f[f'/Trajectories/{traj_base_name}/'].keys())
    types_dset = f[f'/Trajectories/{traj_base_name}/Types']
    types_arr = np.array(types_dset)

    box_dset = f[f'/Trajectories/{traj_base_name}/Box']
    box_arr = np.array(box_dset)
    trajectory_dset = f[f'/Trajectories/{traj_base_name}/Trajectory']
    trajectory_arr = np.array(trajectory_dset)


# # SOAPify Trajectory with dscribe

# ### Define Soap Parameters

# In[11]:


# name of hdf5 file containing
xyz_hdf5_fname = f'{traj_base_name}.hdf5'
trajFileName = f'{traj_dir}/{xyz_hdf5_fname}'


# Soap Parameters

SOAPnmax = 10
# SOAPnmax (int)
# The number of radial basis functions (option passed to the desired
# SOAP engine). Defaults to 8.


SOAPlmax = 10
# SOAPlmax (int)
# The maximum degree of spherical harmonics (option passed to the
# desired SOAP engine). Defaults to 8.


SOAPnJobs = 6
# SOAPnJobs (int, optional)
# the number of concurrent SOAP calculations (option passed to the
# desired SOAP engine). Defaults to 1.


sigma = 0.25
# Standard deviation used to expand the gaussians used to expand the atomic density.
SOAPkwargs = {'sigma': sigma}


SOAPOutputChunkDim = 200
# Size of trajectory chunk to load into memory. Load trajectory in chunks at a time if your computer doesn't have enough RAM.
# SOAPOutputChunkDim (int, optional):
# The dimension of the chunk of data in the SOAP results dataset.
# Defaults to 100.


# ## SOAPIFY


# In[20]:


# Define function to SOAPify trajectory
def worker(trajFileName: str, soapFileName: str, soapGroup, **kwargs) -> None:
    with h5py.File(trajFileName, "r") as workFile, h5py.File(
            soapFileName, "a"
    ) as soapFile:
        SOAPify.saponifyMultipleTrajectories(
            trajContainers=workFile["Trajectories"],
            SOAPoutContainers=soapFile.require_group(soapGroup),
            SOAPOutputChunkDim=1000,
            verbose=False,
            doOverride=True,
            **kwargs,
        )


# In[21]:

soap_dir = './01_soap'
for SOAPrcut in SOAPrcut_list:
    start_time = perf_counter()

    soapGroup = f'SOAP_n{SOAPnmax}_l{SOAPlmax}_rcut{SOAPrcut}_sigma{sigma}'
    print(f'SOAPifying xyz with the following parameters: {soapGroup}')
    soapFileName = f'{soap_dir}/{system}_{soapGroup}.hdf5'
    # Name of Soap file to save
    print(f'Saving SOAPified data to: {soapFileName}')

    worker(
        trajFileName=trajFileName,
        soapFileName=soapFileName,
        soapGroup=soapGroup,
        SOAPnJobs=SOAPnJobs,
        SOAPrcut=SOAPrcut,
        SOAPnmax=SOAPnmax,
        SOAPlmax=SOAPlmax,
        SOAPkwargs=SOAPkwargs,
    )

    end_time = perf_counter()
    elapsed_time = end_time - start_time
    print(f"SOAPify Elapsed Time: {elapsed_time:.6f} seconds")


# In[22]:

# Directory to save the SOAP vectors
soap_dir = './01_soap'

for SOAPrcut in SOAPrcut_list:

    soapGroup = f'SOAP_n{SOAPnmax}_l{SOAPlmax}_rcut{SOAPrcut}_sigma{sigma}'
    soapFileName = f'{soap_dir}/{system}_{soapGroup}.hdf5'

    with h5py.File(soapFileName, "r") as f:
        # print(f[f'{soapGroup}'].keys())
        print(f"Soap dataset saved in:{soapGroup}/{traj_base_name} ")
        soap_dset = f[f'{soapGroup}/{traj_base_name}']
        print(f"Dimmensions of SOAP dataset: {soap_dset.shape}")


# # PCA Analysis

# In[23]:


from applysoappackage.soappca import *


# ## Apply PCA

# In[29]:


# Subset of data to include to fit to PCA object.
fitSetSlice = slice(0, None, 1)

pca_dir = './02_pca'


# In[25]:

soap_dir = './01_soap'
for SOAPrcut in SOAPrcut_list:

    soapGroup = f'SOAP_n{SOAPnmax}_l{SOAPlmax}_rcut{SOAPrcut}_sigma{sigma}'
    soap_dset_path = f'{soapGroup}/{traj_base_name}'

    #print(f'SOAPifying xyz with the following parameters: {soapGroup}')
    soapFileName = f'{soap_dir}/{system}_{soapGroup}.hdf5'
    print(f'Openning SOAPified data is in : {soapFileName}')
    soap_hdf5_file = soapFileName
    print('Training PCA Model')
    # Train your PCA model
    with h5py.File(soap_hdf5_file, "r") as file:
        pcaMaker = preparePCAFitSet(file[soap_dset_path], PCAdim, dataSetSlice=fitSetSlice,
                                    )

    # Create group in PCAFile hdf5, represents system you trained the PCA model on
    pcaGroupName = f"{system}_{soapGroup}"

    # Output name of hdf5 file
    pcaFname = f"{pca_dir}/{pcaGroupName}_{PCAdim}pca.hdf5"
    print(f"Writing {PCAdim} PC vectors to {pcaFname}")
    applypcaNewFile(soapFileName, pcaFname, pcaMaker, pcaGroupName, soapGroup)


# ### Calculate variance explained by number of PCs

# In[26]:

#PCAdim = 8
pcs_explained_var = 8
# Generate individual
#SOAPrcut_list = [5,6,7,8,9]
plot_scatter = True

soap_dir = './01_soap'
for SOAPrcut in SOAPrcut_list:
    soapGroup = f'SOAP_n{SOAPnmax}_l{SOAPlmax}_rcut{SOAPrcut}_sigma{sigma}'
    soap_dset_path = f'{soapGroup}/{traj_base_name}'
    pcaGroupName = f"{system}_{soapGroup}"
    print(f"Working with {pcaGroupName}")

    # Create group in PCAFile hdf5, represents system you trained the PCA model on
    pcaname = f"{system}_{soapGroup}"
    pcaFname = f"{pca_dir}/{pcaGroupName}_{PCAdim}pca.hdf5"
    with h5py.File(pcaFname, "r") as workFile:
        # print(workFile[f'PCAs/{pcaGroupName}'].keys())
        pca_dset_path = f'PCAs/{pcaGroupName}/{traj_base_name}'
        dset = workFile[pca_dset_path]
        var_dset = dset.attrs['variance']
        explained_var = var_dset[:pcs_explained_var].sum()
        print(
            f"The explained variance for {pcs_explained_var} PCs is: {explained_var}")

        i = 0
        if plot_scatter == True:
            plt.figure()
            PC_first_second = dset[:, :, :2]
            PC_first_second_2d = PC_first_second.reshape(
                -1, PC_first_second.shape[-1])
            plt.scatter(PC_first_second_2d[:, 0], PC_first_second_2d[:, 1])
        i += 1


# # HDBScan: Train HDBScan Classifier

# In[32]:


from applysoappackage.soaphdbscan import *


# In[33]:


import importlib
import applysoappackage.soaphdbscan


# ## Perform Clustering

# In[35]:

pca_dir = './02_pca'
classification_dir = './03_classification'

for SOAPrcut in SOAPrcut_list:
    soapGroup = f'SOAP_n{SOAPnmax}_l{SOAPlmax}_rcut{SOAPrcut}_sigma{sigma}'
    print(f"Working with {soapGroup}")
    #soap_dset_path = f'{soapGroup}/{traj_base_name}'
    pcaGroupName = f"{system}_{soapGroup}"

    # Create group in PCAFile hdf5, represents system you trained the PCA model on
    pcaname = f"{system}_{soapGroup}"
    pcaFname = f"{pca_dir}/{pcaGroupName}_{PCAdim}pca.hdf5"
    pca_dset_path = f'PCAs/{pcaGroupName}/{trajGroupName}'

    # Fit the data to hdbscan
    hdnc = trainNoiseClassifier(
        soapFile=pcaFname,
        fitsetAddress=pca_dset_path,
        fitSetSlice=fitSetSlice,
        allow_single_cluster=True
    )

    # Group address/directory where PCA datasets live; for you have many systems it will classify each one that exists in the hdf5 file
    PCAGroupAddr = f'PCAs/{pcaGroupName}/'

    # Where to save classification hdf5
    classification_outfile = f"{classification_dir}/{pcaname}_{PCAdim}PCs_classifications.hdf5"

    # Directory in where to save in hdf5 file
    classification_group_dir = f"Classifications/{pcaname}"

    classifyNPs(
        hdnc,
        soapFile=pcaFname,
        PCAGroupAddr=PCAGroupAddr,
        outFile=classification_outfile,
        whereToSave=classification_group_dir,
    )


# # Plotting: Plot cluster members onto PC Space

# ## Define Data Loading Functions

# In[36]:


from applysoappackage.soapplotclusters import *


# ## Plot cluster members onto PCA space

# In[37]:


# Define plotting function Parameters
pFESsmoothing = 0.5
pFEScmap = plt.cm.coolwarm_r.copy()
pFEScmap.set_over("w")
labelsOptions = dict(fontsize=15)
imgzoom = 0.015

# Define colors for each cluster label
from matplotlib.colors import to_rgb
bottomUpLabels = [
    "Faces",  # 0
    "Concave",  # 1
    "5foldedSS",  # 2
    "Ico",  # 3
    "Bulk",  # 4
    "SubSurf",  # 5
    "Edges",  # 6
    "Vertexes",  # 7
    "test 1"
]

# In[38]:


def get_color_map(cmap, n_unique_clusters):
    """
    Get color map based on number of clusters identified by hdbscan
    Args:
        cmap: matplotlib cmap object
        n_unique_clusters: int

    Returns:
    np.array of where rows are number of clusters, columns are rgb
    """
    cluster_arr = np.arange(0, n_unique_clusters)
    cluster_arr_divided = cluster_arr / n_unique_clusters
    rgb_arr = np.empty([0, 3])
    for value in cluster_arr_divided:
        rgb_value = cmap(value)[0:3]
        print(value)
        rgb_arr = np.vstack([rgb_arr, rgb_value])
    return rgb_arr


# In[39]:


def plot_soap_populations(soap_classification_2d_arr, colormap, ax=None):
    """
    soap_classification_2d_arr: the 2d array representing the cluster label for every atom of every frame.
    colormap: array of rgb values or colors for each cluster label
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(12, 5))
    plt.ylim([0, 1])
    flattened_cluster_labels = soap_classification_2d_arr.flatten()
    #bar_color_list = ['r','b','g','yellow']
    sns.histplot(data=flattened_cluster_labels, stat='probability', ax=ax)
    plt.xticks(np.arange(min(flattened_cluster_labels),
                         max(flattened_cluster_labels) + 1, 1))

    # Color bar by cluster group
    i = 0
    for bar in ax.patches:
        rgb_color = colormap[i]
        if bar.get_height() > 0:
            bar.set_color(rgb_color)
            i += 1


# In[40]:


from applysoappackage.soapplotclusters import *


# In[47]:

trajectorySlice = slice(0, None, 1)
classification_dir = './03_classification'
plot_fes = True
plot_pop = True
cmap_name = 'inferno'
xlims = [-0.3, 0.3]
ylims = [-0.2, 0.3]

for SOAPrcut in SOAPrcut_list:
    soapGroup = f'SOAP_n{SOAPnmax}_l{SOAPlmax}_rcut{SOAPrcut}_sigma{sigma}'

    #soap_dset_path = f'{soapGroup}/{traj_base_name}'
    pcaGroupName = f"{system}_{soapGroup}"

    # Create group in PCAFile hdf5, represents system you trained the PCA model on
    pcaname = f"{system}_{soapGroup}"
    pcaFname = f"{pca_dir}/{pcaGroupName}_{PCAdim}pca.hdf5"
    pca_dset_path = f'PCAs/{pcaGroupName}/{traj_base_name}'
    # Group address/directory where PCA datasets live; for you have many systems it will classify each one that exists in the hdf5 file
    PCAGroupAddr = f'PCAs/{pcaGroupName}/'

    # Search for this system name string that's within classification hdf5 file.
    system_name_prefix = f"{system}"

    # Directory in where to save in hdf5 file
    classification_group_dir = f"Classifications/{pcaname}"

    # Where to save classification hdf5
    classification_outfile = f"{classification_dir}/{pcaname}_{PCAdim}PCs_classifications.hdf5"
    print(f"Working with {classification_outfile}")

    # Load classification into a dict called "data"
    data = pcaLoaderBottomUp(pcaFname, PCAGroupAddr, trajectorySlice)
    # Modify data dict to contain cluster ID information
    loadClassificationBottomUp(
        classification_outfile, data, system_name_prefix, bottomUpLabels, trajectorySlice, classification_group_dir,
    )

    # Generate a coloring map
    soap_classification = data[303]['ClassBU']
    n_unique_clusters = len(np.unique(soap_classification.references))
    print(f'The number of unique clusters:{n_unique_clusters}')
    cmap = matplotlib.cm.get_cmap(cmap_name)
    bottomUpColorMap = get_color_map(cmap, n_unique_clusters)

    if xlims and ylims:
        data['xlims'] = xlims
        data['ylims'] = ylims

    # add free energy information using boltzmann inversion
    for T in [303]:
        addPseudoFes(data[T], 150, rangeHisto=[data["xlims"], data["ylims"]])

    # Sigma = 0.25, rcut = 6
    fig, ax = plt.subplots(figsize=(12, 5))
    plotPCAData(
        ax,
        303,
        data[303],
        data["xlims"],
        data["ylims"],
        bottomUpColorMap,
        zoom=imgzoom,
        smooth=pFESsmoothing
    )
    if plot_pop:
        plot_soap_populations(soap_classification.references, bottomUpColorMap)
    if plot_fes:
        fig, ax = plt.subplots(figsize=(12, 5))
        plotTemperatureData(
            ax,
            303,
            data[303],
            data["xlims"],
            data["ylims"],
            bottomUpColorMap,
            zoom=imgzoom,
            smooth=pFESsmoothing
        )
