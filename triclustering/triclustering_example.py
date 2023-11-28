import cgc
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cgc.triclustering import Triclustering
from cgc.kmeans import KMeans
from cgc.utils import calculate_tricluster_averages


# ----------------------------------------------- {1st step} -----------------------------------------------
print("1. reading the 3d dataset...")
spring_indices = xr.open_zarr("https://raw.githubusercontent.com/esciencecenter-digital-skills/tutorial-cgc/main/data/spring-indices.zarr", chunks=None)
print(spring_indices)


# ----------------------------------------------- {2nd step} -----------------------------------------------
print("2. creating the 4d dataset...")
spring_indices = spring_indices.to_array(dim='spring_index')
print(spring_indices)


# ----------------------------------------------- {3rd step} -----------------------------------------------
print("3. showing the 4d dataset...")
spring_indices.sel(time=slice(1990, 1992)).plot.imshow(row='spring_index', col='time')
plt.show()


# ----------------------------------------------- {4th step} -----------------------------------------------
print("4. creating a 3d array (row-time x column-space x band-spring_index)...")
spring_indices = spring_indices.stack(space=['x', 'y'])
location = np.arange(spring_indices.space.size) # create a combined (x,y) index
spring_indices = spring_indices.assign_coords(location=('space', location))
# drop pixels that are null-valued for any year/spring-index
spring_indices = spring_indices.dropna('space', how='any')
print(spring_indices)


# ----------------------------------------------- {5th step} -----------------------------------------------
print("5. creating the model...")
num_band_clusters = 3
num_time_clusters = 5
num_space_clusters = 20
max_iterations = 50  # maximum number of iterations
conv_threshold = 0.1  # convergence threshold
nruns = 3  # number of differently-initialized runs
tc = Triclustering(
    spring_indices.data,  # data array with shape: (bands, rows, columns)
    num_time_clusters,
    num_space_clusters,
    num_band_clusters,
    max_iterations=max_iterations,
    conv_threshold=conv_threshold,
    nruns=nruns
)
results = tc.run_with_threads(nthreads=4)


# ----------------------------------------------- {6th step} -----------------------------------------------
print("6. inspecting the results...")
print(f"Row (time) clusters: {results.row_clusters}")
print(f"Column (space) clusters: {results.col_clusters}")
print(f"Band clusters: {results.bnd_clusters}")
time_clusters = xr.DataArray(results.row_clusters, dims='time',
                             coords=spring_indices.time.coords,
                             name='time cluster')
space_clusters = xr.DataArray(results.col_clusters, dims='space',
                              coords=spring_indices.space.coords,
                              name='space cluster')
band_clusters = xr.DataArray(results.bnd_clusters, dims='spring_index',
                             coords=spring_indices.spring_index.coords,
                             name='band cluster')


# ----------------------------------------------- {7th step} -----------------------------------------------
print("7. calculating the averages...")
# calculate the tri-cluster averages
means = calculate_tricluster_averages(
    spring_indices.data,
    time_clusters,
    space_clusters,
    band_clusters,
    num_time_clusters,
    num_space_clusters,
    num_band_clusters
)
means = xr.DataArray(
    means,
    coords=(
        ('band_clusters', range(num_band_clusters)),
        ('time_clusters', range(num_time_clusters)),
        ('space_clusters', range(num_space_clusters))
    )
)
space_means = means.sel(space_clusters=space_clusters, drop=True)
space_means = space_means.unstack('space')
space_means.plot.imshow(
    x='x', y='y',
    row='band_clusters',
    col='time_clusters',
    vmin=50, vmax=120
)
plt.show()