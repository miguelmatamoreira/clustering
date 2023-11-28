import numpy as np
from cgc.triclustering import Triclustering

Z = np.array([[[1., 1., 2., 4.],
               [1., 1., 2., 4.]],
              [[5., 5., 8., 8.],
               [5., 5., 8., 8.]],
              [[6., 7., 8., 9.],
               [6., 7., 9., 8.]]])

print(Z)

tc = Triclustering(
    Z,  # data array (3D)
    nclusters_row=4,  # number of row clusters
    nclusters_col=3,  # number of column clusters
    nclusters_bnd=2,  # number of band clusters
    max_iterations=100,  # maximum number of iterations
    conv_threshold=1.e-5,  # error convergence threshold
    nruns=10,  # number of differently-initialized runs
    output_filename='results.json'  # JSON file where to write output
)

results = tc.run_with_threads(nthreads=4)