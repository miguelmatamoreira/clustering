from matplotlib import pyplot as plt
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score
import numpy as np


# ----------------------------------------------- {1st step} -----------------------------------------------
# generate the data
print("1. generating the data...")
n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=42
)
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("original dataset")
plt.show()


# ----------------------------------------------- {2nd step} -----------------------------------------------
# shuffling the data
print("2. shuffling the data...")
rng = np.random.RandomState(0)
row_idx_shuffled = rng.permutation(data.shape[0]) # list of 300 shuffle elements
col_idx_shuffled = rng.permutation(data.shape[1]) # list of 300 shuffle elements


# ----------------------------------------------- {3rd step} -----------------------------------------------
# apply the shuffle data
print("3. applying the shuffle data...")
data = data[row_idx_shuffled][:, col_idx_shuffled]
plt.matshow(data, cmap=plt.cm.Blues)
plt.title("shuffled dataset")
plt.show()


# ----------------------------------------------- {4th step} -----------------------------------------------
# model
print("4. making the model...")
model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
model.fit(data)


# ----------------------------------------------- {5th step} -----------------------------------------------
# compute the similarity of two sets of biclusters
print("5. computing the similarity of two sets of biclusters...")
score = consensus_score(model.biclusters_, (rows[:, row_idx_shuffled], columns[:, col_idx_shuffled]))
print(f"consensus score: {score:.1f}")


# ----------------------------------------------- {6th step} -----------------------------------------------
# results
print("6. showing the results...")
reordered_data = data[np.argsort(model.row_labels_)]
reordered_data = reordered_data[:, np.argsort(model.column_labels_)]
plt.matshow(reordered_data, cmap=plt.cm.Blues)
plt.title("after biclustering, rearranged to show biclusters")
plt.show()


# ----------------------------------------------- {7th step} -----------------------------------------------
# sort for color
print("7. sorting for color...")
plt.matshow(np.outer(np.sort(model.row_labels_) + 1, np.sort(model.column_labels_) + 1), cmap=plt.cm.Blues)
plt.title("checkerboard structure of rearranged data")
plt.show()

