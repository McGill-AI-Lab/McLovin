import torch
import numpy as np
from torch_kmeans import KMeans

model = KMeans(n_clusters=10)

print(model)
