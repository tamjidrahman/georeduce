from geo.dists import SphericalDistribution, HammingDistribution
from matplotlib import pyplot as plt


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

dist = HammingDistribution(2, 3)
dmatrix = dist.distance_matrix