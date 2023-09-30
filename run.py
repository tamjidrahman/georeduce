from geo.dists import SphericalDistribution, HammingDistribution
from matplotlib import pyplot as plt
from encoder.encoder import IdentityEncoder


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

dist = SphericalDistribution(3, 1000)
encoder = IdentityEncoder(dist)

plt.scatter(encoder.input_distribution.distribution, encoder.transformed_distribution.distribution)