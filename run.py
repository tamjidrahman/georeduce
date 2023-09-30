from geo.dists import SphericalDistribution
from matplotlib import pyplot as plt


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

dist = SphericalDistribution(2, 10)
dist.plot()