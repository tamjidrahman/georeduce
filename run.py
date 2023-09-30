from geo.dists import get_spherical_uniform_distribution
from vis.scatter import plot_array

dist = get_spherical_uniform_distribution(dim=2, num_samples=10)
plot_array(dist)