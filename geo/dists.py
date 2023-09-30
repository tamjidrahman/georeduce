from functools import cached_property
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

class SphericalDistribution:
    """A distribution of D dimensional vectors embedded in S^(D-1) unit sphere
    """

    def __init__(self, dim: int, num_samples: int):

        self.dim = dim
        self.num_samples = num_samples

        self.distribution = get_spherical_uniform_distribution(dim, num_samples)

    def plot(self):
        if self.dim > 3:
            raise Exception(NotImplemented)
        
        fig = plt.figure()
        if self.dim == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()

        dims = (self.distribution[:, i] for i in range(self.dim))
        # Scatter plot
        ax.scatter(*dims, cmap='hot')

        # Display the plot
        plt.show()

    @classmethod
    def geodesic_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculates geodesic distance between two points

        In a unit sphere, this is just the radius (1) multiplied by the angle
        """
        if np.array_equal(x1,x2):
            return 0

        return np.arccos(np.dot(x1, x2))

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        """Returns distance matrix of geodesic distances
        """
        return metrics.pairwise_distances(self.distribution, metric=self.geodesic_distance)


def get_spherical_uniform_distribution(dim: int, num_samples:int, radius: float = 1) -> np.ndarray:
    """Generate uniform distribution over a sphere

    'num_samples' samples of vectors of dimension N 
    with an uniform distribution on the (N-1)-Sphere surface of radius R.

    RATIONALE: https://mathworld.wolfram.com/HyperspherePointPicking.html
    """
    # Return 'num_samples' samples of vectors of dimension N 
    # with an uniform distribution on the (N-1)-Sphere surface of radius R.
    # RATIONALE: https://mathworld.wolfram.com/HyperspherePointPicking.html
    
    X = np.random.default_rng().normal(size=(num_samples , dim))

    return radius / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X