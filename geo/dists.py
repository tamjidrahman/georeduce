from functools import cached_property
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from abc import ABC, abstractmethod, abstractclassmethod

class Distribution(ABC):
    """Generic Distribution Class"""

    def __init__(self, dim: int, num_samples: int, generate=True):

        self.dim = dim
        self.num_samples = num_samples
        if generate:
            self.distribution = self.generate_distribution()

    @abstractmethod
    def generate_distribution(self) -> np.ndarray:
        """Generate distribution
        """
        pass

    def plot(self):
        """ Helper to show 2 or 3 dimensional plots
        """
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

    @abstractclassmethod
    def distance_metric(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Return distance metric for this kind of distribution"""
        pass

    @cached_property
    def distance_matrix(self) -> np.ndarray:
        """Returns distance matrix of geodesic distances
        """
        return metrics.pairwise_distances(self.distribution, metric=self.distance_metric)

class SphericalDistribution(Distribution):
    """A distribution of D dimensional vectors embedded in S^(D-1) unit sphere
    """

    def generate_distribution(self) -> np.ndarray:
        """Generate uniform distribution over a sphere

        'num_samples' samples of vectors of dimension N 
        with an uniform distribution on the (N-1)-Sphere surface of radius R.

        RATIONALE: https://mathworld.wolfram.com/HyperspherePointPicking.html
        """

        RADIUS = 1
        X = np.random.default_rng().normal(size=(self.num_samples , self.dim))

        return RADIUS / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X

    
    def distance_metric(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculates geodesic distance between two points

        In a unit sphere, this is just the radius (1) multiplied by the angle
        """
        if np.array_equal(x1,x2):
            return 0

        return np.arccos(np.dot(x1, x2))
    
class HammingDistribution(Distribution):
    """A distribution of D dimensional {0,1} vectors with hamming distance
    """

    def generate_distribution(self) -> np.ndarray:
        """Generate random {0,1} vectors of dimension D
        """
        
        return np.random.randint(2, size=(self.num_samples, self.dim))

    
    def distance_metric(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculates hamming distance between two points
        """
        return np.count_nonzero(x1!=x2)
