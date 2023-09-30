import numpy as np
from vis.scatter import plot_array

class SphericalDistribution:
    """A distribution of D dimensional vectors embedded in S^(D-1) unit sphere
    """

    def __init__(self, dim: int, num_samples: int):

        self.dim = dim
        self.num_samples = num_samples

        self.distribution = get_spherical_uniform_distribution(dim, num_samples)

    def plot(self):
        if self.dim == 2:
            plot_array(self.distribution)
        raise Exception(NotImplemented)


    

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