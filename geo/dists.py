import numpy as np, array

def get_spherical_uniform_distribution(dim: int, num_samples, radius: float = 1) -> array:
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