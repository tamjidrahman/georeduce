from encoder.encoder import Encoding, logger
from geo.dists import HammingDistribution
from copy import deepcopy
from sklearn.metrics import pairwise_distances


import numpy as np


class HyperplaneEncoder(Encoding):
    """Encode by generates a new distribution within the same space, and choosing the closest neighbors"""

    def __init__(self, num_planes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_planes = num_planes

        self.A = np.random.normal(
            0, 1, (self.num_planes, self.input_distribution.distribution.shape[1])
        )

    @property
    def transformed_distribution(self) -> HammingDistribution:
        logger.info("Encoding started")

        transform_distribution = HammingDistribution(
            dim=self.num_planes,
            num_samples=self.input_distribution.num_samples,
            generate=False,
        )

        print("A", self.A)
        print("dist", self.input_distribution.distribution)
        # apply sign to below and try?
        transform_distribution.distribution = np.apply_along_axis(
            lambda x1: (x1 > 0).astype(int),
            1,
            self.A.dot(self.input_distribution.distribution.T).T,
        )

        print("result", transform_distribution.distribution)

        logger.info("Encoding complete")
        return transform_distribution
