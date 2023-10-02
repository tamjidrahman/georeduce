from encoder.encoder import Encoding, logger
from geo.dists import HammingDistribution


import numpy as np


class QuadrantEncoder(Encoding):
    """Encodes Rn into quadrants. e.g. [[4,2,-1],[0,-6.2,3]] => [[1,1,0],[1,0,1]]"""

    @classmethod
    def to_quadrant(self, x1: np.ndarray):
        return (x1 >= 0).astype(int)

    @property
    def transformed_distribution(self) -> HammingDistribution:
        logger.info("Encoding")

        transform_distribution = HammingDistribution(
            dim=self.input_distribution.dim,
            num_samples=self.input_distribution.num_samples,
            generate=False,
        )
        new_distribution = np.apply_along_axis(
            self.to_quadrant, 1, self.input_distribution.distribution
        )
        transform_distribution.distribution = new_distribution

        return transform_distribution
