import logging

import numpy as np

from encoder.encoder import Encoding, logger
from geo.dists import HammingDistribution

logger = logging.getLogger()


class FarthestHyperPlaneEncoder(Encoding):
    """Encode by generating m within the same space, and choosing the closest neighbors"""

    def __init__(self, num_planes: int, neighborhood_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_planes = num_planes
        self.neighborhood_size = neighborhood_size

        A_unnormal = np.random.normal(
            0, 1, (self.num_planes, self.input_distribution.distribution.shape[1])
        )
        row_sums = np.linalg.norm(A_unnormal, axis=1)
        self.A = A_unnormal / row_sums[:, np.newaxis]

    @property
    def transformed_distribution(self) -> HammingDistribution:
        logger.info("Encoding started")

        transform_distribution = HammingDistribution(
            dim=self.num_planes,
            num_samples=self.input_distribution.num_samples,
            generate=False,
        )

        distances_to_hyperplanes = self.A.dot(self.input_distribution.distribution.T).T

        farthest_hyperplanes = (
            distances_to_hyperplanes.T
            >= np.partition(distances_to_hyperplanes, -self.neighborhood_size)[
                :, -1 * self.neighborhood_size
            ]
        ).T.astype(int)

        transform_distribution.distribution = farthest_hyperplanes

        return transform_distribution
