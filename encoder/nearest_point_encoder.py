from copy import deepcopy

import numpy as np
from sklearn.metrics import pairwise_distances

from encoder.encoder import Encoding, logger
from geo.dists import HammingDistribution


class NearestPointEncoder(Encoding):
    """Encode by generates a new distribution within the same space, and choosing the closest neighbors"""

    def __init__(self, num_points: int, neighborhood_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_points = num_points
        self.neighborhood_size = neighborhood_size
        self.reference_distribution = deepcopy(self.input_distribution)
        self.reference_distribution.num_samples = self.num_points
        self.reference_distribution.distribution = (
            self.reference_distribution.generate_distribution()
        )

    @property
    def transformed_distribution(self) -> HammingDistribution:
        logger.info("Encoding started")

        transform_distribution = HammingDistribution(
            dim=self.num_points,
            num_samples=self.input_distribution.num_samples,
            generate=False,
        )

        reference_distance_matrix = pairwise_distances(
            self.input_distribution.distribution,
            self.reference_distribution.distribution,
        )

        nearest_references = (
            reference_distance_matrix.T
            <= np.partition(reference_distance_matrix, self.neighborhood_size)[
                :, self.neighborhood_size - 1
            ]
        ).T.astype(int)

        transform_distribution.distribution = nearest_references

        logger.info("Encoding complete")
        return transform_distribution
