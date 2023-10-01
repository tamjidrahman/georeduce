from abc import ABC, abstractproperty
from functools import cached_property
import numpy as np
from geo.dists import Distribution, SphericalDistribution, HammingDistribution
import logging

logger = logging.getLogger()


class Encoding(ABC):
    """
    Generic class to encode from one distribution to another
    """

    def __init__(self, input_distribution: Distribution):
        self.input_distribution = input_distribution

    @property
    def allowed_distributions(self):
        """Not all encoders may support all distributions"""
        return [Distribution]

    @abstractproperty
    def transformed_distribution(self) -> Distribution:
        """Transforms Distribution from one type to another"""
        pass


class IdentityEncoder(Encoding):
    """Encoder that does nothing/is passthrough

    Transform is identity, and distance is determined from the distribution
    """

    @property
    def transformed_distribution(self) -> Distribution:
        """Transforms Distribution from one type to another"""
        return self.input_distribution


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
