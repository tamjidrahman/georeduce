import logging
from abc import ABC, abstractproperty

from geo.dists import Distribution

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
