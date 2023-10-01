from geo.dists import SphericalDistribution, HammingDistribution
from matplotlib import pyplot as plt
from encoder.encoder import QuadrantEncoder
import logging
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


logger.info("Creating Distribution")
dist = SphericalDistribution(2, 100)
encoder = QuadrantEncoder(dist)

logger.info("Plotting")
sns.relplot(
    x=encoder.input_distribution.distance_matrix.flatten(),
    y=encoder.transformed_distribution.distance_matrix.flatten(),
)

plt.show()
