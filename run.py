from geo.dists import SphericalDistribution, HammingDistribution
from matplotlib import pyplot as plt
from encoder.quadrant_encoder import QuadrantEncoder
from encoder.nearest_point_encoder import NearestPointEncoder
import logging
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
sns.set()


logger.info("Creating Distribution")
dist = SphericalDistribution(2, 1000)
encoder = NearestPointEncoder(
    num_points=10, neighborhood_size=3, input_distribution=dist
)
inputdist_df = pd.DataFrame(dist.distribution, columns=["x", "y"])
ref_df = pd.DataFrame(encoder.reference_distribution.distribution, columns=["x", "y"])
sns.scatterplot(inputdist_df, x="x", y="y")
sns.scatterplot(ref_df, x="x", y="y")
plt.show()


logger.info("Plotting")

sns.relplot(
    # kind="line",
    x=encoder.transformed_distribution.distance_matrix.flatten().round(1),
    y=encoder.input_distribution.distance_matrix.flatten().round(1),
    # errorbar=("pi", 99),
)

plt.show()
