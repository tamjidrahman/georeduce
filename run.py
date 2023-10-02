import logging

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from encoder.farthest_hyperplane_encoder import FarthestHyperPlaneEncoder
from encoder.hyperplane_encoder import HyperplaneEncoder
from encoder.nearest_point_encoder import NearestPointEncoder
from encoder.quadrant_encoder import QuadrantEncoder
from geo.dists import SphericalDistribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
sns.set()


logger.info("Creating Distribution")
dist = SphericalDistribution(10, 1000)

encoders = [
    ("quadrant", QuadrantEncoder(input_distribution=dist)),
    (
        "nearest points",
        NearestPointEncoder(
            num_points=100, neighborhood_size=50, input_distribution=dist
        ),
    ),
    ("hyperplane", HyperplaneEncoder(num_planes=100, input_distribution=dist)),
    # ("identity", IdentityEncoder(input_distribution=dist)),
    (
        "farthest hyperplane",
        FarthestHyperPlaneEncoder(
            num_planes=100, neighborhood_size=50, input_distribution=dist
        ),
    ),
]


data = pd.DataFrame(
    zip(
        dist.distribution,
        *(encoder.transformed_distribution.distribution for _, encoder in encoders)
    ),
    columns=["input", *(encoder_name for encoder_name, _ in encoders)],
)

dist_matrix_3d = pd.DataFrame(
    zip(
        dist.distance_matrix.flatten().round(1),
        *(
            encoder.transformed_distribution.distance_matrix.flatten().round(1)
            for _, encoder in encoders
        )
    ),
    columns=["geodesic distance", *(encoder_name for encoder_name, _ in encoders)],
)

plot = sns.relplot(
    data=dist_matrix_3d.melt(
        "geodesic distance", value_name="distance", var_name="encoding type"
    ),
    kind="line",
    x="distance",
    y="geodesic distance",
    col="encoding type",
    errorbar=("pi", 100),
    col_wrap=2,
    facet_kws=dict(sharex=False, sharey=False),
)

plt.show()
