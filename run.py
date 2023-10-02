from typing import NamedTuple
from encoder.hyperplane_encoder import HyperplaneEncoder
from geo.dists import SphericalDistribution, HammingDistribution
from matplotlib import pyplot as plt
from encoder.quadrant_encoder import QuadrantEncoder
from encoder.nearest_point_encoder import NearestPointEncoder
from encoder.encoder import IdentityEncoder
import logging
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
sns.set()


logger.info("Creating Distribution")
dist = SphericalDistribution(3, 1000)

encoders = [
    ("quadrant", QuadrantEncoder(input_distribution=dist)),
    (
        "nearest point",
        NearestPointEncoder(
            num_points=100, neighborhood_size=15, input_distribution=dist
        ),
    ),
    ("hyperplane", HyperplaneEncoder(num_planes=100, input_distribution=dist)),
    ("identity", IdentityEncoder(input_distribution=dist)),
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
    columns=["input_distance", *(encoder_name for encoder_name, _ in encoders)],
)

plot = sns.relplot(
    data=dist_matrix_3d.melt(
        "input_distance", value_name="distance", var_name="encoding type"
    ),
    kind="line",
    x="distance",
    y="input_distance",
    col="encoding type",
    errorbar=("pi", 100),
    col_wrap=2,
    facet_kws=dict(sharex=False, sharey=False),
)

plt.show()
