from enum import Enum


class MetricSettings(str, Enum):
    INNER_PRODUCT = "inner_product"
    L2_DISTANCE = "L2_distance"
