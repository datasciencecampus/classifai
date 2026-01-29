from enum import Enum


class MetricSettings(str, Enum):
    DOT_PRODUCT = "dot_product"
    L2_DISTANCE = "L2_distance"
