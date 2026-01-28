from enum import Enum


class MetricSettings(str, Enum):
    COSINE = "cosine"
    DOTPROD = "dotprod"
    COSINE_L2 = "cosinel2"
    DOTPROD_L2 = "dotprodl2"
    COSINE_L2_SQUARED = "cosinel2squared"
    DOTPROD_L2_SQUARED = "dotprodl2squared"
