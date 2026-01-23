from typing import Literal, TypeAlias

metric_settings: TypeAlias = Literal[
    "cosine", "dotprod", "cosinel2", "dotprodl2", "cosinel2squared", "dotprodl2squared"
]
