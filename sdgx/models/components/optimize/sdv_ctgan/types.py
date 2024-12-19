from __future__ import annotations

from enum import unique
from typing import List, Union

from sdgx.models.components.sdv_rdt.transformers import (
    ClusterBasedNormalizer,
    NormalizedFrequencyEncoder,
    NormalizedLabelEncoder,
    OneHotEncoder,
)
from sdgx.models.components.utils import StrValuedBaseEnum

CategoricalEncoderInstanceType = Union[
    OneHotEncoder, NormalizedFrequencyEncoder, NormalizedLabelEncoder
]
ContinuousEncoderInstanceType = Union[ClusterBasedNormalizer]
TransformerEncoderInstanceType = Union[
    CategoricalEncoderInstanceType, ContinuousEncoderInstanceType
]


@unique
class ActivationFuncType(StrValuedBaseEnum):
    SOFTMAX = "softmax"
    TANH = "tanh"
    LINEAR = "linear"


@unique
class ColumnTransformType(StrValuedBaseEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class SpanInfo:
    def __init__(self, dim: int, activation_fn: ActivationFuncType | str):
        self.dim: int = dim
        self.activation_fn: ActivationFuncType = ActivationFuncType(activation_fn)
    def __str__(self):
        return str((self.dim, self.activation_fn))
    __repr__ = __str__


class ColumnTransformInfo:
    def __init__(
        self,
        column_name: str,
        column_type: ColumnTransformType | str,
        transform: TransformerEncoderInstanceType,
        output_info: List[SpanInfo],
        output_dimensions: int,
    ):
        self.column_name: str = column_name
        self.column_type: ColumnTransformType = ColumnTransformType(column_type)
        self.transform: TransformerEncoderInstanceType = transform
        self.output_info: List[SpanInfo] = output_info
        self.output_dimensions: int = output_dimensions
    def to_str_log(self):
        return "\t".join(map(str, {
            "column_type": self.column_type,
            "transform": str(self.transform.__class__.__name__),
            "output_dimensions": self.output_dimensions,
            "column_name": self.column_name,
            "output_info": str(self.output_info)
        }.values()))
