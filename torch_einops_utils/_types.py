from __future__ import annotations

from typing import ParamSpec, Protocol, TypedDict, TypeVar

DVar = TypeVar("DVar")
TVar = TypeVar("TVar")
T_co = TypeVar("T_co", covariant=True)

PSpec = ParamSpec("PSpec")


class DimAndValue(TypedDict, total=False):
    dim: int
    value: float


class SupportsIntIndex(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...

