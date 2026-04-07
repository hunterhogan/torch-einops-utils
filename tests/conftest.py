from __future__ import annotations

import torch
from torch import Tensor

import pytest


@pytest.fixture
def list_tensors() -> list[Tensor]:
    return [
        torch.tensor([[2.0, 3.0, 5.0, 73.0, 73.0], [7.0, 11.0, 13.0, 73.0, 73.0]]),
        torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]),
        torch.tensor([[17.0, 19.0, 23.0, 29.0, 31.0], [37.0, 41.0, 43.0, 47.0, 53.0]]),
        torch.tensor([[59.0, 61.0, 83.0, 83.0, 83.0], [67.0, 71.0, 83.0, 83.0, 83.0]]),
        torch.tensor([[59.0, 61.0, 97.0, 97.0, 97.0], [67.0, 71.0, 97.0, 97.0, 97.0]]),
        torch.tensor([[59.0, 61.0], [67.0, 71.0]]),
        torch.tensor([[73.0, 73.0, 2.0, 3.0, 5.0], [73.0, 73.0, 7.0, 11.0, 13.0]]),
        torch.tensor(
            [
                [79.0, 79.0, 79.0],
                [2.0, 3.0, 5.0],
                [7.0, 11.0, 13.0],
                [79.0, 79.0, 79.0],
                [79.0, 79.0, 79.0],
            ],
        ),
        torch.tensor(
            [
                [89.0, 89.0, 2.0, 3.0, 5.0, 89.0, 89.0, 89.0],
                [89.0, 89.0, 7.0, 11.0, 13.0, 89.0, 89.0, 89.0],
            ],
        ),
        torch.tensor([[89.0, 89.0, 89.0, 59.0, 61.0], [89.0, 89.0, 89.0, 67.0, 71.0]]),
        torch.tensor([[101.0, 101.0, 101.0, 59.0, 61.0], [101.0, 101.0, 101.0, 67.0, 71.0]]),
        torch.tensor([2, 0, 3]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([3, 5, 2]),
        torch.tensor([5.0, 7.0]),
        torch.tensor([11.0, 13.0]),
    ]


@pytest.fixture
def empty_tensor_sequence() -> list[Tensor]:
    return []


@pytest.fixture
def empty_optional_tensor_sequence() -> list[Tensor | None]:
    return []


