from __future__ import annotations

from functools import wraps

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Concatenate, List, Literal, TypeGuard, overload
from typing_extensions import Unpack
from torch.types import Number

import torch
from torch import tensor, is_tensor, cat, stack, arange, Tensor
import torch.nn.functional as F

from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map, PyTree

from einops import rearrange, repeat, reduce, pack, unpack

from torch_einops_utils import decreasing, DimAndValue, IdentityCallable, PSpec, RVar, SupportsIntIndex, T_co, TVar, zeroIndexed

# helper functions

def exists(v: TVar | None) -> TypeGuard[TVar]:
    return v is not None

def default(v: TVar | None, d: TVar) -> TVar:
    return v if exists(v) else d

def identity(t: TVar, *args: object, **kwargs: object) -> TVar:
    return t

def first(arr: SupportsIntIndex[TVar]) -> TVar:
    return arr[0]

def compact(arr: Iterable[T_co | None]) -> list[T_co]:
    return [*filter(exists, arr)]

@overload
def maybe(fn: Callable[Concatenate[TVar, PSpec], RVar]) -> Callable[Concatenate[TVar | None, PSpec], RVar | None]: ...
@overload
def maybe(fn: None) -> IdentityCallable: ...
def maybe(
    fn: Callable[Concatenate[TVar, PSpec], RVar] | None,
) -> Callable[Concatenate[TVar | None, PSpec], RVar | None] | IdentityCallable:
    if not exists(fn):
        return identity

    @wraps(fn)
    def inner(t: TVar | None, *args: PSpec.args, **kwargs: PSpec.kwargs) -> RVar | None:
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner

def safe(
    fn: Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None],
) -> Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]:

    @wraps(fn)
    def inner(tensors: Sequence[Tensor | None], *args: PSpec.args, **kwargs: PSpec.kwargs) -> Tensor | None:
        safe_tensors: list[Tensor] = compact(tensors)
        if len(safe_tensors) == 0:
            return None
        return fn(safe_tensors, *args, **kwargs)

    return inner

# exported functions

def masked_mean(
    t: Tensor,
    mask: Tensor | None = None,
    dim: int | None = None,
    eps: float = 1e-5,
) -> Tensor:
    if not exists(mask):
        return t.mean(dim = dim) if exists(dim) else t.mean()

    if mask.ndim < t.ndim:
        mask = pad_right_ndim(mask, t.ndim - mask.ndim)

    mask = mask.expand_as(t)

    if not exists(dim):
        return t[mask].mean() if mask.any() else t[mask].sum()

    num: Tensor = (t * mask).sum(dim = dim)
    den: Tensor = mask.sum(dim = dim)

    return num / den.clamp(min = eps)

# shapes

def shape_with_replace(
    t: Tensor,
    replace_dict: dict[int, int] | None = None
) -> torch.Size:
    shape: torch.Size = t.shape

    if not exists(replace_dict):
        return shape

    shape_list: list[int] = list(shape)

    for index, value in replace_dict.items():
        if index >= len(shape_list):
            message: str = f"I received `{index = }`, but I need `index` to be less than `{len(shape_list) = }`."
            raise ValueError(message)
        shape_list[index] = value

    return torch.Size(shape_list)

# slicing

def slice_at_dim(t: Tensor, slc: slice, dim: int = -1) -> Tensor:
    dims: int = t.ndim
    dim = (dim + dims) if dim < 0 else dim

    full_slice: list[slice] = [slice(None)] * dims
    full_slice[dim] = slc

    return t[tuple(full_slice)]

def slice_left_at_dim(t: Tensor, length: int, dim: int = -1) -> Tensor:
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim = dim)

    return slice_at_dim(t, slice(None, length), dim = dim)

def slice_right_at_dim(t: Tensor, length: int, dim: int = -1) -> Tensor:
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim = dim)

    return slice_at_dim(t, slice(-length, None), dim = dim)

# dimensions

def pad_ndim(t: Tensor, ndims: tuple[int, int]) -> Tensor:
    shape: tuple[int, ...] = t.shape
    left, right = ndims
    if left < 0 or right < 0:
        message: str = f"I received `{left = }` and `{right = }`, but I need both values to be greater than or equal to `0`."
        raise ValueError(message)

    ones: tuple[int] = (1,)
    ones_left: tuple[int, ...] = ones * left
    ones_right: tuple[int, ...] = ones * right
    return t.reshape(*ones_left, *shape, *ones_right)

def pad_left_ndim(t: Tensor, ndims: int) -> Tensor:
    return pad_ndim(t, (ndims, 0))

def pad_right_ndim(t: Tensor, ndims: int) -> Tensor:
    return pad_ndim(t, (0, ndims))

def pad_right_ndim_to(t: Tensor, ndims: int) -> Tensor:
    if t.ndim >= ndims:
        return t

    return pad_right_ndim(t, ndims - t.ndim)

def pad_left_ndim_to(t: Tensor, ndims: int) -> Tensor:
    if t.ndim >= ndims:
        return t

    return pad_left_ndim(t, ndims - t.ndim)

def align_dims_left(
    tensors: Sequence[Tensor],
    *,
    ndim: int | None = None,
) -> tuple[Tensor, ...]:
    if not exists(ndim):
        ndim = max([t.ndim for t in tensors])

    return tuple(pad_right_ndim(t, ndim - t.ndim) for t in tensors)

# cat and stack

@safe
def safe_stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    return stack(tensors, dim = dim)  # pyright: ignore[reportArgumentType] https://github.com/pytorch/pytorch/issues/179391

@safe
def safe_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    return cat(tensors, dim = dim)  # pyright: ignore[reportUnknownVariableType, reportCallIssue, reportArgumentType] https://github.com/pytorch/pytorch/issues/179391

# masking

def lens_to_mask(lens: Tensor, max_len: Number | None = None) -> Tensor:
    device: torch.device = lens.device

    if not exists(max_len):
        max_len = lens.amax().item()

    seq: Tensor = arange(max_len, device = device)
    lens = rearrange(lens, '... -> ... 1')
    return seq < lens

@safe
def reduce_masks(masks: Sequence[Tensor], op: Callable[[Tensor, Tensor], Tensor]) -> Tensor | None:
    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask: Tensor = op(mask, rest_mask)

    return mask

def and_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
    return reduce_masks(masks, torch.logical_and)

def or_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
    return reduce_masks(masks, torch.logical_or)

# padding

def pad_at_dim(
    t: Tensor,
    pad: tuple[int, int],
    *,
    dim: int = -1,
    value: float = 0.
) -> Tensor:
    dims_from_right: int = ((decreasing * dim) - zeroIndexed + t.ndim) % t.ndim
    zeros: tuple[Literal[0], ...] = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value = value)

def pad_left_at_dim(t: Tensor, pad: int, **kwargs: Unpack[DimAndValue]) -> Tensor:
    return pad_at_dim(t, (pad, 0), **kwargs)

def pad_right_at_dim(t: Tensor, pad: int, **kwargs: Unpack[DimAndValue]) -> Tensor:
    return pad_at_dim(t, (0, pad), **kwargs)

def pad_left_at_dim_to(t: Tensor, length: int, dim: int = -1, **kwargs: float) -> Tensor:
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_left_at_dim(t, length - curr_len, dim = dim, **kwargs)

def pad_right_at_dim_to(t: Tensor, length: int, dim: int = -1, **kwargs: float) -> Tensor:
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_right_at_dim(t, length - curr_len, dim = dim, **kwargs)

# better pad sequence

@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[True] = True,
    return_lens: Literal[False] = False,
    pad_lens: bool = False
) -> Tensor | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[True] = True,
    return_lens: Literal[True],
    pad_lens: bool = False
) -> tuple[Tensor, Tensor] | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[False],
    return_lens: Literal[False] = False,
    pad_lens: bool = False
) -> list[Tensor] | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[False],
    return_lens: Literal[True],
    pad_lens: bool = False
) -> tuple[list[Tensor], Tensor] | None: ...
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: bool = True,
    return_lens: bool = False,
    pad_lens: bool = False
) -> Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None:
    if len(tensors) == 0:
        return None

    device: torch.device = first(tensors).device

    lens: list[int] | Tensor = [t.shape[dim] for t in tensors]
    max_len: int = max(lens)

    pad_fn: Callable[..., Tensor] = pad_left_at_dim if left else pad_right_at_dim
    padded_tensors: list[Tensor] = [pad_fn(t, max_len - t_len, dim = dim, value = value) for t, t_len in zip(tensors, lens, strict=True)]

    output: Tensor | list[Tensor] = padded_tensors
    if return_stacked:
        output = stack(output, dim = dim_stack)

    if not return_lens:
        return output

    lens = tensor(lens, device=device)

    if pad_lens:
        lens = max_len - lens

    return output, lens

def pad_sequence_and_cat(
    tensors: Sequence[Tensor],
    *,
    dim_cat: int = 0,
    dim: int = -1,
    value: float = 0.,
    left: bool = False
) -> Tensor | None:

    padded: Tensor | list[Tensor] | None = pad_sequence(tensors, dim = dim, value = value, left = left, return_stacked = False, return_lens = False)
    if padded is not None:
        return cat(padded, dim = dim_cat)
    return padded

# tree flatten with inverse

def tree_map_tensor(fn: Callable[[Tensor], Tensor], tree: PyTree) -> PyTree:
    return tree_map(lambda t: fn(t) if is_tensor(t) else t, tree)

def tree_flatten_with_inverse(tree: PyTree) -> tuple[list[Any], Callable[[Iterable[Any]], PyTree]]:
    flattened, spec = tree_flatten(tree)

    def inverse(out: Iterable[Any]) -> PyTree:
        return tree_unflatten(out, spec)

    return flattened, inverse

# einops pack

@overload
def pack_with_inverse(t: Tensor, pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], Tensor]]: ...
@overload
def pack_with_inverse(t: list[Tensor], pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], list[Tensor]]]: ...
def pack_with_inverse(t: Tensor | list[Tensor], pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], Tensor | list[Tensor]]]:
    is_one: bool = is_tensor(t)

    if is_one:
        t = [t]

    packed, packed_shape = pack(t, pattern)

    def inverse(out: Tensor, inv_pattern: str | None = None) -> Tensor | list[Tensor]:
        inv_pattern = default(inv_pattern, pattern)
        unpacked: List[Tensor] = unpack(out, packed_shape, inv_pattern)

        if is_one:
            return first(unpacked)

        return unpacked

    return packed, inverse
