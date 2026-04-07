# ruff: noqa: PLC0414
from __future__ import annotations

from functools import wraps

from torch import is_tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from einops import pack, unpack
from torch_einops_utils import (
    align_dims_left as align_dims_left,
    and_masks as and_masks,
    default,
    exists,
    first,
    identity,
    lens_to_mask as lens_to_mask,
    or_masks as or_masks,
    pad_at_dim as pad_at_dim,
    pad_left_at_dim as pad_left_at_dim,
    pad_left_at_dim_to as pad_left_at_dim_to,
    pad_left_ndim as pad_left_ndim,
    pad_left_ndim_to as pad_left_ndim_to,
    pad_ndim as pad_ndim,
    pad_right_at_dim as pad_right_at_dim,
    pad_right_at_dim_to as pad_right_at_dim_to,
    pad_right_ndim as pad_right_ndim,
    pad_right_ndim_to as pad_right_ndim_to,
    pad_sequence as pad_sequence,
    pad_sequence_and_cat as pad_sequence_and_cat,
    safe_cat as safe_cat,
    safe_stack as safe_stack,
    shape_with_replace as shape_with_replace,
    slice_at_dim as slice_at_dim,
    slice_left_at_dim as slice_left_at_dim,
    slice_right_at_dim as slice_right_at_dim
)


def masked_mean(
    t,
    mask=None,
    dim=None,
    eps=1e-5,
):
    if not exists(mask):
        return t.mean(dim=dim) if exists(dim) else t.mean()

    if mask.ndim < t.ndim:
        mask = pad_right_ndim(mask, t.ndim - mask.ndim)

    mask = mask.expand_as(t)

    if not exists(dim):
        return t[mask].mean() if mask.any() else t[mask].sum()

    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim)

    return num / den.clamp(min=eps)


# tree flatten with inverse


def tree_map_tensor(fn, tree):
    return tree_map(lambda t: fn(t) if is_tensor(t) else t, tree)


def tree_flatten_with_inverse(tree):
    flattened, spec = tree_flatten(tree)

    def inverse(out):
        return tree_unflatten(out, spec)

    return flattened, inverse


# einops pack


def pack_with_inverse(t, pattern):
    is_one = is_tensor(t)

    if is_one:
        t = [t]

    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, packed_shape, inv_pattern)

        if is_one:
            out = first(out)

        return out

    return packed, inverse


def maybe(fn):

    if not exists(fn):
        return identity

    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner
