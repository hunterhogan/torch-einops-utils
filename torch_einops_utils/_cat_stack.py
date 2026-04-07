from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, cat, stack

from torch_einops_utils import safe


@safe
def safe_stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    """Stack tensors from `tensors` along a new dimension, skipping `None` values.

    You can use `safe_stack` to stack a mixed sequence of `Tensor` and `None` values. The `safe` [1]
    decorator filters out `None` values from `tensors` before passing the remaining `Tensor` values
    to `torch.stack` [2]. If `tensors` contains no non-`None` values, `safe_stack` returns `None`.

    Parameters
    ----------
    tensors : Sequence[Tensor | None]
        A `Sequence` of `Tensor` or `None` values. `None` values are filtered out before stacking.
        All non-`None` `Tensor` values must have the same shape.
    dim : int = 0
        The dimension along which to stack. The result has one more dimension than each input
        `Tensor`.

    Returns
    -------
    stacked : Tensor | None
        The stacked `Tensor`, or `None` if `tensors` contains no non-`None` values.

    See Also
    --------
    safe_cat : Concatenate tensors along an existing dimension, skipping `None` values.

    Examples
    --------
    From the test suite [3]:

        >>> import torch
        >>> from torch_einops_utils import safe_stack
        >>> t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
        >>> safe_stack([]) is None
        True
        >>> safe_stack([None]) is None
        True
        >>> safe_stack([t1]).shape
        torch.Size([1, 2, 3])
        >>> safe_stack([t1, None]).shape  # None is skipped; only t1 is stacked
        torch.Size([1, 2, 3])
        >>> safe_stack([t1, t2]).shape
        torch.Size([2, 2, 3])

    From dreamer4 [4], collecting optional per-layer intermediates where some layers
    may not produce output (returning `None`):

        ```python
            intermediates = TransformerIntermediates(
                stack(time_attn_kv_caches),
                safe_stack(normed_time_attn_inputs),
                safe_stack(normed_space_attn_inputs),
                safe_stack(rnn_hiddens),
                hiddens,
        )

    References
    ----------
    [1] torch_einops_utils._helpers.safe

    [2] torch.stack - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.stack.html
    [3] tests/test_utils.py

    [4] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4
    """
    return stack(tensors, dim=dim)  # pyright: ignore[reportArgumentType] https://github.com/pytorch/pytorch/issues/179391


@safe
def safe_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    """Concatenate tensors from `tensors` along an existing dimension, skipping `None` values.

    You can use `safe_cat` to concatenate a mixed sequence of `Tensor` and `None` values. The `safe`
    [1] decorator filters out `None` values from `tensors` before passing the remaining `Tensor`
    values to `torch.cat` [2]. If `tensors` contains no non-`None` values, `safe_cat` returns `None`.

    A common pattern is iterative accumulation where the accumulator starts as `None`. On the first
    iteration, `safe_cat` receives one non-`None` `Tensor` and returns it unchanged. On subsequent
    iterations, `safe_cat` concatenates the accumulator with the new `Tensor`.

    Parameters
    ----------
    tensors : Sequence[Tensor | None]
        A sequence of `Tensor` or `None` values. `None` values are filtered out before concatenation.
        All non-`None` `Tensor` values must have the same shape in every dimension except `dim`.
    dim : int = 0
        The dimension along which to concatenate.

    Returns
    -------
    concatenated : Tensor | None
        The concatenated `Tensor`, or `None` if `tensors` contains no non-`None` values.

    See Also
    --------
    safe_stack : Stack tensors along a new dimension, skipping `None` values.

    Examples
    --------
    From the test suite [3]:

        >>> import torch
        >>> from torch_einops_utils import safe_cat
        >>> t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
        >>> safe_cat([]) is None
        True
        >>> safe_cat([None]) is None
        True
        >>> safe_cat([t1, None]).shape  # None is skipped; only t1 is returned
        torch.Size([2, 3])
        >>> safe_cat([t1, t2]).shape
        torch.Size([4, 3])

    From sdft_pytorch [4], accumulating per-step token losses across a generation loop where
    `token_kl_div_losses` is initialized to `None` before the loop:

        ```python
        token_kl_div_losses = safe_cat((token_kl_div_losses, token_kl_div), dim=1)
        ```

    References
    ----------
    [1] torch_einops_utils._helpers.safe

    [2] torch.cat - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.cat.html
    [3] tests/test_utils.py

    [4] lucidrains/sdft-pytorch
        https://github.com/lucidrains/sdft-pytorch
    """
    return cat(tensors, dim=dim)  # pyright: ignore[reportUnknownVariableType, reportArgumentType, reportCallIssue] https://github.com/pytorch/pytorch/issues/179391
