# DanLing
# Copyright (C) 2022-Present  DanLing

# This file is part of DanLing.

# DanLing is free software: you can redistribute it and/or modify
# it under the terms of the following licenses:
# - The Unlicense
# - GNU Affero General Public License v3.0 or later
# - GNU General Public License v2.0 or later
# - BSD 4-Clause "Original" or "Old" License
# - MIT License
# - Apache License 2.0

# DanLing is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file for more details.

r"""Packed segmented primitives.

Operators that act along a ragged dimension have historically materialized a dense padded
tensor, paying ``O(B x max_len)`` memory to run a dense kernel and mask the padding back
out. These primitives do the same work on the packed values directly.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import Tensor


@contextmanager
def _native_composite():
    r"""Resolve lazy arithmetic created inside native composites under AOT.

    AOT's boxed runtime excludes conjugate and negative dispatch. Native
    cumprod backward creates conjugate views internally, so its arithmetic and
    the final copy must resolve those views before leaving this local guard.
    """
    with (
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Conjugate, False),
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.Negative, False),
    ):
        yield


@contextmanager
def _native_autograd():
    r"""Build a local native derivative graph below a custom-op boundary."""
    with (
        _native_composite(),
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.AutogradFunctionality, False),
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, False),
        torch.enable_grad(),
    ):
        yield


def _native_vjp(outputs, arguments, cotangents, *, create_graph=False):
    r"""Differentiate the live native edges, including constant empty scans."""
    live_outputs = [(output, grad) for output, grad in zip(outputs, cotangents) if output.requires_grad]
    selected = tuple(argument for argument in arguments if argument.requires_grad)
    if live_outputs and selected:
        gradients = iter(
            torch.autograd.grad(
                tuple(output for output, _ in live_outputs),
                selected,
                tuple(grad for _, grad in live_outputs),
                create_graph=create_graph,
                allow_unused=True,
            )
        )
    else:
        gradients = iter(None for _ in selected)
    result = []
    for argument in arguments:
        gradient = next(gradients) if argument.requires_grad else None
        if gradient is None and argument.requires_grad and create_graph:
            gradient = argument.reshape(-1)[:0].sum().expand_as(argument)
        result.append(gradient)
    return tuple(result)


def _segmented_cdist_lengths(row_lengths: Tensor, column_lengths: Tensor) -> tuple[list[int], list[int]]:
    r"""Validate CPU length metadata and return concrete runtime segment sizes."""
    if row_lengths.dim() != 1 or column_lengths.dim() != 1:
        raise ValueError("segmented cdist lengths must be one-dimensional")
    if row_lengths.shape != column_lengths.shape:
        raise ValueError("segmented cdist length tensors must contain the same number of samples")
    if row_lengths.device.type != "cpu" or column_lengths.device.type != "cpu":
        raise ValueError("segmented cdist lengths must be CPU tensors")
    if row_lengths.dtype != torch.long or column_lengths.dtype != torch.long:
        raise TypeError("segmented cdist lengths must use torch.int64")
    rows = [int(length) for length in row_lengths.tolist()]
    columns = [int(length) for length in column_lengths.tolist()]
    if any(length < 0 for length in rows) or any(length < 0 for length in columns):
        raise ValueError("segmented cdist lengths must be non-negative")
    return rows, columns


def _validate_segmented_cdist_operands(
    left: Tensor,
    right: Tensor,
    rows: list[int],
    columns: list[int],
    compute_mode: int,
) -> None:
    if left.dim() != 2 or right.dim() != 2:
        raise ValueError("segmented cdist operands must be two-dimensional")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("segmented cdist operands must have the same device and dtype")
    if left.shape[1] != right.shape[1]:
        raise ValueError("segmented cdist operands must have the same feature width")
    if sum(rows) != left.shape[0] or sum(columns) != right.shape[0]:
        raise ValueError("segmented cdist lengths must cover their packed operands exactly")
    if compute_mode not in (0, 1, 2):
        raise ValueError(f"invalid segmented cdist compute_mode code: {compute_mode}")


def _mm_cdist_backward(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    distances: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Differentiate the Euclidean MM path through its composite matrix formula.

    Dense cdist's Euclidean MM path is composite-autograd. Writing the same
    derivative as two matrix products preserves the live matmul precision
    policy, supports FP16/BF16 where native ``_cdist_backward`` does not, and
    needs only the distance matrix rather than a ``[P, R, M]`` difference.
    """
    scale = torch.where(distances != 0, grad / distances, torch.zeros_like(grad))
    grad_left = left * scale.sum(dim=1, keepdim=True) - scale @ right
    grad_right = right * scale.sum(dim=0).unsqueeze(1) - scale.mT @ left
    return grad_left, grad_right


def _differentiable_cdist_backward(
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor]:
    r"""Keep the native per-segment VJP graph for subsequent derivatives."""
    rows, columns = _segmented_cdist_lengths(row_lengths, column_lengths)
    left_gradients, right_gradients = [], []
    left_start = right_start = output_start = 0
    for row_count, column_count in zip(rows, columns):
        pair_count = row_count * column_count
        left_segment = left.narrow(0, left_start, row_count)
        right_segment = right.narrow(0, right_start, column_count)
        grad_segment = grad_output.narrow(0, output_start, pair_count).reshape(row_count, column_count)
        distance_segment = distances.narrow(0, output_start, pair_count).reshape(row_count, column_count)
        uses_mm = p == 2 and (compute_mode == 1 or (compute_mode == 0 and (row_count > 25 or column_count > 25)))
        if uses_mm:
            left_vjp, right_vjp = _mm_cdist_backward(grad_segment, left_segment, right_segment, distance_segment)
        else:
            left_vjp = torch.ops.aten._cdist_backward.default(
                grad_segment.unsqueeze(0),
                left_segment.unsqueeze(0),
                right_segment.unsqueeze(0),
                p,
                distance_segment.unsqueeze(0),
            ).squeeze(0)
            right_vjp = torch.ops.aten._cdist_backward.default(
                grad_segment.mT.unsqueeze(0),
                right_segment.unsqueeze(0),
                left_segment.unsqueeze(0),
                p,
                distance_segment.mT.unsqueeze(0),
            ).squeeze(0)
        left_gradients.append(left_vjp)
        right_gradients.append(right_vjp)
        left_start += row_count
        right_start += column_count
        output_start += pair_count
    if not rows:
        return left.clone(), right.clone()
    return torch.cat(left_gradients), torch.cat(right_gradients)


@torch.library.custom_op("danling::_segmented_cdist_backward", mutates_args=())
def segmented_cdist_backward(
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor]:
    r"""Native per-segment VJP for :func:`segmented_cdist`."""
    rows, columns = _segmented_cdist_lengths(row_lengths, column_lengths)
    _validate_segmented_cdist_operands(left, right, rows, columns, compute_mode)
    total_pairs = sum(row * column for row, column in zip(rows, columns))
    if grad_output.numel() != total_pairs or distances.numel() != total_pairs:
        raise ValueError("segmented cdist gradients and distances must cover every output pair")

    grad_left = torch.zeros_like(left)
    grad_right = torch.zeros_like(right)
    left_start = right_start = output_start = 0
    for row_count, column_count in zip(rows, columns):
        pair_count = row_count * column_count
        if row_count and column_count and left.shape[1]:
            left_segment = left[left_start : left_start + row_count]
            right_segment = right[right_start : right_start + column_count]
            grad_segment = grad_output[output_start : output_start + pair_count].reshape(row_count, column_count)
            distance_segment = distances[output_start : output_start + pair_count].reshape(row_count, column_count)
            uses_mm = p == 2 and (compute_mode == 1 or (compute_mode == 0 and (row_count > 25 or column_count > 25)))
            if uses_mm:
                left_vjp, right_vjp = _mm_cdist_backward(
                    grad_segment,
                    left_segment,
                    right_segment,
                    distance_segment,
                )
                grad_left[left_start : left_start + row_count].copy_(left_vjp)
                grad_right[right_start : right_start + column_count].copy_(right_vjp)
            else:
                torch.ops.aten._cdist_backward.out(
                    grad_segment.unsqueeze(0),
                    left_segment.unsqueeze(0),
                    right_segment.unsqueeze(0),
                    p,
                    distance_segment.unsqueeze(0),
                    out=grad_left[left_start : left_start + row_count].unsqueeze(0),
                )
                torch.ops.aten._cdist_backward.out(
                    grad_segment.mT.unsqueeze(0),
                    right_segment.unsqueeze(0),
                    left_segment.unsqueeze(0),
                    p,
                    distance_segment.mT.unsqueeze(0),
                    out=grad_right[right_start : right_start + column_count].unsqueeze(0),
                )
        left_start += row_count
        right_start += column_count
        output_start += pair_count
    return grad_left, grad_right


@segmented_cdist_backward.register_fake
def _segmented_cdist_backward_fake(
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor]:
    del grad_output, row_lengths, column_lengths, distances, p, compute_mode
    return torch.empty_like(left), torch.empty_like(right)


@torch.library.custom_op("danling::_segmented_cdist_backward_vjp", mutates_args=())
def _segmented_cdist_backward_vjp(
    grad_left: Tensor,
    grad_right: Tensor,
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Differentiate each native VJP without padding or feature-expanded pairs.

    Custom-op kernels run below autograd dispatch. Enable only its functionality
    key for this local, detached recomputation, preserving the other dispatch
    state (including autocast). No internal graph escapes the kernel.
    """
    with _native_autograd():
        arguments = tuple(tensor.detach().requires_grad_() for tensor in (grad_output, left, right, distances))
        gradients = _differentiable_cdist_backward(
            arguments[0], arguments[1], arguments[2], row_lengths, column_lengths, arguments[3], p, compute_mode
        )
        result = _native_vjp(gradients, arguments, (grad_left, grad_right))
    return tuple(
        torch.zeros_like(argument) if gradient is None else gradient.detach().clone()
        for argument, gradient in zip(arguments, result)
    )


@_segmented_cdist_backward_vjp.register_fake
def _segmented_cdist_backward_vjp_fake(
    grad_left: Tensor,
    grad_right: Tensor,
    grad_output: Tensor,
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    distances: Tensor,
    p: float,
    compute_mode: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del grad_left, grad_right, row_lengths, column_lengths, p, compute_mode
    return torch.empty_like(grad_output), torch.empty_like(left), torch.empty_like(right), torch.empty_like(distances)


def _segmented_cdist_backward_setup_context(ctx, inputs, output) -> None:
    del output
    grad_output, left, right, row_lengths, column_lengths, distances, p, compute_mode = inputs
    ctx.save_for_backward(grad_output, left, right, row_lengths, column_lengths, distances)
    ctx.p, ctx.compute_mode = p, compute_mode


def _segmented_cdist_backward_autograd(ctx, grad_left, grad_right):
    grad_output, left, right, row_lengths, column_lengths, distances = ctx.saved_tensors
    if torch.is_grad_enabled():
        arguments = (grad_output, left, right, distances)
        gradients = _differentiable_cdist_backward(
            grad_output, left, right, row_lengths, column_lengths, distances, ctx.p, ctx.compute_mode
        )
        result = _native_vjp(gradients, arguments, (grad_left, grad_right), create_graph=True)
        return result[0], result[1], result[2], None, None, result[3], None, None
    grad_grad_output, grad_left, grad_right, grad_distances = _segmented_cdist_backward_vjp(
        grad_left,
        grad_right,
        grad_output,
        left,
        right,
        row_lengths,
        column_lengths,
        distances,
        ctx.p,
        ctx.compute_mode,
    )
    return grad_grad_output, grad_left, grad_right, None, None, grad_distances, None, None


segmented_cdist_backward.register_autograd(
    _segmented_cdist_backward_autograd, setup_context=_segmented_cdist_backward_setup_context
)


@torch.library.custom_op("danling::_segmented_cdist", mutates_args=())
def segmented_cdist(
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    p: float,
    compute_mode: int,
) -> Tensor:
    r"""Run native cdist independently over packed variable-length segments.

    The operator is a single graph node with a data-dependent packed output
    extent. At runtime each segment retains PyTorch's optimized direct/MM
    selection and dtype behavior. No padded batch, cross-sample matrix, pair
    index tensor, or feature-expanded Cartesian gather is materialized.
    """
    rows, columns = _segmented_cdist_lengths(row_lengths, column_lengths)
    _validate_segmented_cdist_operands(left, right, rows, columns, compute_mode)
    output = left.new_empty((sum(row * column for row, column in zip(rows, columns)),))
    left_start = right_start = output_start = 0
    for row_count, column_count in zip(rows, columns):
        pair_count = row_count * column_count
        torch.ops.aten._cdist_forward.out(
            left[left_start : left_start + row_count],
            right[right_start : right_start + column_count],
            p,
            compute_mode,
            out=output[output_start : output_start + pair_count].view(row_count, column_count),
        )
        left_start += row_count
        right_start += column_count
        output_start += pair_count
    return output


@segmented_cdist.register_fake
def _segmented_cdist_fake(
    left: Tensor,
    right: Tensor,
    row_lengths: Tensor,
    column_lengths: Tensor,
    p: float,
    compute_mode: int,
) -> Tensor:
    del right, row_lengths, column_lengths, p, compute_mode
    total_pairs = torch.library.get_ctx().new_dynamic_size(min=0)
    return left.new_empty((total_pairs,))


def _segmented_cdist_setup_context(ctx, inputs, output) -> None:
    left, right, row_lengths, column_lengths, p, compute_mode = inputs
    ctx.save_for_backward(left, right, row_lengths, column_lengths, output)
    ctx.p = p
    ctx.compute_mode = compute_mode
    ctx.set_materialize_grads(False)


def _segmented_cdist_autograd_backward(ctx, grad_output):
    if grad_output is None:
        return None, None, None, None, None, None
    left, right, row_lengths, column_lengths, distances = ctx.saved_tensors
    if torch.is_grad_enabled():
        grad_left, grad_right = _differentiable_cdist_backward(
            grad_output, left, right, row_lengths, column_lengths, distances, ctx.p, ctx.compute_mode
        )
        return grad_left, grad_right, None, None, None, None
    grad_left, grad_right = segmented_cdist_backward(
        grad_output,
        left,
        right,
        row_lengths,
        column_lengths,
        distances,
        ctx.p,
        ctx.compute_mode,
    )
    return grad_left, grad_right, None, None, None, None


torch.library.register_autograd(
    segmented_cdist,
    _segmented_cdist_autograd_backward,
    setup_context=_segmented_cdist_setup_context,
)


def _segment_lengths(offsets: Tensor, total: int) -> list[int]:
    r"""Validate packed offsets and return concrete runtime segment lengths."""
    if offsets.dim() != 1:
        raise ValueError("segmented offsets must be one-dimensional")
    if offsets.device.type != "cpu":
        raise ValueError("segmented offsets must be a CPU tensor")
    if offsets.dtype != torch.long:
        raise TypeError("segmented offsets must use torch.int64")
    from .nested_tensor import _transform_metadata

    positions = [int(offset) for offset in _transform_metadata(offsets).tolist()]
    if not positions or positions[0] != 0:
        raise ValueError("segmented offsets must start at zero")
    if any(left > right for left, right in zip(positions, positions[1:])):
        raise ValueError("segmented offsets must be nondecreasing")
    if positions[-1] != total:
        raise ValueError("segmented offsets must cover the packed values exactly")
    return [right - left for left, right in zip(positions, positions[1:])]


@torch.library.custom_op("danling::_segmented_cumsum", mutates_args=())
def segmented_cumsum(values: Tensor, offsets: Tensor) -> Tensor:
    r"""Run native cumulative sums independently over packed sample boundaries.

    Each nonempty segment issues a native scan to preserve its accumulation
    semantics. Host scheduling is O(batch size); explicit storage stays linear
    in the packed input rather than a padded batch envelope.
    """
    lengths = _segment_lengths(offsets, values.shape[0])
    output = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            torch.ops.aten.cumsum.out(values.narrow(0, start, length), 0, out=output.narrow(0, start, length))
        start += length
    return output


@segmented_cumsum.register_fake
def _segmented_cumsum_fake(values: Tensor, offsets: Tensor) -> Tensor:
    del offsets
    return torch.empty_like(values)


def _segmented_cumsum_setup_context(ctx, inputs, output) -> None:
    _, offsets = inputs
    ctx.save_for_backward(offsets)


def _segmented_cumsum_autograd_backward(ctx, grad_output):
    (offsets,) = ctx.saved_tensors
    reversed_offsets = grad_output.shape[0] - offsets.flip(0)
    return segmented_cumsum(grad_output.flip(0), reversed_offsets).flip(0), None


torch.library.register_autograd(
    segmented_cumsum,
    _segmented_cumsum_autograd_backward,
    setup_context=_segmented_cumsum_setup_context,
)


@torch.library.custom_op("danling::_segmented_cumprod_backward", mutates_args=())
def segmented_cumprod_backward(
    grad_output: Tensor,
    values: Tensor,
    offsets: Tensor,
    output: Tensor,
) -> Tensor:
    r"""Apply native cumprod's first-order VJP independently to packed segments."""
    lengths = _segment_lengths(offsets, values.shape[0])
    if grad_output.shape != values.shape or output.shape != values.shape:
        raise ValueError("segmented cumprod values, output, and gradient must have the same shape")
    grad_values = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            segment = values.narrow(0, start, length)
            with _native_composite():
                grad_segment = torch.ops.aten.cumprod_backward.default(
                    grad_output.narrow(0, start, length),
                    segment,
                    0,
                    output.narrow(0, start, length),
                )
                grad_values.narrow(0, start, length).copy_(grad_segment)
        start += length
    return grad_values


@segmented_cumprod_backward.register_fake
def _segmented_cumprod_backward_fake(
    grad_output: Tensor,
    values: Tensor,
    offsets: Tensor,
    output: Tensor,
) -> Tensor:
    del grad_output, offsets, output
    return torch.empty_like(values)


def _differentiable_cumprod_backward(grad_output: Tensor, values: Tensor, offsets: Tensor, output: Tensor) -> Tensor:
    r"""Keep native zero, complex-conjugate, and non-finite VJP semantics."""
    lengths = _segment_lengths(offsets, values.shape[0])
    gradients = []
    start = 0
    for length in lengths:
        gradients.append(
            torch.ops.aten.cumprod_backward.default(
                grad_output.narrow(0, start, length),
                values.narrow(0, start, length),
                0,
                output.narrow(0, start, length),
            )
        )
        start += length
    return torch.cat(gradients) if gradients else values.clone()


@torch.library.custom_op("danling::_segmented_cumprod_backward_vjp", mutates_args=())
def _segmented_cumprod_backward_vjp(
    cotangent: Tensor, grad_output: Tensor, values: Tensor, offsets: Tensor, output: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Native VJP of the opaque backward, with independent saved-output edges."""
    with _native_autograd():
        arguments = tuple(tensor.detach().requires_grad_() for tensor in (grad_output, values, output))
        gradients = _differentiable_cumprod_backward(arguments[0], arguments[1], offsets, arguments[2])
        result = _native_vjp((gradients,), arguments, (cotangent,))
    return tuple(
        torch.zeros_like(argument) if gradient is None else gradient.detach().clone()
        for argument, gradient in zip(arguments, result)
    )


@_segmented_cumprod_backward_vjp.register_fake
def _segmented_cumprod_backward_vjp_fake(
    cotangent: Tensor, grad_output: Tensor, values: Tensor, offsets: Tensor, output: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    del cotangent, offsets
    return torch.empty_like(grad_output), torch.empty_like(values), torch.empty_like(output)


def _segmented_cumprod_backward_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _segmented_cumprod_backward_autograd(ctx, cotangent):
    grad_output, values, offsets, output = ctx.saved_tensors
    if torch.is_grad_enabled():
        arguments = (grad_output, values, output)
        gradients = _differentiable_cumprod_backward(grad_output, values, offsets, output)
        result = _native_vjp((gradients,), arguments, (cotangent,), create_graph=True)
        return result[0], result[1], None, result[2]
    grad_grad_output, grad_values, grad_output_value = _segmented_cumprod_backward_vjp(
        cotangent, grad_output, values, offsets, output
    )
    return grad_grad_output, grad_values, None, grad_output_value


segmented_cumprod_backward.register_autograd(
    _segmented_cumprod_backward_autograd, setup_context=_segmented_cumprod_backward_setup_context
)


_cumprod_library = torch.library.Library("danling", "FRAGMENT")
_cumprod_library.define("_segmented_cumprod(Tensor values, Tensor offsets) -> Tensor")


def _segmented_cumprod_kernel(values: Tensor, offsets: Tensor) -> Tensor:
    r"""Run native cumprod independently over packed variable-length segments.

    Calling the device's native kernel per segment preserves its accumulation order, including
    device-specific rounding, underflow, zero, and non-finite behavior. The custom-op boundary
    keeps the data-dependent segment loop opaque to full-graph compilation without padding.
    """
    lengths = _segment_lengths(offsets, values.shape[0])
    output = torch.empty_like(values)
    start = 0
    for length in lengths:
        if length:
            torch.ops.aten.cumprod.out(
                values.narrow(0, start, length),
                0,
                out=output.narrow(0, start, length),
            )
        start += length
    return output


_cumprod_library.impl("_segmented_cumprod", _segmented_cumprod_kernel, "CompositeExplicitAutograd")
_segmented_cumprod = torch.ops.danling._segmented_cumprod.default


@torch.library.register_fake("danling::_segmented_cumprod")
def _segmented_cumprod_fake(values: Tensor, offsets: Tensor) -> Tensor:
    del offsets
    return torch.empty_like(values)


def _segmented_cumprod_setup_context(ctx, inputs, output) -> None:
    values, offsets = inputs
    ctx.save_for_backward(values, offsets, output)
    ctx.set_materialize_grads(False)


def _segmented_cumprod_autograd_backward(ctx, grad_output):
    if grad_output is None:
        return None, None
    values, offsets, output = ctx.saved_tensors
    if torch.is_grad_enabled():
        # Native cumprod_backward is itself differentiable. Keep its operations
        # visible when create_graph requests another derivative instead of
        # crossing the opaque first-order kernel boundary.
        return _differentiable_cumprod_backward(grad_output, values, offsets, output), None
    return segmented_cumprod_backward(grad_output, values, offsets, output), None


def _cumprod_derivative_arity(program: str) -> tuple[int, int]:
    r"""Count primal and derivative operands without inspecting tensor data."""
    inputs = outputs = 1
    for step in program.split("/") if program else ():
        added, _, targets = step[1:].partition(":")
        inputs += added.count("1")
        if step[0] in "vVbB":
            outputs = targets.count("1")
    return inputs, outputs


def _cumprod_reverse_mode():
    stack = torch._C._functorch.get_interpreter_stack() or ()
    live = bool(stack)
    # Preserve both live transform wrappers and the caller's create_graph state.
    # Upper/lower case distinguishes the two GradMode choices for each API.
    return ("v" if torch.is_grad_enabled() else "V") if live else ("B" if torch.is_grad_enabled() else "b")


def _cumprod_forward_mode():
    stack = torch._C._functorch.get_interpreter_stack() or ()
    return "j" if any(interpreter.key() == torch._C._functorch.TransformType.Jvp for interpreter in stack) else "f"


def _native_cumprod_derivative(program, offsets, arguments):
    r"""Replay AD over native scans after the mapped partition is selected."""
    if not program:
        lengths = _segment_lengths(offsets, arguments[0].shape[0])
        parts = [part.cumprod(0) for part in arguments[0].split(lengths)]
        return (torch.cat(parts) if parts else arguments[0].clone(),)
    prefix, _, step = program.rpartition("/")
    count, _ = _cumprod_derivative_arity(prefix)
    primals, derivatives = arguments[:count], arguments[count:]
    added, _, targets = step[1:].partition(":")
    active = [index for index, bit in enumerate(added) if bit == "1"]

    def parent(*args):
        # This prefix already ran at its original AD event. Rebuilding its
        # graph must not repeat old diagnostics under a newly enabled anomaly
        # mode. The new native VJP below runs in the restored caller context.
        with torch.autograd.set_detect_anomaly(False):
            return _native_cumprod_derivative(prefix, offsets, args)

    if step[0] in "vVbB":
        selected = [index for index, bit in enumerate(targets) if bit == "1"]
        create_graph = step[0] in "vB"

        def with_derivative(*args):
            operands = list(primals)
            for index, argument in zip(selected, args):
                operands[index] = argument
            outputs = parent(*operands)
            live = [(outputs[index], grad) for index, grad in zip(active, derivatives) if outputs[index].requires_grad]
            gradients = torch.autograd.grad(
                tuple(output for output, _ in live),
                args,
                tuple(grad for _, grad in live),
                create_graph=create_graph,
                allow_unused=True,
                materialize_grads=True,
            )
            return outputs, gradients

        selected_primals = tuple(primals[index] for index in selected)
        if step[0] in "vV":
            # func.grad differentiates with live GradTracking. Ordinary
            # backward must instead keep its unwrapped native graph.
            _, _, result = torch.func.vjp(with_derivative, *selected_primals, has_aux=True)
            return result
        return with_derivative(*selected_primals)[1]

    def selected_parent(*args):
        operands = list(primals)
        for index, argument in zip(active, args):
            operands[index] = argument
        return parent(*operands)

    if not active:
        return tuple(torch.zeros_like(output) for output in parent(*primals))
    if step[0] == "f":
        forward_ad = torch.autograd.forward_ad
        level = forward_ad.dual_level() if forward_ad._current_level < 0 else nullcontext()
        with level, forward_ad._set_fwd_grad_enabled(True):
            duals = tuple(
                forward_ad.make_dual(forward_ad.unpack_dual(primals[index]).primal, tangent)
                for index, tangent in zip(active, derivatives)
            )
            outputs = selected_parent(*duals)
            return tuple(forward_ad.unpack_dual(output).tangent for output in outputs)
    return torch.func.jvp(selected_parent, tuple(primals[index] for index in active), derivatives)[1]


_cumprod_library.define("_segmented_cumprod_derivative(Tensor[] arguments, Tensor offsets, str program) -> Tensor?[]")


def _segmented_cumprod_derivative_kernel(arguments: list[Tensor], offsets: Tensor, program: str) -> list[Tensor | None]:
    r"""Keep native derivative scheduling opaque to full-graph compilation."""
    with (
        _native_autograd(),
        # AOT's boxed runtime excludes these keys. Local native transforms
        # need their wrappers and front/back dispatch; the guards restore the
        # caller's exclusions on exit.
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.FuncTorchGradWrapper, False),
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.FuncTorchDynamicLayerFrontMode, False),
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.FuncTorchDynamicLayerBackMode, False),
        # Forward-over-reverse native AD can create efficient zero tensors,
        # which have no storage and require their own dispatch even under AOT.
        torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ZeroTensor, False),
    ):
        # Expanded cotangents can have overlapping storage, which make_dual
        # cannot use. Materializing the operands does not change their values.
        ordinary_targets: set[int] = set()
        for step in program.split("/"):
            if step[0] in "bB":
                targets = step.partition(":")[2]
                ordinary_targets.update(index for index, bit in enumerate(targets) if bit == "1")
        operands = tuple(
            argument.detach().clone().requires_grad_(index in ordinary_targets)
            for index, argument in enumerate(arguments)
        )
        return [
            None if result is None else result.detach().clone(memory_format=torch.contiguous_format)
            for result in _native_cumprod_derivative(program, offsets, operands)
        ]


_cumprod_library.impl(
    "_segmented_cumprod_derivative", _segmented_cumprod_derivative_kernel, "CompositeExplicitAutograd"
)
_segmented_cumprod_derivative = torch.ops.danling._segmented_cumprod_derivative.default


@torch.library.register_fake("danling::_segmented_cumprod_derivative")
def _segmented_cumprod_derivative_fake(arguments, offsets, program):
    del offsets
    outputs = [arguments[0]]
    for step in program.split("/"):
        if step[0] in "vVbB":
            targets = step.partition(":")[2]
            outputs = [arguments[index] for index, bit in enumerate(targets) if bit == "1"]
    return [output.new_empty(output.shape) for output in outputs]


class _SegmentedCumprodDerivative(torch.autograd.Function):
    r"""Batch each native AD program before reading its partition metadata."""

    @classmethod
    def apply(cls, program, offsets, *arguments):
        # Let Torch lower every functional transform before reading native
        # dual metadata; unpack_dual has no rule for nested batched wrappers.
        if torch._C._are_functorch_transforms_active():
            return super().apply(program, offsets, *arguments)
        forward_ad = torch.autograd.forward_ad
        parts = tuple(forward_ad.unpack_dual(argument) for argument in arguments)
        if not any(tangent is not None for _, tangent in parts):
            return super().apply(program, offsets, *arguments)
        primals = tuple(primal for primal, _ in parts)
        active = "".join("0" if tangent is None else "1" for _, tangent in parts)
        derivatives = tuple(tangent for _, tangent in parts if tangent is not None)
        # Function.jvp requires a Tensor tangent for every Tensor output.
        # Native AD also permits absent tangents for constant derivatives, so
        # split and rebuild native duals outside that interface. Functorch's
        # own apply dispatch lowers its wrappers before entering here again.
        with forward_ad._set_fwd_grad_enabled(False):
            results = super().apply(program, offsets, *primals)
            tangents = super().apply(f"{program}/f{active}", offsets, *primals, *derivatives)
        with forward_ad._set_fwd_grad_enabled(True):
            return tuple(
                primal if tangent is None else forward_ad.make_dual(primal, tangent)
                for primal, tangent in zip(results, tangents)
            )

    @staticmethod
    @torch.compiler.allow_in_graph
    def forward(program, offsets, *arguments):
        with torch._C._AutoDispatchBelowAutograd():
            return tuple(_segmented_cumprod_derivative(list(arguments), offsets, program))

    @staticmethod
    def setup_context(ctx, inputs, output):
        del output
        ctx.program, offsets, *arguments = inputs
        ctx.needs_grad = ctx.needs_input_grad[2:]
        saved = list(arguments)
        start = 1
        for step in ctx.program.split("/"):
            end = start + step[1:].partition(":")[0].count("1")
            if any(ctx.needs_grad[start:end]) and not any(ctx.needs_grad[:start] + ctx.needs_grad[end:]):
                # The requested derivative does not need this direction's
                # value, so native AD permits its alias to change. Replaying
                # earlier AD still needs its original IEEE values, however.
                # Save a detached snapshot without locking the source version.
                saved[start:end] = [argument.detach().clone() for argument in arguments[start:end]]
                break
            start = end
        ctx.save_for_backward(offsets, *saved)
        ctx.save_for_forward(offsets, *arguments)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, *cotangents):
        offsets, *arguments = ctx.saved_tensors
        # Absent derivatives must stay absent: adding explicit zero duals or
        # cotangents creates 0 * non-finite arithmetic that native AD omits.
        active = "".join("0" if grad is None else "1" for grad in cotangents)
        targets = "".join("1" if needed else "0" for needed in ctx.needs_grad)
        step = f"{_cumprod_reverse_mode()}{active}:{targets}"
        program = f"{ctx.program}/{step}"
        derivatives = (grad for grad in cotangents if grad is not None)
        gradients = _SegmentedCumprodDerivative.apply(program, offsets, *arguments, *derivatives)
        gradients = iter(gradients)
        return None, None, *(next(gradients) if needed else None for needed in ctx.needs_grad)

    @staticmethod
    def jvp(ctx, grad_program, grad_offsets, *tangents):
        del grad_program, grad_offsets
        offsets, *arguments = ctx.saved_tensors
        step = _cumprod_forward_mode() + "".join("0" if tangent is None else "1" for tangent in tangents)
        program = f"{ctx.program}/{step}"
        derivatives = (tangent for tangent in tangents if tangent is not None)
        return _SegmentedCumprodDerivative.apply(program, offsets, *arguments, *derivatives)

    @staticmethod
    def vmap(info, in_dims, program, offsets, *arguments):
        _, offsets_dim, *argument_dims = in_dims
        outputs = _cumprod_derivative_arity(program)[1]
        if info.batch_size == 0:
            # There are no partitions. Run the same native AD on zero rows,
            # retaining its primal/tangent edges without synthetic zero links.
            operands = [
                argument.unsqueeze(0).expand(0, *argument.shape) if dim is None else argument.movedim(dim, 0)
                for argument, dim in zip(arguments, argument_dims)
            ]
            result = _SegmentedCumprodDerivative.apply(program, offsets.new_zeros(1), *operands)
            return result, (0,) * outputs
        results = []
        for index in range(info.batch_size):
            operands = [
                argument if dim is None else argument.select(dim, index)
                for argument, dim in zip(arguments, argument_dims)
            ]
            partition = offsets if offsets_dim is None else offsets.select(offsets_dim, index)
            results.append(_SegmentedCumprodDerivative.apply(program, partition, *operands))
        return tuple(torch.stack(items) for items in zip(*results)), (0,) * outputs


class _SegmentedCumprod(torch.autograd.Function):
    r"""Expose the native segmented scan to reverse and forward transforms."""

    @staticmethod
    def vmap(info, in_dims, values: Tensor, offsets: Tensor):
        values_dim, offsets_dim = in_dims
        if offsets_dim is None:
            if values_dim is None:
                return _SegmentedCumprod.apply(values, offsets), None
            # Scanning the leading row axis treats mapped values as independent
            # static channels, so every member keeps the same sample boundaries.
            return _SegmentedCumprod.apply(values.movedim(values_dim, -1), offsets), values.dim() - 1
        if info.batch_size == 0:
            # There are no offset partitions to scan. Keep the empty result
            # connected to values, including when only offsets are mapped.
            empty = (
                values.unsqueeze(0).expand(0, *values.shape) if values_dim is None else values.movedim(values_dim, 0)
            )
            return empty.clone(), 0
        results = [
            _SegmentedCumprod.apply(
                values.select(values_dim, index) if values_dim is not None else values,
                offsets.select(offsets_dim, index),
            )
            for index in range(info.batch_size)
        ]
        return torch.stack(results), 0

    @staticmethod
    @torch.compiler.allow_in_graph
    def forward(values: Tensor, offsets: Tensor) -> Tensor:
        with torch._C._AutoDispatchBelowAutograd():
            return _segmented_cumprod(values, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        _segmented_cumprod_setup_context(ctx, inputs, output)
        ctx.save_for_forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is not None and torch._C._are_functorch_transforms_active():
            values, offsets, _ = ctx.saved_tensors
            return (
                _SegmentedCumprodDerivative.apply(f"{_cumprod_reverse_mode()}1:1", offsets, values, grad_output)[0],
                None,
            )
        return _segmented_cumprod_autograd_backward(ctx, grad_output)

    @staticmethod
    def jvp(ctx, grad_values, grad_offsets):
        del grad_offsets
        values, offsets = ctx.saved_tensors
        if grad_values is None:
            return torch.zeros_like(values)
        return _SegmentedCumprodDerivative.apply(f"{_cumprod_forward_mode()}1", offsets, values, grad_values)[0]


# Compiled graphs call these opaque operators directly, so their runtime
# Autograd dispatch must retain the same native duals as the public Functions.
_cumprod_library.impl("_segmented_cumprod", _SegmentedCumprod.apply, "Autograd")
_cumprod_library.impl(
    "_segmented_cumprod_derivative",
    lambda arguments, offsets, program: list(_SegmentedCumprodDerivative.apply(program, offsets, *arguments)),
    "Autograd",
)


@torch.compiler.allow_in_graph
def segmented_cumprod(values: Tensor, offsets: Tensor) -> Tensor:
    r"""Run native cumprod over packed segments with reverse and forward AD."""
    return _SegmentedCumprod.apply(values, offsets)


def align_rows(rows: Tensor, values: Tensor) -> Tensor:
    r"""
    Broadcast a per-row vector across the static tail of ``values``.

    Packed metadata — segment ids, local positions — carries one entry per packed row, while
    the values it describes are ``[N, *tail]``. Torch aligns a ``[N]`` vector to the *trailing*
    axis, so it has to be reshaped to lead before it is expanded: left as it is, a tail whose
    width happens to equal ``N`` broadcasts legally and silently answers the wrong question,
    and any other width raises an opaque size mismatch.

    Args:
        rows: One entry per packed row.
        values: The packed values ``rows`` describes.

    Returns:
        Tensor: ``rows`` with ``values``' shape, or ``rows`` itself when there is no tail.
    """
    if values.dim() <= 1:
        return rows
    return rows.view(-1, *([1] * (values.dim() - 1))).expand_as(values)


def segment_row_bounds(offsets: Tensor, total: int) -> tuple[Tensor, Tensor]:
    r"""
    Report, for every packed row, where its segment starts and how long that segment is.

    An operator that addresses a position *inside* a segment — a scatter destination, a
    per-segment lookup — needs both halves of this. The start rebases the position onto the
    packed buffer, and the length is the only thing that separates a legal write from one that
    lands in the neighbouring sample: once the position has been rebased, a dense kernel sees a
    row number that is in range for the buffer and writes it without complaint.

    Args:
        offsets: Segment offsets along packed dim 0, length ``segments + 1``.
        total: Number of packed rows, i.e. ``offsets[-1]``.

    Returns:
        tuple[Tensor, Tensor]: The starting row and the length of each row's segment.
    """
    lengths = offsets[1:] - offsets[:-1]
    starts = torch.repeat_interleave(offsets[:-1], lengths, output_size=total)
    return starts, torch.repeat_interleave(lengths, lengths, output_size=total)


def segmented_sort_perm(
    values: Tensor,
    offsets: Tensor,
    batch_idx: Tensor,
    *,
    descending: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""
    Sort each segment of a packed tensor along packed dim 0.

    Two stable global sorts compose into a segmented one: sort by value, then stably sort by
    segment id. Because the second pass is stable it regroups by segment while preserving the
    ordering the first pass established, so segments come out contiguous and internally sorted.

    Args:
        values: Packed values, sorted along dim 0. A static tail is sorted per column.
        offsets: Segment offsets, length ``batch + 1``.
        batch_idx: Segment id of every packed row.
        descending: Sort each segment in descending order.

    Returns:
        tuple[Tensor, Tensor]: The permutation into ``values``, and the same permutation
        expressed as per-segment indices.
    """
    segments = align_rows(batch_idx, values)
    by_value = torch.argsort(values, dim=0, stable=True, descending=descending)
    by_segment = torch.argsort(torch.gather(segments, 0, by_value), dim=0, stable=True)
    perm = torch.gather(by_value, 0, by_segment)
    return perm, perm - offsets.to(perm.device)[torch.gather(segments, 0, perm)]


def _shift_down(values: Tensor, step: int) -> Tensor:
    r"""Return ``values`` shifted ``step`` rows later, zero-filled at the front."""
    shifted = torch.zeros_like(values)
    shifted[step:] = values[:-step]
    return shifted


def _same_segment(batch_idx: Tensor, step: int, rank: int) -> Tensor:
    r"""Mask rows whose partner ``step`` back belongs to the same segment."""
    same = torch.zeros(batch_idx.shape[0], dtype=torch.bool, device=batch_idx.device)
    same[step:] = batch_idx[step:] == batch_idx[:-step]
    return same.view(-1, *([1] * (rank - 1)))


def segmented_scan(values: Tensor, batch_idx: Tensor, combine) -> Tensor:
    r"""
    Inclusive scan of ``combine`` within each segment.

    A log-step Hillis-Steele scan, masked so a row only combines with a partner in its own
    segment. The mask prevents unrelated samples from contributing to a prefix.
    Operators requiring native accumulation order use their dedicated segmented
    kernels instead; this helper applies to associative composite scans.

    A log-step scan combines whole blocks rather than one element at a time, so an intermediate
    result can leave the representable range even when every prefix is inside it. Callers whose
    ``combine`` can overflow — ``mul`` can, ``logaddexp`` cannot — should widen the dtype before
    scanning and narrow the result afterwards.

    Args:
        values: Packed values, scanned along dim 0. A static tail is scanned per column.
        batch_idx: Segment id of every packed row.
        combine: The associative operator to scan with.

    Returns:
        Tensor: The inclusive scan, with ``values``' shape.
    """
    total = values.shape[0]
    if total == 0:
        return values.clone()
    result = values.clone()
    step = 1
    while step < total:
        mask = _same_segment(batch_idx, step, result.dim())
        result = torch.where(mask, combine(result, _shift_down(result, step)), result)
        step *= 2
    return result


def segmented_arg_scan(
    values: Tensor,
    batch_idx: Tensor,
    local_idx: Tensor,
    *,
    largest: bool,
) -> tuple[Tensor, Tensor]:
    r"""
    Running extremum within each segment, with the per-segment index that produced it.

    The same masked log-step scan as :func:`segmented_scan`, carrying the index alongside the
    value so it follows whichever operand wins. ``candidate`` is the earlier window and
    ``running`` the later one, so a strict comparison keeps the *latest* tied index — which is
    what ``cummax`` and ``cummin`` report. NaN is absorbing for both, and a strict comparison
    is false against NaN, so NaN has to win explicitly or a later ordinary value would
    overwrite it.

    Args:
        values: Packed values, scanned along dim 0. A static tail is scanned per column.
        batch_idx: Segment id of every packed row.
        local_idx: Position of every packed row within its own segment.
        largest: Track the running maximum rather than the running minimum.

    Returns:
        tuple[Tensor, Tensor]: The running extremum and the per-segment index that produced it,
        both with ``values``' shape.
    """
    indices = align_rows(local_idx, values)
    total = values.shape[0]
    if total == 0:
        return values.clone(), indices.clone()
    running, indices = values.clone(), indices.clone()
    nan_absorbs = values.is_floating_point()
    step = 1
    while step < total:
        candidate, candidate_idx = _shift_down(running, step), _shift_down(indices, step)
        better = candidate > running if largest else candidate < running
        if nan_absorbs:
            better = better | (candidate.isnan() & ~running.isnan())
        take = _same_segment(batch_idx, step, running.dim()) & better
        running = torch.where(take, candidate, running)
        indices = torch.where(take, candidate_idx, indices)
        step *= 2
    return running, indices
