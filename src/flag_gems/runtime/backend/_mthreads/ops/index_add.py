# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.index_add import _validate_index_add_args
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INDEX_OUT_OF_BOUNDS_MESSAGE = "0 <= index < self.size(dim)"
_UNIQUE_DETECTOR_BLOCK = 256
_UNIQUE_DETECTOR_MAX_BITMAP_BYTES = 64 * 1024 * 1024
_UNIQUE_PATH_MIN_SUFFIX = 32
# The detector allocates/clears a receiver bitmap and synchronizes one status
# scalar.  Require enough contiguous work to amortize that fixed cost; smaller
# workloads remain on the existing #5679 atomic path.
_UNIQUE_PATH_MIN_UPDATES = 1 << 23
_ALL_SAME_PATH_MIN_SUFFIX = 32
_ALL_SAME_PATH_MIN_UPDATES = 1 << 15
_GROUPED_PATH_MIN_SUFFIX = 32
_GROUPED_PATH_MIN_UPDATES = 1 << 20
_GROUPED_REUSE_THRESHOLD = 2
_GROUPED_MAX_RECEIVERS = 8192
_NATIVE_FALLBACK_CPU_VALIDATE_MAX_INDEX = 8192
# CompositeExplicitAutograd bypasses the Python PrivateUse1 registration used
# by FlagGems while still reaching the MUSA native implementation underneath.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _read_index_bounds(index):
    lower, upper = torch.ops.aten.aminmax.default.redispatch(
        _FALLBACK_KEYSET, index, dim=None, keepdim=False
    )
    return lower.item(), upper.item()


def _resolve_index_for_kernel(index):
    # A contiguous lazy-negative tensor still exposes the un-negated storage
    # to a pointer-based Triton kernel. Materialize only that exceptional case.
    # Calling resolve_neg() from inside use_gems() re-enters FlagGems' Python
    # override and can negate the logical value twice. Toggle the metadata bit
    # off first, then explicitly negate the ordinary physical view.
    if index.is_neg():
        return torch.neg(torch._neg_view(index))
    return index


def _assert_index_in_bounds(index, upper_bound):
    if index.numel() == 0:
        return
    idx_min, idx_max = _read_index_bounds(index)
    if idx_min < 0 or idx_max >= upper_bound:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


def _assert_index_in_bounds_for_native_fallback(index, upper_bound):
    if index.numel() == 0:
        return
    # Native MUSA redispatch is fast for fallback cases, but it does not reject
    # invalid receivers before writing. Validate explicitly to preserve
    # index_add_ failure atomicity.
    if index.numel() <= _NATIVE_FALLBACK_CPU_VALIDATE_MAX_INDEX:
        host_index = index.cpu()
        idx_min = host_index.min().item()
        idx_max = host_index.max().item()
        if idx_min < 0 or idx_max >= upper_bound:
            raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
        return
    _assert_index_in_bounds(index, upper_bound)


def _native_index_add(inp, dim, index, src, alpha):
    return torch.ops.aten.index_add.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, src, alpha=alpha
    )


def _native_index_add_(inp, dim, index, src, alpha):
    return torch.ops.aten.index_add_.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, src, alpha=alpha
    )


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _can_use_contiguous_suffix_path(inp, dim, index, src):
    return (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and inp.dtype in (torch.float16, torch.float32)
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
        and _volume(src.shape[dim + 1 :]) > 1
    )


def _can_use_bf16_unique_path(inp, dim, index, src):
    if not (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype == torch.bfloat16
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    updates = prefix_size * index.numel() * suffix_size
    return (
        suffix_size >= _UNIQUE_PATH_MIN_SUFFIX
        and updates >= _UNIQUE_PATH_MIN_UPDATES
        and inp.size(dim) * 4 <= _UNIQUE_DETECTOR_MAX_BITMAP_BYTES
    )


def _can_use_bf16_all_same_path(inp, dim, index, src):
    if not (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype == torch.bfloat16
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    updates = prefix_size * index.numel() * suffix_size
    return (
        suffix_size >= _ALL_SAME_PATH_MIN_SUFFIX
        and updates >= _ALL_SAME_PATH_MIN_UPDATES
    )


def _can_use_bf16_grouped_path(inp, dim, index, src):
    if not (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype == torch.bfloat16
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    updates = prefix_size * index.numel() * suffix_size
    return (
        suffix_size >= _GROUPED_PATH_MIN_SUFFIX
        and updates >= _GROUPED_PATH_MIN_UPDATES
        and inp.size(dim) <= _GROUPED_MAX_RECEIVERS
    )


@libentry()
@triton.jit
def _index_add_unique_detector_kernel(
    status,
    bitmap,
    index,
    index_len,
    upper_bound,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    negative = mask & (values < 0)
    upper = mask & (values >= upper_bound)
    valid = mask & ~negative & ~upper

    has_negative = tl.max(negative.to(tl.int32))
    has_upper = tl.max(upper.to(tl.int32))

    safe_values = tl.where(valid, values, 0)
    previous = tl.atomic_add(bitmap + safe_values, 1, mask=valid)
    duplicate = valid & (previous > 0)
    has_duplicate = tl.max(duplicate.to(tl.int32))
    status_bits = has_negative + has_upper * 2 + has_duplicate * 4
    tl.atomic_or(status, status_bits, mask=status_bits != 0)


@libentry()
@triton.jit
def _index_add_all_same_detector_kernel(
    status,
    index,
    index_len,
    upper_bound,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    first = tl.load(index).to(tl.int64)
    negative = mask & (values < 0)
    upper = mask & (values >= upper_bound)
    different = mask & (values != first)
    status_bits = tl.max(negative.to(tl.int32))
    status_bits = status_bits + tl.max(upper.to(tl.int32)) * 2
    status_bits = status_bits + tl.max(different.to(tl.int32)) * 4
    tl.atomic_or(status, status_bits, mask=status_bits != 0)


@libentry()
@triton.jit
def _index_add_receiver_count_kernel(
    status_blocks,
    receiver_counts,
    index,
    index_len,
    upper_bound,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    negative = mask & (values < 0)
    upper = mask & (values >= upper_bound)
    valid = mask & ~negative & ~upper

    has_negative = tl.max(negative.to(tl.int32))
    has_upper = tl.max(upper.to(tl.int32))
    status_bits = has_negative + has_upper * 2
    tl.store(status_blocks + pid, status_bits)

    safe_values = tl.where(valid, values, 0)
    tl.atomic_add(receiver_counts + safe_values, 1, mask=valid)


@libentry()
@triton.jit
def _index_add_compact_receiver_groups_kernel(
    receiver_offsets,
    touched_receivers,
    group_offsets,
    meta,
    status_blocks,
    num_index_blocks,
    upper_bound,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < upper_bound
    block_status = tl.load(
        status_blocks + offsets, mask=offsets < num_index_blocks, other=0
    )
    has_negative = tl.max(((block_status & 1) != 0).to(tl.int32))
    has_upper = tl.max(((block_status & 2) != 0).to(tl.int32))
    counts = tl.load(receiver_offsets + offsets, mask=mask, other=0).to(tl.int32)
    touched = mask & (counts > 0)
    touched_i32 = touched.to(tl.int32)
    touched_rank = tl.cumsum(touched_i32, axis=0) - 1
    count_prefix = tl.cumsum(counts, axis=0)
    group_start = count_prefix - counts

    tl.store(touched_receivers + touched_rank, offsets.to(tl.int64), mask=touched)
    tl.store(group_offsets + touched_rank, group_start, mask=touched)
    tl.store(receiver_offsets + offsets, group_start, mask=touched)

    touched_total = tl.sum(touched_i32, axis=0)
    total = tl.sum(counts, axis=0)
    tl.store(meta, has_negative + has_upper * 2)
    tl.store(group_offsets + touched_total, total)
    tl.store(meta + 1, touched_total)


@libentry()
@triton.jit
def _index_add_group_positions_kernel(
    receiver_offsets,
    grouped_positions,
    index,
    index_len,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    receiver = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    group_pos = tl.atomic_add(receiver_offsets + receiver, 1, mask=mask)
    tl.store(grouped_positions + group_pos, offsets.to(tl.int32), mask=mask)


@libentry()
@triton.jit
def _index_add_group_positions_by_receiver_kernel(
    touched_receivers,
    group_offsets,
    grouped_positions,
    index,
    index_len,
    BLOCK: tl.constexpr,
):
    group_pid = ext.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < index_len
    receiver = tl.load(touched_receivers + group_pid).to(tl.int64)
    values = tl.load(index + offsets, mask=mask, other=-1).to(tl.int64)
    matched = mask & (values == receiver)
    rank = tl.cumsum(matched.to(tl.int32), axis=0) - 1
    start = tl.load(group_offsets + group_pid).to(tl.int32)
    tl.store(grouped_positions + start + rank, offsets.to(tl.int32), mask=matched)


def _validate_and_detect_unique(index, upper_bound):
    """Validate receivers and report uniqueness in one GPU pass.

    The bitmap is intentionally bounded.  Large receiver dimensions use the
    existing aminmax validation and atomic scatter path instead of allocating
    an unbounded auxiliary tensor.
    """
    if index.numel() == 0:
        return True
    bitmap = torch.zeros((upper_bound,), dtype=torch.int32, device=index.device)
    status = torch.zeros((1,), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(index.numel(), _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_unique_detector_kernel[grid](
            status,
            bitmap,
            index,
            index.numel(),
            upper_bound,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
    status_bits = int(status.cpu().item())
    if status_bits & 0x3:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    return not (status_bits & 0x4)


def _validate_and_detect_all_same(index, upper_bound):
    if index.numel() == 0:
        return False
    status = torch.zeros((1,), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(index.numel(), _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_all_same_detector_kernel[grid](
            status,
            index,
            index.numel(),
            upper_bound,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
    status_bits = int(status.cpu().item())
    if status_bits & 0x3:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    return not (status_bits & 0x4)


@libentry()
@triton.jit
def _index_add_unique_contiguous_suffix_kernel(
    out,
    index,
    src,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    ALPHA_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    suffix_pid = ext.program_id(0)
    index_pid = ext.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    prefix_pid = ext.program_id(2)
    cols = suffix_pid * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    index_mask = index_pid < index_len
    mask = index_mask & (cols < suffix_size)
    receiver = tl.load(index + index_pid, mask=index_mask, other=0).to(tl.int32)
    src_base = (prefix_pid * index_len + index_pid) * suffix_size
    dst_base = (prefix_pid * out_dim + receiver) * suffix_size
    src_ptrs = src + src_base + cols
    dst_ptrs = out + dst_base + cols
    values = tl.load(src_ptrs, mask=mask, other=0.0)
    current = tl.load(dst_ptrs, mask=mask, other=0.0)
    update = values if ALPHA_ONE else values * alpha
    tl.store(dst_ptrs, current + update, mask=mask)


def _run_bf16_unique_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    block_m = 4
    # S5000's BF16 direct path is resource-bound beyond a 256-wide suffix
    # tile; splitting wider suffixes improves occupancy and measured kernel
    # latency without changing addressing or memory traffic.
    block_n = min(256, triton.next_power_of_2(suffix_size))
    alpha_is_one = alpha == 1
    grid = (
        triton.cdiv(suffix_size, block_n),
        triton.cdiv(index.numel(), block_m),
        prefix_size,
    )
    with torch_device_fn.device(out.device):
        _index_add_unique_contiguous_suffix_kernel[grid](
            out,
            index,
            src,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ALPHA_ONE=alpha_is_one,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return out


@libentry()
@triton.jit
def _index_add_all_same_suffix_kernel(
    out,
    index,
    src,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce an all-same receiver group before updating the destination.

    One program owns one prefix and one contiguous suffix tile.  The complete
    index dimension is reduced in registers, replacing ``index_len`` BF16
    atomic RMW operations per destination element with one load/add/store.
    """
    suffix_pid = ext.program_id(0)
    prefix_pid = ext.program_id(1)
    cols = suffix_pid * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    col_mask = cols < suffix_size
    receiver = tl.load(index).to(tl.int64)
    summed = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in tl.range(0, index_len, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)[:, None]
        k_mask = k < index_len
        mask = k_mask & col_mask
        src_base = (prefix_pid * index_len + k) * suffix_size
        src_ptrs = src + src_base + cols
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)
        summed += tl.sum(values, axis=0)
    dst_base = (prefix_pid * out_dim + receiver) * suffix_size
    dst_ptrs = out + dst_base + cols
    current = tl.load(dst_ptrs, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(dst_ptrs, (current + summed * alpha).to(tl.bfloat16), mask=col_mask)


def _run_bf16_all_same_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    # Narrow tiles keep the K reduction's live register footprint bounded.
    block_n = min(32, triton.next_power_of_2(suffix_size))
    block_k = min(128, triton.next_power_of_2(index.numel()))
    grid = (triton.cdiv(suffix_size, block_n), prefix_size)
    with torch_device_fn.device(out.device):
        _index_add_all_same_suffix_kernel[grid](
            out,
            index,
            src,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
        )
    return out


def _build_bf16_receiver_groups(index, upper_bound):
    """Build compact receiver groups on the device for BF16 duplicate updates."""
    index_len = index.numel()
    receiver_offsets = torch.zeros(
        (upper_bound,), dtype=torch.int32, device=index.device
    )
    touched_receivers = torch.empty(
        (upper_bound,), dtype=torch.int64, device=index.device
    )
    group_offsets = torch.empty(
        (upper_bound + 1,), dtype=torch.int32, device=index.device
    )
    grouped_positions = torch.empty(
        (index_len,), dtype=torch.int32, device=index.device
    )
    meta = torch.empty((2,), dtype=torch.int32, device=index.device)

    with torch_device_fn.device(index.device):
        count_grid = (triton.cdiv(index_len, _UNIQUE_DETECTOR_BLOCK),)
        compact_block = triton.next_power_of_2(max(upper_bound, 1))
        _index_add_receiver_count_kernel[count_grid](
            group_offsets,
            receiver_offsets,
            index,
            index_len,
            upper_bound,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
        _index_add_compact_receiver_groups_kernel[(1,)](
            receiver_offsets,
            touched_receivers,
            group_offsets,
            meta,
            group_offsets,
            count_grid[0],
            upper_bound,
            BLOCK=compact_block,
        )
    meta_cpu = meta.cpu()
    status_bits = int(meta_cpu[0].item())
    if status_bits & 0x3:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    touched_count = int(meta_cpu[1].item())
    return (
        receiver_offsets,
        touched_receivers,
        group_offsets,
        grouped_positions,
        touched_count,
    )


def _scatter_bf16_receiver_group_positions(
    receiver_offsets,
    grouped_positions,
    index,
):
    grid = (triton.cdiv(index.numel(), _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_group_positions_kernel[grid](
            receiver_offsets,
            grouped_positions,
            index,
            index.numel(),
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )


def _scatter_bf16_receiver_group_positions_by_receiver(
    touched_receivers,
    group_offsets,
    grouped_positions,
    index,
    touched_count,
):
    block = triton.next_power_of_2(max(index.numel(), 1))
    with torch_device_fn.device(index.device):
        _index_add_group_positions_by_receiver_kernel[(touched_count,)](
            touched_receivers,
            group_offsets,
            grouped_positions,
            index,
            index.numel(),
            BLOCK=block,
            num_warps=8,
        )


@libentry()
@triton.jit
def _index_add_grouped_suffix_kernel(
    out,
    touched_receivers,
    group_offsets,
    grouped_positions,
    src,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    ALPHA_ONE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Receiver-owned reduction for arbitrary duplicate index groups."""
    suffix_pid = ext.program_id(0)
    group_pid = ext.program_id(1)
    prefix_pid = ext.program_id(2)
    cols = suffix_pid * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    col_mask = cols < suffix_size

    start = tl.load(group_offsets + group_pid).to(tl.int32)
    end = tl.load(group_offsets + group_pid + 1).to(tl.int32)
    group_len = end - start
    receiver = tl.load(touched_receivers + group_pid).to(tl.int64)

    summed = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in tl.range(0, group_len, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)[:, None]
        k_mask = k < group_len
        pos = tl.load(grouped_positions + start + k, mask=k_mask, other=0).to(tl.int64)
        mask = k_mask & col_mask
        src_base = (prefix_pid * index_len + pos) * suffix_size
        values = tl.load(src + src_base + cols, mask=mask, other=0.0).to(tl.float32)
        summed += tl.sum(values, axis=0)

    dst_base = (prefix_pid * out_dim + receiver) * suffix_size
    dst_ptrs = out + dst_base + cols
    current = tl.load(dst_ptrs, mask=col_mask, other=0.0).to(tl.float32)
    update = summed if ALPHA_ONE else summed * alpha
    tl.store(dst_ptrs, (current + update).to(tl.bfloat16), mask=col_mask)


def _run_bf16_grouped_path(
    out,
    dim,
    index,
    src,
    alpha,
    touched_receivers,
    group_offsets,
    grouped_positions,
    touched_count,
):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    block_n = min(512, triton.next_power_of_2(suffix_size))
    avg_group = triton.cdiv(index.numel(), max(touched_count, 1))
    if avg_group <= 8:
        block_k = 8
    elif avg_group <= 32:
        block_k = 16
    else:
        block_k = min(64, triton.next_power_of_2(avg_group))
    grid = (triton.cdiv(suffix_size, block_n), touched_count, prefix_size)
    with torch_device_fn.device(out.device):
        _index_add_grouped_suffix_kernel[grid](
            out,
            touched_receivers,
            group_offsets,
            grouped_positions,
            src,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ALPHA_ONE=alpha == 1,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
        )
    return out


def _try_run_bf16_receiver_owned_path(out, dim, index, src, alpha):
    groups = _build_bf16_receiver_groups(index, out.size(dim))
    (
        receiver_offsets,
        touched_receivers,
        group_offsets,
        grouped_positions,
        touched_count,
    ) = groups
    if touched_count == 1:
        return _run_bf16_all_same_path(out, dim, index, src, alpha), True
    if touched_count == index.numel():
        return _run_bf16_unique_path(out, dim, index, src, alpha), True
    if index.numel() < touched_count * _GROUPED_REUSE_THRESHOLD:
        return out, False
    if index.numel() <= 2048 and 64 < touched_count <= 128:
        _scatter_bf16_receiver_group_positions_by_receiver(
            touched_receivers,
            group_offsets,
            grouped_positions,
            index,
            touched_count,
        )
    else:
        _scatter_bf16_receiver_group_positions(
            receiver_offsets,
            grouped_positions,
            index,
        )
    return (
        _run_bf16_grouped_path(
            out,
            dim,
            index,
            src,
            alpha,
            touched_receivers,
            group_offsets,
            grouped_positions,
            touched_count,
        ),
        True,
    )


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def index_add_kernel(
    out_ptr,
    index_ptr,
    src_ptr,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel for index_add operation with autotune.

    After dim_compress, tensors are reshaped so that:
    - inp has shape (M, inp_len) where inp_len is the size of target dimension
    - src has shape (M, N) where N is the size of index

    For each row m and each index position n:
        out[m, index[n]] += alpha * src[m, n]
    """
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    # Calculate row and column offsets
    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Create masks
    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    # Load indices for this block of columns
    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    # Calculate offsets into inp/out (which has shape M x inp_len)
    inp_off = rows_offset * inp_len + cur_indices

    # Calculate offsets into src (which has shape M x N)
    src_off = rows_offset * N + cols_offset

    # Load source values
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    # Use atomic_add to correctly handle repeated indices in index,
    # aligned with the common op (src/flag_gems/ops/index_add.py).
    # When multiple source elements map to the same output position (duplicate
    # indices), plain load-store would cause race conditions or lost updates.
    # atomic_add guarantees all contributions are accumulated correctly.
    tl.atomic_add(out_ptr + inp_off, alpha * cur_src, mask=block_mask)


@libentry()
@triton.jit
def _index_add_contiguous_suffix_kernel(
    out,
    index,
    src,
    row_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    mask = row_mask & (cols < suffix_size)
    edge = rows % index_len
    prefix = rows // index_len
    receiver = tl.load(index + edge, mask=row_mask, other=0).to(tl.int64)
    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(out + out_offsets, values * alpha, mask=mask)


def _contiguous_suffix_config(suffix_size):
    block_n = min(512, triton.next_power_of_2(suffix_size))
    return 4, block_n


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    block_m, block_n = _contiguous_suffix_config(suffix_size)
    grid = (
        triton.cdiv(row_count, block_m),
        triton.cdiv(suffix_size, block_n),
    )
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_kernel[grid](
            out,
            index,
            src,
            row_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return out


def index_add(inp, dim, index, src, alpha=1):
    """
    Optimized index_add for mthreads backend.

    self.index_add_(dim, index, source, alpha=1) -> Tensor

    For a 3-D tensor the output is:
        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    """
    logger.debug("GEMS_MTHREADS INDEX_ADD")

    dim = _validate_index_add_args(inp, dim, index, src)
    if src.numel() == 0:
        return inp.clone(memory_format=torch.contiguous_format)

    input_src_alias = torch._C._is_alias_of(inp, src)
    use_bf16_grouped_path = (
        _can_use_bf16_grouped_path(inp, dim, index, src) and not input_src_alias
    )
    use_contiguous_suffix_path = False
    use_bf16_unique_path = False
    use_bf16_all_same_path = False
    if not use_bf16_grouped_path:
        use_contiguous_suffix_path = (
            _can_use_contiguous_suffix_path(inp, dim, index, src)
            and not input_src_alias
        )
        use_bf16_unique_path = (
            _can_use_bf16_unique_path(inp, dim, index, src) and not input_src_alias
        )
        use_bf16_all_same_path = (
            _can_use_bf16_all_same_path(inp, dim, index, src) and not input_src_alias
        )

    # Make inputs contiguous. resolve_neg() is a no-op for normal indices.
    inp = inp.contiguous()
    index = _resolve_index_for_kernel(index).contiguous()
    src = src.contiguous()

    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Reject invalid receivers before a pointer kernel can observe them.
    # Use min/max to avoid allocating full-size boolean tensors.
    unique_index = False
    all_same_checked = False
    index_validated = False
    if use_bf16_grouped_path:
        out, handled = _try_run_bf16_receiver_owned_path(
            inp.clone(), dim, index, src, alpha
        )
        index_validated = True
        if handled:
            return out
    if not index_validated:
        if use_bf16_all_same_path:
            all_same_checked = True
            if _validate_and_detect_all_same(index, inp_len):
                return _run_bf16_all_same_path(inp.clone(), dim, index, src, alpha)
            index_validated = True
        if use_bf16_unique_path:
            unique_index = _validate_and_detect_unique(index, inp_len)
            index_validated = True
        elif not all_same_checked and use_contiguous_suffix_path:
            _assert_index_in_bounds(index, inp_len)
            index_validated = True

    if unique_index:
        return _run_bf16_unique_path(inp.clone(), dim, index, src, alpha)

    if use_contiguous_suffix_path:
        out = inp.clone()
        return _run_contiguous_suffix_path(out, dim, index, src, alpha)

    if not index_validated:
        _assert_index_in_bounds_for_native_fallback(index, inp_len)

    try:
        return _native_index_add(inp, dim, index, src, alpha)
    except NotImplementedError:
        pass

    # Move target dim to last position for coalesced memory access
    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    # Clone input for output
    out = inp.clone()

    # Calculate grid with autotune
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](out, index, src, M, N, alpha, inp_len)

    # Restore original dimension order if needed
    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    """
    In-place version of index_add.
    """
    logger.debug("GEMS_MTHREADS INDEX_ADD_")

    dim = _validate_index_add_args(inp, dim, index, src)
    if src is inp or index is inp:
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )
    if src.numel() == 0:
        return inp
    input_src_alias = torch._C._is_alias_of(inp, src)
    input_index_alias = torch._C._is_alias_of(inp, index)
    if input_src_alias or input_index_alias:
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )

    use_bf16_grouped_path = (
        _can_use_bf16_grouped_path(inp, dim, index, src) and not input_src_alias
    )
    use_contiguous_suffix_path = False
    use_bf16_unique_path = False
    use_bf16_all_same_path = False
    if not use_bf16_grouped_path:
        use_contiguous_suffix_path = (
            _can_use_contiguous_suffix_path(inp, dim, index, src)
            and not input_src_alias
        )
        use_bf16_unique_path = (
            _can_use_bf16_unique_path(inp, dim, index, src) and not input_src_alias
        )
        use_bf16_all_same_path = (
            _can_use_bf16_all_same_path(inp, dim, index, src) and not input_src_alias
        )

    use_native_fallback = inp.is_contiguous()

    # Make index and src contiguous. resolve_neg() is a no-op normally.
    index = _resolve_index_for_kernel(index).contiguous()
    src = src.contiguous()

    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Reject invalid receivers before a pointer kernel can observe them.
    # Use min/max to avoid allocating full-size boolean tensors.
    unique_index = False
    all_same_checked = False
    index_validated = False
    if use_bf16_grouped_path:
        out, handled = _try_run_bf16_receiver_owned_path(inp, dim, index, src, alpha)
        index_validated = True
        if handled:
            return out
    if not index_validated:
        if use_bf16_all_same_path:
            all_same_checked = True
            if _validate_and_detect_all_same(index, inp_len):
                return _run_bf16_all_same_path(inp, dim, index, src, alpha)
            index_validated = True
        if use_bf16_unique_path:
            unique_index = _validate_and_detect_unique(index, inp_len)
            index_validated = True
        elif not all_same_checked and use_contiguous_suffix_path:
            _assert_index_in_bounds(index, inp_len)
            index_validated = True

    if unique_index:
        return _run_bf16_unique_path(inp, dim, index, src, alpha)

    if use_contiguous_suffix_path:
        return _run_contiguous_suffix_path(inp, dim, index, src, alpha)

    if not index_validated:
        _assert_index_in_bounds_for_native_fallback(index, inp_len)

    if use_native_fallback:
        try:
            return _native_index_add_(inp, dim, index, src, alpha)
        except NotImplementedError:
            pass

    # Move target dim to last position
    final_dim = inp.ndim - 1

    if dim != final_dim:
        # Need to work on a permuted copy
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_work, index, src_work, M, N, alpha, inp_len)

        # Restore original dimension order and copy back
        order = list(range(inp_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = inp_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        # Can work directly on input if already contiguous
        inp_contig = inp.contiguous()

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_contig, index, src, M, N, alpha, inp_len)

        # Copy back if input wasn't contiguous
        if not inp.is_contiguous():
            inp.copy_(inp_contig)

    return inp
