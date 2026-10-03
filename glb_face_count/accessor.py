"""索引 accessor 的读取。

- componentType 支持 5121/5123/5125（无符号 8/16/32 位），小端；
- 逻辑索引 = accessor.byteOffset + bufferView.byteOffset 起的 count 项，
  严格按 bufferView.byteLength 截断，不触碰 BIN 的对齐填充字节；
- sparse 覆盖：先取基础 bufferView 的值（没有则基础项全为 0），
  再按 sparse.indices 给出的逻辑下标替换为 sparse.values 的值。
  覆盖只作用于所属 accessor，即使多个 accessor 共享同一段 BIN。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from glb_face_count.glb import GlbFormatError

_COMPONENT_DTYPES = {
    5121: np.dtype(np.uint8),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
}


@dataclass(frozen=True)
class SparseOverrides:
    """稀疏覆盖：positions[i] 处的逻辑索引被替换为 values[i]。"""

    positions: np.ndarray  # uint64，升序
    values: np.ndarray  # uint32

    def as_dict(self) -> dict[int, int]:
        return dict(zip(self.positions.tolist(), self.values.tolist()))


def _buffer_view_span(document: dict, buffer_view_index: int) -> tuple[int, int]:
    views = document.get("bufferViews", [])
    try:
        view = views[buffer_view_index]
    except IndexError:
        raise GlbFormatError(f"bufferView 下标越界: {buffer_view_index}")
    start = int(view.get("byteOffset", 0))
    length = int(view["byteLength"])
    return start, length


def _read_region(
    bin_chunk: memoryview, start: int, length: int, dtype: np.dtype
) -> np.ndarray:
    if length % dtype.itemsize != 0:
        raise GlbFormatError("数据区长度不是元素大小的整数倍")
    if start < 0 or start + length > bin_chunk.nbytes:
        raise GlbFormatError("数据区超出 BIN 块范围")
    return np.frombuffer(bin_chunk[start : start + length], dtype=dtype)


def get_accessor_count(document: dict, accessor_index: int) -> int:
    accessors = document.get("accessors", [])
    try:
        return int(accessors[accessor_index]["count"])
    except IndexError:
        raise GlbFormatError(f"accessor 下标越界: {accessor_index}")


def load_index_accessor(
    document: dict, bin_chunk: memoryview, accessor_index: int
) -> "IndexAccessor":
    accessors = document.get("accessors", [])
    try:
        accessor = accessors[accessor_index]
    except IndexError:
        raise GlbFormatError(f"accessor 下标越界: {accessor_index}")

    component_type = accessor.get("componentType")
    dtype = _COMPONENT_DTYPES.get(component_type)
    if dtype is None:
        raise GlbFormatError(f"索引 componentType 不支持: {component_type}")
    count = int(accessor["count"])
    accessor_offset = int(accessor.get("byteOffset", 0))

    base_span: tuple[int, int] | None = None
    if "bufferView" in accessor:
        view_start, view_length = _buffer_view_span(document, accessor["bufferView"])
        start = view_start + accessor_offset
        available = view_start + view_length - start
        needed = count * dtype.itemsize
        if available < needed:
            raise GlbFormatError("索引数据超出 bufferView 范围")
        base_span = (start, needed)

    overrides: SparseOverrides | None = None
    sparse = accessor.get("sparse")
    if sparse is not None:
        overrides = _load_sparse(document, bin_chunk, sparse, dtype)

    return IndexAccessor(
        count=count, dtype=dtype, base_span=base_span, overrides=overrides,
        bin_chunk=bin_chunk,
    )


def _load_sparse(
    document: dict, bin_chunk: memoryview, sparse: dict, value_dtype: np.dtype
) -> SparseOverrides:
    sparse_count = int(sparse["count"])

    idx = sparse["indices"]
    idx_dtype = _COMPONENT_DTYPES.get(idx.get("componentType"))
    if idx_dtype is None:
        raise GlbFormatError(f"sparse.indices componentType 不支持: {idx.get('componentType')}")
    idx_start, idx_len = _buffer_view_span(document, idx["bufferView"])
    idx_start += int(idx.get("byteOffset", 0))
    idx_need = sparse_count * idx_dtype.itemsize
    if idx_len < idx_need:
        raise GlbFormatError("sparse.indices 数据不足")
    positions = _read_region(bin_chunk, idx_start, idx_need, idx_dtype)

    val = sparse["values"]
    val_start, val_len = _buffer_view_span(document, val["bufferView"])
    val_start += int(val.get("byteOffset", 0))
    val_need = sparse_count * value_dtype.itemsize
    if val_len < val_need:
        raise GlbFormatError("sparse.values 数据不足")
    values = _read_region(bin_chunk, val_start, val_need, value_dtype)

    positions64 = positions.astype(np.uint64)
    order = np.argsort(positions64, kind="stable")
    return SparseOverrides(
        positions=positions64[order],
        values=values.astype(np.uint32)[order],
    )


@dataclass(frozen=True)
class IndexAccessor:
    """一个索引 accessor 的读取计划。"""

    count: int
    dtype: np.dtype
    base_span: tuple[int, int] | None
    overrides: SparseOverrides | None
    bin_chunk: memoryview

    def iter_chunks(self, chunk_items: int) -> Iterator[tuple[int, np.ndarray]]:
        """按块产出 (起始逻辑下标, uint32 索引数组)，已应用稀疏覆盖。"""
        itemsize = self.dtype.itemsize
        overrides = self.overrides
        pos = 0
        while pos < self.count:
            end = min(pos + chunk_items, self.count)
            if self.base_span is not None:
                base_start, _ = self.base_span
                byte_start = base_start + pos * itemsize
                byte_end = base_start + end * itemsize
                chunk = np.frombuffer(
                    self.bin_chunk[byte_start:byte_end], dtype=self.dtype
                ).astype(np.uint32)
            else:
                chunk = np.zeros(end - pos, dtype=np.uint32)
            if overrides is not None:
                lo = np.searchsorted(overrides.positions, pos)
                hi = np.searchsorted(overrides.positions, end)
                if hi > lo:
                    chunk[overrides.positions[lo:hi] - pos] = overrides.values[lo:hi]
            yield pos, chunk
            pos = end
