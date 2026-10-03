"""按拓扑模式统计三角面数。

三条路径：
- 无 indices：逻辑索引为 0..count-1，连续整数必互不相同，直接算术求解。
- 有 indices 但无基础 bufferView：基础项全为 0，只有被 sparse 覆盖
  位置附近的面可能非退化，按覆盖点解析求解，时间与覆盖数成正比。
- 有 indices 且有基础 bufferView：分块流式读取并应用 sparse 覆盖，
  向量化统计，内存占用与文件大小无关。
"""
from __future__ import annotations

import numpy as np

from .reader import GlbFile, IndexSource, read_index_source

MODE_TRIANGLES = 4
MODE_TRIANGLE_STRIP = 5
MODE_TRIANGLE_FAN = 6

_BLOCK_ENTRIES = 1 << 21  # 每块约 2M 个索引，控制峰值内存


def count_primitive_faces(glb: GlbFile, primitive: dict) -> int:
    """统计单个 primitive 的有效三角面数。"""
    mode = int(primitive.get("mode", MODE_TRIANGLES))
    if mode not in (MODE_TRIANGLES, MODE_TRIANGLE_STRIP, MODE_TRIANGLE_FAN):
        return 0

    if "indices" not in primitive:
        position_count = _position_count(glb, primitive)
        return _count_implicit(position_count, mode)

    source = read_index_source(glb, int(primitive["indices"]))
    if source.count < 3:
        return 0
    if source.has_base:
        return _count_from_base(glb, source, mode)
    return _count_sparse_only(source, mode)


def _position_count(glb: GlbFile, primitive: dict) -> int:
    attributes = primitive.get("attributes") or {}
    position = attributes.get("POSITION")
    if position is None:
        return 0
    accessors = glb.gltf.get("accessors") or []
    try:
        return int(accessors[int(position)].get("count", 0))
    except IndexError:
        return 0


def _count_implicit(count: int, mode: int) -> int:
    """无 indices：逻辑索引为连续整数，所有候选面都非退化。"""
    if count < 3:
        return 0
    if mode == MODE_TRIANGLES:
        return count // 3
    return count - 2


def _distinct3(a: int, b: int, c: int) -> bool:
    return a != b and b != c and a != c


def _count_sparse_only(source: IndexSource, mode: int) -> int:
    """无基础 bufferView：基础项全 0，仅覆盖点附近的面可能有效。"""
    overrides: dict[int, int] = {}
    if source.sparse_positions is not None:
        for pos, val in zip(source.sparse_positions, source.sparse_values):
            overrides[int(pos)] = int(val)
    if not overrides:
        return 0
    count = source.count

    if mode == MODE_TRIANGLES:
        groups = {p // 3 for p in overrides}
        return sum(
            1
            for g in groups
            if _distinct3(
                overrides.get(3 * g, 0),
                overrides.get(3 * g + 1, 0),
                overrides.get(3 * g + 2, 0),
            )
        )

    if mode == MODE_TRIANGLE_STRIP:
        starts: set[int] = set()
        for p in overrides:
            for s in (p - 2, p - 1, p):
                if 0 <= s <= count - 3:
                    starts.add(s)
        return sum(
            1
            for s in starts
            if _distinct3(
                overrides.get(s, 0),
                overrides.get(s + 1, 0),
                overrides.get(s + 2, 0),
            )
        )

    # MODE_TRIANGLE_FAN
    first = overrides.get(0, 0)
    starts = set()
    for p in overrides:
        for s in (p - 1, p):
            if 1 <= s <= count - 2:
                starts.add(s)
    return sum(
        1
        for s in starts
        if _distinct3(first, overrides.get(s, 0), overrides.get(s + 1, 0))
    )


def _count_from_base(glb: GlbFile, source: IndexSource, mode: int) -> int:
    """有基础 bufferView：分块读取、应用覆盖、向量化统计。"""
    dtype = source.dtype
    itemsize = dtype.itemsize
    base_bin_offset = source.base_file_offset - glb.bin_offset
    count = source.count
    positions = source.sparse_positions
    values = source.sparse_values

    block = _BLOCK_ENTRIES
    if mode == MODE_TRIANGLES:
        block -= block % 3  # 三角形分组不能跨块

    total = 0
    tail = np.empty(0, dtype=np.uint32)
    fan_first: int | None = None
    start = 0
    while start < count:
        n = min(block, count - start)
        raw = glb.read_bin(base_bin_offset + start * itemsize, n * itemsize)
        arr = np.frombuffer(raw, dtype=dtype).astype(np.uint32)

        if positions is not None and len(positions):
            lo = int(np.searchsorted(positions, start))
            hi = int(np.searchsorted(positions, start + n))
            if hi > lo:
                arr[positions[lo:hi] - start] = values[lo:hi]

        if mode == MODE_TRIANGLES:
            usable = n - (n % 3)
            tri = arr[:usable].reshape(-1, 3)
            total += int(
                np.count_nonzero(
                    (tri[:, 0] != tri[:, 1])
                    & (tri[:, 1] != tri[:, 2])
                    & (tri[:, 0] != tri[:, 2])
                )
            )
            tail = arr[usable:]
        elif mode == MODE_TRIANGLE_STRIP:
            window = np.concatenate([tail, arr]) if len(tail) else arr
            if len(window) >= 3:
                a, b, c = window[:-2], window[1:-1], window[2:]
                total += int(np.count_nonzero((a != b) & (b != c) & (a != c)))
            tail = window[-2:] if len(window) >= 2 else window.copy()
        else:  # MODE_TRIANGLE_FAN
            if fan_first is None:
                fan_first = int(arr[0])
            window = np.concatenate([tail, arr]) if len(tail) else arr
            offset = 1 if start == 0 else 0  # 首块从逻辑下标 1 开始成面
            if len(window) > offset + 1:
                a = window[offset:-1]
                b = window[offset + 1 :]
                total += int(
                    np.count_nonzero(
                        (a != b) & (a != fan_first) & (b != fan_first)
                    )
                )
            tail = window[-1:] if len(window) >= 1 else window.copy()
        start += n
    return total
