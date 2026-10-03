"""按拓扑模式统计非退化三角面数。

退化判定只看索引值：候选面中任意两个顶点索引相同则不计入；
坐标是否重合、是否共线、朝向如何都不影响。

提供两条路径：
- count_faces_stream：对分块产出的索引流逐项统计（有基础数据时）；
- count_faces_zero_base：基础项全为 0、仅有稀疏覆盖时，不展开
  全部逻辑索引，只枚举“包含至少两个被覆盖项”的候选面解析求值，
  从而支持十亿级逻辑索引在毫秒级完成且内存恒定。
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

TRIANGLES = 4
TRIANGLE_STRIP = 5
TRIANGLE_FAN = 6


def _face_positions(mode: int, count: int) -> int:
    """给定拓扑与索引总数，返回候选面（未做退化判定）的数量。"""
    if mode == TRIANGLES:
        return count // 3
    if mode in (TRIANGLE_STRIP, TRIANGLE_FAN):
        return max(0, count - 2)
    return 0


def count_faces_stream(
    mode: int, chunks: Iterator[tuple[int, np.ndarray]], count: int
) -> int:
    """对 (起始下标, uint32 数组) 块序列统计非退化面数。"""
    if mode not in (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN):
        return 0

    total = 0
    if mode == TRIANGLES:
        tail = np.empty(0, dtype=np.uint32)
        for _, chunk in chunks:
            seq = np.concatenate([tail, chunk])
            whole = (len(seq) // 3) * 3
            triples = seq[:whole].reshape(-1, 3)
            a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
            total += int(np.count_nonzero((a != b) & (b != c) & (a != c)))
            tail = seq[whole:]
        return total

    if mode == TRIANGLE_STRIP:
        tail = np.empty(0, dtype=np.uint32)
        for _, chunk in chunks:
            seq = np.concatenate([tail, chunk])
            if len(seq) >= 3:
                a, b, c = seq[:-2], seq[1:-1], seq[2:]
                total += int(np.count_nonzero((a != b) & (b != c) & (a != c)))
                tail = seq[-2:]
            else:
                tail = seq
        return total

    # TRIANGLE_FAN：每个面都含首项，跨块只需记住首项与上一项。
    first: np.uint32 | None = None
    prev = np.empty(0, dtype=np.uint32)
    for _, chunk in chunks:
        if first is None:
            if len(chunk) == 0:
                continue
            first = chunk[0]
            chunk = chunk[1:]
        seq = np.concatenate([prev, chunk])
        if len(seq) < 2:
            prev = seq
            continue
        # 每个相邻配对 (seq[i], seq[i+1]) 与首项构成一个候选面。
        b, c = seq[:-1], seq[1:]
        total += int(np.count_nonzero((first != b) & (b != c) & (first != c)))
        prev = seq[-1:]
    return total


def count_faces_zero_base(mode: int, count: int, overrides: dict[int, int]) -> int:
    """基础项全为 0 时的解析式统计。

    值非 0 即被覆盖项。一个面若被覆盖项少于两个，则至少两个顶点
    取值为 0，必然退化；因此只需枚举包含被覆盖项的面。
    """
    if mode not in (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN):
        return 0
    n_faces = _face_positions(mode, count)
    if n_faces == 0 or not overrides:
        return 0

    candidate_faces: set[int] = set()
    for pos in overrides:
        if pos < 0 or pos >= count:
            continue
        if mode == TRIANGLES:
            face = pos // 3
            if face < n_faces:
                candidate_faces.add(face)
        elif mode == TRIANGLE_STRIP:
            for face in (pos - 2, pos - 1, pos):
                if 0 <= face < n_faces:
                    candidate_faces.add(face)
        else:  # TRIANGLE_FAN
            if pos == 0:
                # 首项被覆盖时，含首项的面是否有效取决于另外两个位置，
                # 这些面会经由其他被覆盖项枚举到；单独首项覆盖只产生
                # (v0, 0, 0) 形的退化面。
                continue
            for face in (pos - 2, pos - 1):
                if 0 <= face < n_faces:
                    candidate_faces.add(face)

    total = 0
    for face in candidate_faces:
        if mode == TRIANGLES:
            positions = (3 * face, 3 * face + 1, 3 * face + 2)
        elif mode == TRIANGLE_STRIP:
            positions = (face, face + 1, face + 2)
        else:
            positions = (0, face + 1, face + 2)
        a = overrides.get(positions[0], 0)
        b = overrides.get(positions[1], 0)
        c = overrides.get(positions[2], 0)
        if a != b and b != c and a != c:
            total += 1
    return total
