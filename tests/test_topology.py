"""topology 单元测试：分块跨界与暴力参考实现对比。"""

from __future__ import annotations

import random

import numpy as np

from glb_face_count.topology import (
    TRIANGLE_FAN,
    TRIANGLE_STRIP,
    TRIANGLES,
    count_faces_stream,
    count_faces_zero_base,
)


def brute_force(mode: int, seq: list[int]) -> int:
    faces = []
    if mode == TRIANGLES:
        faces = [tuple(seq[i:i + 3]) for i in range(0, len(seq) - 2, 3)]
    elif mode == TRIANGLE_STRIP:
        faces = [(seq[i], seq[i + 1], seq[i + 2]) for i in range(len(seq) - 2)]
    elif mode == TRIANGLE_FAN:
        faces = [(seq[0], seq[i], seq[i + 1]) for i in range(1, len(seq) - 1)]
    return sum(1 for a, b, c in faces if a != b and b != c and a != c)


def _stream(mode: int, seq: list[int], chunk: int) -> int:
    arr = np.array(seq, dtype=np.uint32)
    chunks = ((i, arr[i:i + chunk]) for i in range(0, len(seq), chunk))
    return count_faces_stream(mode, chunks, len(seq))


def test_stream_matches_brute_force_all_chunk_sizes():
    rng = random.Random(42)
    for mode in (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN):
        for _ in range(30):
            seq = [rng.randrange(6) for _ in range(rng.randrange(0, 40))]
            expected = brute_force(mode, seq)
            for chunk in (1, 2, 3, 5, 7, 100):
                assert _stream(mode, seq, chunk) == expected, (
                    mode, seq, chunk)


def test_zero_base_matches_brute_force():
    rng = random.Random(7)
    for mode in (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN):
        for _ in range(30):
            count = rng.randrange(0, 60)
            positions = rng.sample(range(count), min(count, rng.randrange(0, 8)))
            overrides = {p: rng.randrange(4) for p in positions}
            seq = [overrides.get(i, 0) for i in range(count)]
            assert count_faces_zero_base(mode, count, overrides) == brute_force(mode, seq)


def test_zero_base_out_of_range_override_ignored():
    assert count_faces_zero_base(TRIANGLES, 3, {5: 1}) == 0
    assert count_faces_zero_base(TRIANGLE_FAN, 4, {0: 1, 2: 2}) == 2
