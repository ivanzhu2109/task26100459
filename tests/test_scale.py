"""规模与性能测试：耗时与峰值额外内存记录。

运行: python3 -m pytest tests/test_scale.py -q -s
"""
from __future__ import annotations

import json
import os
import resource
import struct
import subprocess
import sys
import time

import numpy as np

from glb_facecount import count_faces
from tests.glb_builder import write_glb


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS 返回字节，Linux 返回 KiB
    return rss if sys.platform == "darwin" else rss * 1024


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child_peak_bytes(code: str) -> int:
    """在子进程中运行代码，返回子进程峰值 RSS（字节）。"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, cwd=_REPO_ROOT, capture_output=True
    )
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _report(name: str, elapsed: float, rss_before: int) -> int:
    extra = _peak_rss_bytes() - rss_before
    print(f"\n[scale] {name}: elapsed={elapsed:.3f}s peak_extra_mem={extra / 2**20:.1f} MiB")
    return extra


def _sparse_only_gltf(count, mode, sparse_pos, sparse_vals):
    pos_bytes = np.asarray(sparse_pos, dtype="<u4").tobytes()
    val_bytes = np.asarray(sparse_vals, dtype="<u4").tobytes()
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes)},
        {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(val_bytes)},
    ]
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(pos_bytes) + len(val_bytes)}],
        "bufferViews": views,
        "accessors": [
            {
                "componentType": 5125,
                "count": count,
                "type": "SCALAR",
                "sparse": {
                    "count": len(sparse_pos),
                    "indices": {"bufferView": 0, "componentType": 5125},
                    "values": {"bufferView": 1},
                },
            }
        ],
        "meshes": [{"primitives": [{"indices": 0, "mode": mode}]}],
    }
    return gltf, pos_bytes + val_bytes


def test_billion_implicit_zero_fan(tmp_path):
    # 10 亿项隐式零索引扇形：仅下标 0->1、500000000->2，应为 2 面
    gltf, bin_data = _sparse_only_gltf(1_000_000_000, 6, [0, 500_000_000], [1, 2])
    path = str(tmp_path / "billion.glb")
    write_glb(path, gltf, bin_data)

    rss_before = _peak_rss_bytes()
    start = time.perf_counter()
    result = count_faces(path)
    elapsed = time.perf_counter() - start
    extra = _report("billion_implicit_zero_fan", elapsed, rss_before)

    assert result.total_face_count == 2
    assert elapsed < 5.0
    assert extra < 96 * 2**20


def test_hundred_thousand_sparse_overrides(tmp_path):
    # 10 万项覆盖，与物化索引的参考实现对比
    count = 10_000_000
    n_sparse = 100_000
    positions = [i * 100 for i in range(n_sparse)]
    values = [(i % 1000) + 1 for i in range(n_sparse)]
    gltf, bin_data = _sparse_only_gltf(count, 6, positions, values)
    path = str(tmp_path / "sparse100k.glb")
    write_glb(path, gltf, bin_data)

    rss_before = _peak_rss_bytes()
    start = time.perf_counter()
    result = count_faces(path)
    elapsed = time.perf_counter() - start
    extra = _report("hundred_thousand_sparse_overrides", elapsed, rss_before)

    # 参考实现：物化全部逻辑索引后向量化统计
    ref = np.zeros(count, dtype=np.uint32)
    ref[np.asarray(positions, dtype=np.int64)] = np.asarray(values, dtype=np.uint32)
    first = ref[0]
    a, b = ref[1:-1], ref[2:]
    expected = int(np.count_nonzero((a != b) & (a != first) & (b != first)))

    assert result.total_face_count == expected
    assert elapsed < 5.0
    assert extra < 96 * 2**20


def test_near_500mib_file(tmp_path):
    # 接近 500 MiB 的文件：u32 索引铺满 BIN，分块流式统计
    bin_len = 500 * 2**20 - 4096
    count = bin_len // 4
    bin_len = count * 4
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": bin_len}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": bin_len}],
        "accessors": [
            {"bufferView": 0, "componentType": 5125, "count": count, "type": "SCALAR"}
        ],
        "meshes": [{"primitives": [{"indices": 0, "mode": 4}]}],
    }
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    path = str(tmp_path / "big.glb")
    total = 12 + 8 + len(json_chunk) + 8 + bin_len
    with open(path, "wb") as fp:
        fp.write(struct.pack("<III", 0x46546C67, 2, total))
        fp.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        fp.write(json_chunk)
        fp.write(struct.pack("<II", bin_len, 0x004E4942))
        fp.write(struct.pack("<3I", 0, 1, 2))  # 首组 (0,1,2) 有效
        remaining = bin_len - 12
        zeros = b"\x00" * (1 << 26)
        while remaining > 0:
            chunk = zeros[: min(remaining, len(zeros))]
            fp.write(chunk)
            remaining -= len(chunk)
    assert os.path.getsize(path) == total

    # 子进程测量：与仅导入的空闲子进程对比，得到检查本身的峰值额外内存
    baseline = _child_peak_bytes("import glb_facecount")
    start = time.perf_counter()
    peak = _child_peak_bytes(
        "from glb_facecount import count_faces;"
        f"r = count_faces({path!r});"
        "assert r.total_face_count == 1, r"
    )
    elapsed = time.perf_counter() - start
    extra = peak - baseline
    print(
        f"\n[scale] near_500mib_file: elapsed={elapsed:.3f}s "
        f"peak_extra_mem={extra / 2**20:.1f} MiB"
    )
    assert extra < 96 * 2**20
