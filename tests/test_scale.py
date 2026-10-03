"""规模测试：十亿级逻辑索引、十万级稀疏覆盖、近 500 MiB 文件。

运行后把耗时与峰值额外内存写入仓库根目录 scale_report.json。
峰值额外内存用 resource.ru_maxrss 在调用前后的差值近似。
"""

from __future__ import annotations

import json
import os
import resource
import struct
import sys
import time

from glb_face_count import count_glb_faces
from tests.glb_builder import GlbBuilder

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "scale_report.json")


# macOS 的 ru_maxrss 单位是字节，Linux 是 KiB。
_RSS_UNIT = 1 if sys.platform == "darwin" else 1024
MAX_PEAK_EXTRA = 96 * 1024 * 1024


def _measure(path):
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    report = count_glb_faces(path)
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_extra_bytes = (after - before) * _RSS_UNIT
    return report, elapsed, peak_extra_bytes


def _record(results: dict):
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as fh:
            results = {**json.load(fh), **results}
    with open(REPORT_PATH, "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


def test_billion_implicit_zero_fan(tmp_path):
    b = GlbBuilder()
    accessor = b.add_accessor(
        {
            "componentType": 5125,
            "count": 1_000_000_000,
            "type": "SCALAR",
            "sparse": b.add_sparse([0, 500_000_000], [1, 2], 5125, 5125),
        }
    )
    b.add_mesh([{"indices": accessor, "mode": 6}])
    path = str(tmp_path / "billion.glb")
    b.build(path)

    report, elapsed, peak_extra = _measure(path)
    assert report.total_faces == 2
    assert elapsed < 5.0, f"耗时 {elapsed:.2f}s 超过 5s 上限"
    assert peak_extra < MAX_PEAK_EXTRA, "峰值额外内存超过 96 MiB"
    _record({"billion_zero_fan": {"faces": report.total_faces,
                                  "seconds": round(elapsed, 4),
                                  "peak_extra_bytes": peak_extra}})
    print(f"\n[billion] faces=2 elapsed={elapsed:.4f}s peak_extra={peak_extra/1e6:.1f}MB")


def test_hundred_thousand_overrides(tmp_path):
    b = GlbBuilder()
    # 99999 个覆盖组成 33333 个完整三角形（值循环 1,2,3，各自有效），
    # 第 100000 个覆盖落在不完整分组里，不产生面。
    positions = list(range(99_999)) + [100_000]
    values = [i % 3 + 1 for i in range(99_999)] + [9]
    accessor = b.add_accessor(
        {
            "componentType": 5125,
            "count": 100_001,
            "type": "SCALAR",
            "sparse": b.add_sparse(positions, values, 5125, 5125),
        }
    )
    b.add_mesh([{"indices": accessor, "mode": 4}])
    path = str(tmp_path / "100k.glb")
    b.build(path)

    report, elapsed, peak_extra = _measure(path)
    assert report.total_faces == 33_333
    assert peak_extra < MAX_PEAK_EXTRA, "峰值额外内存超过 96 MiB"
    _record({"hundred_k_overrides": {"faces": report.total_faces,
                                     "seconds": round(elapsed, 4),
                                     "peak_extra_bytes": peak_extra}})
    print(f"\n[100k] faces=33333 elapsed={elapsed:.4f}s peak_extra={peak_extra/1e6:.1f}MB")


def test_near_500mib_file(tmp_path):
    """近 500 MiB 文件：BIN 几乎全是与索引无关的数据，索引为零基稀疏。"""
    b = GlbBuilder()
    accessor = b.add_accessor(
        {
            "componentType": 5125,
            "count": 1_000_000_000,
            "type": "SCALAR",
            "sparse": b.add_sparse([0, 500_000_000], [1, 2], 5125, 5125),
        }
    )
    b.add_mesh([{"indices": accessor, "mode": 6}])

    path = str(tmp_path / "big.glb")
    # 先用构造器写出小文件，再把 BIN 块扩展到接近 500 MiB。
    b.build(path)
    target_size = 500 * 1024 * 1024 - 1024  # 略低于 500 MiB 上限
    with open(path, "r+b") as fh:
        fh.seek(0)
        header = fh.read(12)
        json_len, _ = struct.unpack("<II", fh.read(8))
        fh.seek(12 + 8 + json_len)
        bin_len, bin_type = struct.unpack("<II", fh.read(8))
        new_bin_len = target_size - (12 + 8 + json_len + 8)
        new_bin_len -= new_bin_len % 4
        fh.seek(12 + 8 + json_len)
        fh.write(struct.pack("<II", new_bin_len, bin_type))
        total = 12 + 8 + json_len + 8 + new_bin_len
        fh.seek(0)
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.truncate(total)
    assert os.path.getsize(path) == total

    report, elapsed, peak_extra = _measure(path)
    assert report.total_faces == 2
    assert elapsed < 5.0, f"耗时 {elapsed:.2f}s 超过 5s 上限"
    assert peak_extra < MAX_PEAK_EXTRA, "峰值额外内存超过 96 MiB"
    _record({"near_500mib": {"file_bytes": total,
                             "faces": report.total_faces,
                             "seconds": round(elapsed, 4),
                             "peak_extra_bytes": peak_extra}})
    print(f"\n[500MiB] size={total/1e6:.0f}MB faces=2 elapsed={elapsed:.4f}s "
          f"peak_extra={peak_extra/1e6:.1f}MB")
