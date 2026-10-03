"""稀疏覆盖测试：共享基础数据、扇形首项、分组边界、零基、类型与偏移。"""

from __future__ import annotations

import struct

from glb_face_count import count_glb_faces
from tests.glb_builder import GlbBuilder


def _report(tmp_path, builder: GlbBuilder):
    path = str(tmp_path / "sparse.glb")
    builder.build(path)
    return count_glb_faces(path)


def test_shared_base_different_overrides(tmp_path):
    b = GlbBuilder()
    base_view = b.add_buffer_view(struct.pack("<6H", 0, 0, 0, 0, 0, 0))

    def accessor_with_sparse(pairs):
        sparse = b.add_sparse(
            [p for p, _ in pairs], [v for _, v in pairs], 5121, 5123
        )
        return b.add_accessor(
            {
                "bufferView": base_view,
                "componentType": 5123,
                "count": 6,
                "type": "SCALAR",
                "sparse": sparse,
            }
        )

    first = accessor_with_sparse([(0, 1), (1, 2), (2, 3)])   # 覆盖前三项 -> 面0有效
    second = accessor_with_sparse([(3, 4), (4, 5), (5, 6)])  # 覆盖后三项 -> 面1有效
    plain = b.add_accessor(
        {"bufferView": base_view, "componentType": 5123, "count": 6, "type": "SCALAR"}
    )
    b.add_mesh([
        {"indices": first, "mode": 4},
        {"indices": second, "mode": 4},
        {"indices": plain, "mode": 4},
    ])
    report = _report(tmp_path, b)
    # 共享同一段 BIN，但覆盖互不影响：1 + 1 + 0
    assert report.meshes[0].primitive_faces == (1, 1, 0)
    assert report.total_faces == 2


def test_override_changes_fan_first_item(tmp_path):
    b = GlbBuilder()
    base_view = b.add_buffer_view(struct.pack("<5H", 0, 0, 0, 0, 0))

    with_first = b.add_accessor(
        {
            "bufferView": base_view,
            "componentType": 5123,
            "count": 5,
            "type": "SCALAR",
            "sparse": b.add_sparse([0, 2], [1, 2], 5123, 5123),
        }
    )
    without_first = b.add_accessor(
        {
            "bufferView": base_view,
            "componentType": 5123,
            "count": 5,
            "type": "SCALAR",
            "sparse": b.add_sparse([2], [2], 5123, 5123),
        }
    )
    b.add_mesh([
        {"indices": with_first, "mode": 6},
        {"indices": without_first, "mode": 6},
    ])
    report = _report(tmp_path, b)
    # 首项覆盖为 1 后：(1,0,2)、(1,2,0) 有效，(1,0,0) 退化 -> 2；
    # 不覆盖首项则所有面都含两个 0 -> 0。
    assert report.meshes[0].primitive_faces == (2, 0)


def test_override_on_triangle_group_boundary(tmp_path):
    b = GlbBuilder()
    base_view = b.add_buffer_view(struct.pack("<6H", 0, 1, 2, 9, 9, 9))

    break_first = b.add_accessor(
        {
            "bufferView": base_view,
            "componentType": 5123,
            "count": 6,
            "type": "SCALAR",
            # 下标 2 是第一组最后一项：覆盖成 1 使 (0,1,1) 退化
            "sparse": b.add_sparse([2], [1], 5123, 5123),
        }
    )
    fix_second = b.add_accessor(
        {
            "bufferView": base_view,
            "componentType": 5123,
            "count": 6,
            "type": "SCALAR",
            # 下标 3 是第二组首项：与 4、5 一起覆盖成 (5,6,7)
            "sparse": b.add_sparse([3, 4, 5], [5, 6, 7], 5123, 5123),
        }
    )
    b.add_mesh([
        {"indices": break_first, "mode": 4},
        {"indices": fix_second, "mode": 4},
    ])
    report = _report(tmp_path, b)
    # 第一个 accessor：面0退化、面1(9,9,9)退化 -> 0
    # 第二个 accessor：面0(0,1,2)有效、面1(5,6,7)有效 -> 2
    assert report.meshes[0].primitive_faces == (0, 2)


def test_zero_base_sparse_without_buffer_view(tmp_path):
    b = GlbBuilder()
    tri = b.add_accessor(
        {
            "componentType": 5125,
            "count": 6,
            "type": "SCALAR",
            "sparse": b.add_sparse([0, 1, 2], [1, 2, 3], 5121, 5125),
        }
    )
    fan = b.add_accessor(
        {
            "componentType": 5125,
            "count": 5,
            "type": "SCALAR",
            "sparse": b.add_sparse([0, 3], [1, 2], 5123, 5125),
        }
    )
    b.add_mesh([{"indices": tri, "mode": 4}, {"indices": fan, "mode": 6}])
    report = _report(tmp_path, b)
    # 三角形：仅面0 (1,2,3) 有效 -> 1
    # 扇形：面 (1,0,0) 退化、(1,0,2) 有效、(1,2,0) 有效 -> 2
    assert report.meshes[0].primitive_faces == (1, 2)


def test_sparse_offsets_and_index_types(tmp_path):
    b = GlbBuilder()
    # sparse.indices 用 16 位、带前导垃圾与非零 byteOffset；
    # sparse.values 同样带非零 byteOffset。
    b.bin.extend(b"\xde\xad")
    idx_offset = len(b.bin)
    b.bin.extend(struct.pack("<3H", 0, 1, 2))
    b.buffer_views.append({"buffer": 0, "byteOffset": idx_offset, "byteLength": 6})
    b.bin.extend(b"\xbe\xef")
    val_offset = len(b.bin)
    b.bin.extend(struct.pack("<3H", 7, 8, 9))
    b.buffer_views.append({"buffer": 0, "byteOffset": val_offset, "byteLength": 6})
    accessor = b.add_accessor(
        {
            "componentType": 5123,
            "count": 3,
            "type": "SCALAR",
            "sparse": {
                "count": 3,
                "indices": {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5123,
                },
                "values": {"bufferView": 1, "byteOffset": 0},
            },
        }
    )
    b.add_mesh([{"indices": accessor, "mode": 4}])
    report = _report(tmp_path, b)
    # 零基 + 覆盖 (7,8,9) -> 1 面
    assert report.total_faces == 1
