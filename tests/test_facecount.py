"""面数统计功能测试。"""
from __future__ import annotations

import struct

import pytest

from glb_facecount import count_faces
from tests.glb_builder import build_glb, make_gltf_with_indices, write_glb


def _run(tmp_path, gltf, bin_data):
    path = str(tmp_path / "m.glb")
    write_glb(path, gltf, bin_data)
    return count_faces(path)


def test_triangles_basic_and_degenerate(tmp_path):
    # 组 (0,1,2) 有效；(0,1,1) 退化；(3,4,5) 有效；尾部不足 3 项忽略
    gltf, bin_data = make_gltf_with_indices([0, 1, 2, 0, 1, 1, 3, 4, 5, 6], mode=4)
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 2
    assert result.total_face_count == 2


def test_strip_sequence_acceptance(tmp_path):
    # 0,1,1,2,3 -> 窗口 (0,1,1)x (1,1,2)x (1,2,3)v -> 1 面
    gltf, bin_data = make_gltf_with_indices([0, 1, 1, 2, 3], mode=5)
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 1


def test_fan_sequence_acceptance(tmp_path):
    # 0,1,1,2,3 -> (0,1,1)x (0,1,2)v (0,2,3)v -> 2 面
    gltf, bin_data = make_gltf_with_indices([0, 1, 1, 2, 3], mode=6)
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 2


@pytest.mark.parametrize("component_type", [5121, 5123, 5125])
def test_index_bit_widths(tmp_path, component_type):
    gltf, bin_data = make_gltf_with_indices(
        [0, 1, 2, 3, 4, 5], mode=4, component_type=component_type
    )
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 2


def test_u32_large_values(tmp_path):
    # 超过 16 位的索引值必须被正确读出
    gltf, bin_data = make_gltf_with_indices([0, 70000, 2], mode=4, component_type=5125)
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 1


def test_nonzero_offsets_and_padding(tmp_path):
    # bufferView 前有 4 字节垃圾，accessor 内再有 4 字节偏移，尾部有填充
    indices = [0, 1, 2]
    index_bytes = struct.pack("<3H", *indices)
    prefix = b"\xde\xad\xbe\xef"
    inner_pad = b"\xaa\xbb\xcc\xdd"
    suffix = b"\xee\xee\xee\xee"  # 填充字节不得被算作索引
    bin_data = prefix + inner_pad + index_bytes + suffix
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": len(prefix), "byteLength": len(inner_pad) + len(index_bytes) + len(suffix)}
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": len(inner_pad),
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            }
        ],
        "meshes": [{"primitives": [{"indices": 0, "mode": 4}]}],
    }
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 1


def test_no_indices_uses_position_count(tmp_path):
    position_count = 7
    bin_data = b"\x00" * (position_count * 12)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(bin_data)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": position_count, "type": "VEC3"}
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 5}]}],
    }
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == position_count - 2


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_non_triangle_modes_contribute_zero(tmp_path, mode):
    gltf, bin_data = make_gltf_with_indices([0, 1, 2, 3, 4, 5], mode=mode)
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 0
    assert result.total_face_count == 0


def test_meshes_sum_equals_total_and_no_node_double_count(tmp_path):
    gltf, bin_data = make_gltf_with_indices([0, 1, 2], mode=4)
    # 第二个 mesh 未被场景引用，仍须计入；第一个 mesh 被 3 个节点实例化，只计一次
    gltf["meshes"].append(
        {"name": "orphan", "primitives": [{"indices": 0, "mode": 6}]}
    )
    gltf["nodes"] = [{"mesh": 0}, {"mesh": 0}, {"mesh": 0}]
    gltf["scenes"] = [{"nodes": [0, 1, 2]}]
    gltf["scene"] = 0
    result = _run(tmp_path, gltf, bin_data)
    assert [m.face_count for m in result.meshes] == [1, 1]
    assert result.total_face_count == sum(m.face_count for m in result.meshes)


def _sparse_gltf(count, mode, sparse_pos, sparse_vals, base_indices=None, component_type=5123):
    """构造带 sparse 索引 accessor 的单 primitive gltf。"""
    fmt = {5121: "<%dB", 5123: "<%dH", 5125: "<%dI"}[component_type]
    parts = []
    views = []
    offset = 0

    def add(data):
        nonlocal offset
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        parts.append(data)
        offset += len(data)
        return len(views) - 1

    accessor = {"componentType": component_type, "count": count, "type": "SCALAR"}
    if base_indices is not None:
        accessor["bufferView"] = add(struct.pack(fmt % len(base_indices), *base_indices))
    pos_bytes = struct.pack("<%dI" % len(sparse_pos), *sparse_pos)
    val_bytes = struct.pack(fmt % len(sparse_vals), *sparse_vals)
    accessor["sparse"] = {
        "count": len(sparse_pos),
        "indices": {"bufferView": add(pos_bytes), "componentType": 5125},
        "values": {"bufferView": add(val_bytes)},
    }
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": offset}],
        "bufferViews": views,
        "accessors": [accessor],
        "meshes": [{"primitives": [{"indices": 0, "mode": mode}]}],
    }
    return gltf, b"".join(parts)


def test_sparse_shared_base_different_overrides(tmp_path):
    # 两个 accessor 共享同一段基础数据，覆盖不同，互不影响
    base = [0, 0, 0, 0, 0, 0]
    gltf_a, bin_a = _sparse_gltf(6, 4, [0, 1], [1, 2], base_indices=base)
    # 在同一文件里放两个共享基础 bufferView 的 accessor
    base_bytes = struct.pack("<6H", *base)
    pos_a = struct.pack("<2I", 0, 1)
    val_a = struct.pack("<2H", 1, 2)
    pos_b = struct.pack("<2I", 3, 4)
    val_b = struct.pack("<2H", 1, 2)
    bin_data = base_bytes + pos_a + val_a + pos_b + val_b
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(base_bytes)},
        {"buffer": 0, "byteOffset": 12, "byteLength": len(pos_a)},
        {"buffer": 0, "byteOffset": 20, "byteLength": len(val_a)},
        {"buffer": 0, "byteOffset": 24, "byteLength": len(pos_b)},
        {"buffer": 0, "byteOffset": 32, "byteLength": len(val_b)},
    ]
    accessors = [
        {
            "bufferView": 0, "componentType": 5123, "count": 6, "type": "SCALAR",
            "sparse": {"count": 2, "indices": {"bufferView": 1, "componentType": 5125}, "values": {"bufferView": 2}},
        },
        {
            "bufferView": 0, "componentType": 5123, "count": 6, "type": "SCALAR",
            "sparse": {"count": 2, "indices": {"bufferView": 3, "componentType": 5125}, "values": {"bufferView": 4}},
        },
    ]
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [
            {"primitives": [{"indices": 0, "mode": 4}]},
            {"primitives": [{"indices": 1, "mode": 4}]},
        ],
    }
    result = _run(tmp_path, gltf, bin_data)
    # A: 组0 (1,2,0) 有效，组1 (0,0,0) 退化 -> 1
    # B: 组0 (0,0,0) 退化，组1 (1,2,0) 有效 -> 1
    assert [m.face_count for m in result.meshes] == [1, 1]
    assert result.total_face_count == 2


def test_sparse_override_changes_fan_first(tmp_path):
    # 无基础 bufferView，覆盖下标 0 改变扇形首项
    # 覆盖: 0->1, 2->2, 3->3；首项为 1 时 (1,0,2)(1,2,3)(1,3,0) 均有效 -> 3
    gltf, bin_data = _sparse_gltf(5, 6, [0, 2, 3], [1, 2, 3])
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 3


def test_sparse_override_at_triangle_group_boundary(tmp_path):
    # 无基础 bufferView，覆盖落在相邻两组的分界下标 2、3
    # 组0 逻辑值 (0,1,2) 有效；组1 (2,1,0) 有效 -> 2
    gltf, bin_data = _sparse_gltf(6, 4, [1, 2, 3, 4], [1, 2, 2, 1])
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 2


def test_sparse_with_base_replacement_not_append(tmp_path):
    # 有基础数据时覆盖是替换：基础 [9,9,9] 被覆盖为 [0,1,2]
    gltf, bin_data = _sparse_gltf(3, 4, [0, 1, 2], [0, 1, 2], base_indices=[9, 9, 9])
    result = _run(tmp_path, gltf, bin_data)
    assert result.meshes[0].face_count == 1


def test_chunk_boundary_large_base(tmp_path):
    # 基础索引超过单块（4M 项），验证跨块窗口逻辑
    from glb_facecount.counter import _BLOCK_ENTRIES

    count = _BLOCK_ENTRIES + 3
    indices = struct.pack("<3H", 0, 1, 2) + b"\x00\x00" * (count - 3)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(indices)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(indices)}],
        "accessors": [
            {"bufferView": 0, "componentType": 5123, "count": count, "type": "SCALAR"}
        ],
        "meshes": [
            {"name": "tri", "primitives": [{"indices": 0, "mode": 4}]},
            {"name": "strip", "primitives": [{"indices": 0, "mode": 5}]},
            {"name": "fan", "primitives": [{"indices": 0, "mode": 6}]},
        ],
    }
    result = _run(tmp_path, gltf, indices)
    # tri: 组0 (0,1,2) 有效，其余组全 0 -> 1
    # strip: (0,1,2)v (1,2,0)v (2,0,0)x 其余全 0 -> 2
    # fan: (0,1,2)v (0,2,0)x 其余全 0 -> 1
    assert [m.face_count for m in result.meshes] == [1, 2, 1]
    assert result.total_face_count == 4
