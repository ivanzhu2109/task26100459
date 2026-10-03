"""功能测试：拓扑、位宽、偏移、稀疏覆盖、汇总规则。"""

from __future__ import annotations

import struct

import pytest

from glb_face_count import count_glb_faces
from tests.glb_builder import GlbBuilder

ACCEPT_SEQ = [0, 1, 1, 2, 3]


def _single_mesh_report(tmp_path, builder: GlbBuilder):
    path = str(tmp_path / "model.glb")
    builder.build(path)
    return count_glb_faces(path)


def test_strip_acceptance_sequence(tmp_path):
    b = GlbBuilder()
    idx = b.add_index_accessor(ACCEPT_SEQ, 5123)
    b.add_mesh([{"indices": idx, "mode": 5}])
    report = _single_mesh_report(tmp_path, b)
    assert report.meshes[0].primitive_faces == (1,)
    assert report.total_faces == 1


def test_fan_acceptance_sequence(tmp_path):
    b = GlbBuilder()
    idx = b.add_index_accessor(ACCEPT_SEQ, 5123)
    b.add_mesh([{"indices": idx, "mode": 6}])
    report = _single_mesh_report(tmp_path, b)
    assert report.meshes[0].primitive_faces == (2,)


def test_triangles_grouping_and_degenerate(tmp_path):
    b = GlbBuilder()
    # (0,1,2) 有效；(3,4,4) 退化；尾部 [5] 不足三项忽略
    idx = b.add_index_accessor([0, 1, 2, 3, 4, 4, 5], 5121)
    b.add_mesh([{"indices": idx, "mode": 4}])
    report = _single_mesh_report(tmp_path, b)
    assert report.total_faces == 1


@pytest.mark.parametrize("component_type", [5121, 5123, 5125])
def test_component_widths_agree(tmp_path, component_type):
    b = GlbBuilder()
    idx = b.add_index_accessor([0, 1, 2, 3, 4, 5, 6, 7, 8], component_type)
    b.add_mesh([{"indices": idx, "mode": 5}])
    report = _single_mesh_report(tmp_path, b)
    assert report.total_faces == 7


def test_nonzero_offsets_and_padding_ignored(tmp_path):
    b = GlbBuilder()
    # 手工铺一段 BIN：前导垃圾 + 索引数据 + 尾部填充，
    # accessor 与 bufferView 都带非零偏移。
    b.bin.extend(b"\xaa\xbb\xcc\xdd")
    view_offset = len(b.bin)
    payload = struct.pack("<4H", 0, 1, 2, 3)
    b.bin.extend(payload)
    b.buffer_views.append(
        {"buffer": 0, "byteOffset": view_offset, "byteLength": len(payload)}
    )
    idx = b.add_accessor(
        {
            "bufferView": 0,
            "byteOffset": 2,  # 跳过第一个索引，从 1,2,3 开始
            "componentType": 5123,
            "count": 3,
            "type": "SCALAR",
        }
    )
    b.add_mesh([{"indices": idx, "mode": 5}])
    report = _single_mesh_report(tmp_path, b)
    # 逻辑索引为 1,2,3：恰好 1 面；BIN 尾部对齐零字节不得计入。
    assert report.total_faces == 1


def test_no_indices_uses_position_count(tmp_path):
    b = GlbBuilder()
    pos = b.add_position_accessor(7)
    prim_tri = {"attributes": {"POSITION": pos}, "mode": 4}
    prim_strip = {"attributes": {"POSITION": pos}, "mode": 5}
    prim_fan = {"attributes": {"POSITION": pos}, "mode": 6}
    b.add_mesh([prim_tri, prim_strip, prim_fan])
    report = _single_mesh_report(tmp_path, b)
    assert report.meshes[0].primitive_faces == (2, 5, 5)


def test_other_modes_contribute_zero(tmp_path):
    b = GlbBuilder()
    idx = b.add_index_accessor([0, 1, 2, 3, 4, 5], 5121)
    b.add_mesh([{"indices": idx, "mode": 1}, {"indices": idx, "mode": 0}])
    report = _single_mesh_report(tmp_path, b)
    assert report.total_faces == 0


def test_mesh_sum_equals_total_and_no_instance_double_count(tmp_path):
    b = GlbBuilder()
    idx = b.add_index_accessor([0, 1, 2, 3, 4, 5], 5121)
    b.add_mesh([{"indices": idx, "mode": 4}], name="used")
    b.add_mesh([{"indices": idx, "mode": 4}], name="orphan")
    # 场景把 mesh 0 实例化两次；mesh 1 不被引用。
    b.extra["nodes"] = [{"mesh": 0}, {"mesh": 0}]
    b.extra["scenes"] = [{"nodes": [0, 1]}]
    b.extra["scene"] = 0
    report = _single_mesh_report(tmp_path, b)
    assert [m.faces for m in report.meshes] == [2, 2]
    assert sum(m.faces for m in report.meshes) == report.total_faces == 4
