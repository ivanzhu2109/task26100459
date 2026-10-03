"""遍历 mesh/primitive，汇总面数。

计数规则：
- 每个 mesh 的每个 primitive 统计一次，不按节点实例重复计数，
  未被场景引用的 mesh 也计入；
- 仅 mode 4/5/6 贡献面数，其余为 0；
- 无 indices 时逻辑索引为 0..POSITION.count-1，全部互不相同，
  不存在退化面，直接按拓扑公式计算；
- 有 indices 且无基础 bufferView 时走零基解析路径，不展开全部索引。
"""

from __future__ import annotations

from glb_face_count.accessor import get_accessor_count, load_index_accessor
from glb_face_count.glb import parse_glb
from glb_face_count.result import FaceCountReport, MeshFaceCount
from glb_face_count.topology import (
    TRIANGLE_FAN,
    TRIANGLE_STRIP,
    TRIANGLES,
    _face_positions,
    count_faces_stream,
    count_faces_zero_base,
)

_COUNTED_MODES = (TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN)
# 每块 4M 项：uint32 物化约 16 MiB，远低于 96 MiB 额外内存上限。
_CHUNK_ITEMS = 1 << 22


def count_glb_faces(path: str) -> FaceCountReport:
    """检查 GLB 文件并返回面数报告。"""
    with parse_glb(path) as glb:
        document = glb.document
        meshes = document.get("meshes", [])
        results: list[MeshFaceCount] = []
        for mesh_index, mesh in enumerate(meshes):
            primitive_faces = tuple(
                _count_primitive(document, glb.bin_chunk, primitive)
                for primitive in mesh.get("primitives", [])
            )
            results.append(
                MeshFaceCount(
                    mesh_index=mesh_index,
                    name=mesh.get("name"),
                    primitive_faces=primitive_faces,
                )
            )
        return FaceCountReport(meshes=tuple(results))


def _count_primitive(document: dict, bin_chunk: memoryview, primitive: dict) -> int:
    mode = int(primitive.get("mode", TRIANGLES))
    if mode not in _COUNTED_MODES:
        return 0

    if "indices" not in primitive:
        position_accessor = primitive.get("attributes", {}).get("POSITION")
        if position_accessor is None:
            return 0
        count = get_accessor_count(document, position_accessor)
        return _face_positions(mode, count)

    accessor = load_index_accessor(document, bin_chunk, primitive["indices"])
    if accessor.base_span is None:
        overrides = accessor.overrides.as_dict() if accessor.overrides else {}
        return count_faces_zero_base(mode, accessor.count, overrides)
    return count_faces_stream(
        mode, accessor.iter_chunks(_CHUNK_ITEMS), accessor.count
    )
