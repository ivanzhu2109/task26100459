"""结果类型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MeshFaceCount:
    """单个 mesh 的面数统计。"""

    mesh_index: int
    name: str | None
    primitive_faces: tuple[int, ...]

    @property
    def faces(self) -> int:
        return sum(self.primitive_faces)


@dataclass(frozen=True)
class FaceCountReport:
    """整个 GLB 文件的面数报告。"""

    meshes: tuple[MeshFaceCount, ...] = field(default_factory=tuple)

    @property
    def total_faces(self) -> int:
        return sum(m.faces for m in self.meshes)
