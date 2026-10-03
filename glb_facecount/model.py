"""数据类型定义。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MeshFaceCount:
    """单个 mesh 的面数统计结果。"""

    name: str | None
    face_count: int


@dataclass(frozen=True)
class FaceCountResult:
    """整个 GLB 文件的面数统计结果。"""

    meshes: tuple[MeshFaceCount, ...] = field(default_factory=tuple)
    total_face_count: int = 0
