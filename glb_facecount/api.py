"""对外入口：count_faces。"""
from __future__ import annotations

from .counter import count_primitive_faces
from .model import FaceCountResult, MeshFaceCount
from .reader import open_glb


def count_faces(path: str) -> FaceCountResult:
    """统计 GLB 文件中每个 mesh 的面数与文件总面数。

    每个 mesh 的每个 primitive 统计一次，不按场景节点实例重复计数，
    未被场景引用的 mesh 也计入。
    """
    with open_glb(path) as glb:
        meshes = glb.gltf.get("meshes") or []
        results: list[MeshFaceCount] = []
        total = 0
        for mesh in meshes:
            mesh_count = 0
            for primitive in mesh.get("primitives") or []:
                mesh_count += count_primitive_faces(glb, primitive)
            results.append(MeshFaceCount(name=mesh.get("name"), face_count=mesh_count))
            total += mesh_count
    return FaceCountResult(meshes=tuple(results), total_face_count=total)
