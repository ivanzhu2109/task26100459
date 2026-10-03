"""GLB 模型三角面数检查库。"""
from .api import count_faces
from .model import FaceCountResult, MeshFaceCount

__all__ = ["count_faces", "FaceCountResult", "MeshFaceCount"]
