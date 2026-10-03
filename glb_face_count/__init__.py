"""GLB 三角面数检查库。

输入本地 glTF 2.0 二进制 GLB 文件，返回每个 mesh 的面数与文件总面数。
结果仅存在内存中，不做持久化。
"""

from glb_face_count.counter import count_glb_faces
from glb_face_count.result import FaceCountReport, MeshFaceCount

__all__ = ["count_glb_faces", "FaceCountReport", "MeshFaceCount"]
