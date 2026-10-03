"""测试用 GLB 文件构造辅助。"""
from __future__ import annotations

import json
import struct


def _pad4(data: bytes, pad: bytes) -> bytes:
    remainder = len(data) % 4
    if remainder:
        data += pad * (4 - remainder)
    return data


def build_glb(gltf: dict, bin_data: bytes) -> bytes:
    """由 gltf JSON 与 BIN 数据拼装完整 GLB 字节串。"""
    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(bin_data, b"\x00")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = [struct.pack("<III", 0x46546C67, 2, total)]
    out.append(struct.pack("<II", len(json_chunk), 0x4E4F534A))
    out.append(json_chunk)
    out.append(struct.pack("<II", len(bin_chunk), 0x004E4942))
    out.append(bin_chunk)
    return b"".join(out)


def write_glb(path: str, gltf: dict, bin_data: bytes) -> None:
    """把 GLB 写入本地文件。"""
    with open(path, "wb") as fp:
        fp.write(build_glb(gltf, bin_data))


def make_gltf_with_indices(
    indices: list[int],
    mode: int,
    component_type: int = 5123,
    byte_offset: int = 0,
    bin_prefix: bytes = b"",
    bin_suffix: bytes = b"",
) -> tuple[dict, bytes]:
    """构造含单个 mesh/primitive 的 gltf 与 BIN，索引按指定位宽打包。

    bin_prefix/bin_suffix 用于制造非零偏移与填充字节。
    """
    fmt = {5121: "<%dB", 5123: "<%dH", 5125: "<%dI"}[component_type]
    index_bytes = struct.pack(fmt % len(indices), *indices)
    buffer_view_offset = len(bin_prefix)
    bin_data = bin_prefix + index_bytes + bin_suffix
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": buffer_view_offset,
                "byteLength": len(index_bytes),
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": byte_offset,
                "componentType": component_type,
                "count": len(indices),
                "type": "SCALAR",
            }
        ],
        "meshes": [
            {
                "name": "m0",
                "primitives": [{"indices": 0, "mode": mode}],
            }
        ],
    }
    return gltf, bin_data
