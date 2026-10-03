"""测试用 GLB 构造器：手工拼装 JSON 块与 BIN 块。"""

from __future__ import annotations

import json
import struct


class GlbBuilder:
    def __init__(self) -> None:
        self.bin = bytearray()
        self.buffer_views: list[dict] = []
        self.accessors: list[dict] = []
        self.meshes: list[dict] = []
        self.extra: dict = {}

    def _align(self, multiple: int = 4) -> int:
        """补齐 BIN 到对齐边界，返回填充前的偏移。"""
        pad = (-len(self.bin)) % multiple
        self.bin.extend(b"\x00" * pad)
        return len(self.bin)

    def add_buffer_view(self, data: bytes, align: int = 4) -> int:
        offset = self._align(align)
        self.bin.extend(data)
        self.buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        return len(self.buffer_views) - 1

    def add_accessor(self, accessor: dict) -> int:
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_index_accessor(
        self,
        values: list[int],
        component_type: int,
        byte_offset: int = 0,
        sparse: dict | None = None,
    ) -> int:
        fmt = {5121: "B", 5123: "H", 5125: "I"}[component_type]
        data = struct.pack(f"<{len(values)}{fmt}", *values)
        view = self.add_buffer_view(data)
        accessor = {
            "bufferView": view,
            "byteOffset": byte_offset,
            "componentType": component_type,
            "count": len(values),
            "type": "SCALAR",
        }
        if sparse is not None:
            accessor["sparse"] = sparse
        return self.add_accessor(accessor)

    def add_sparse(
        self,
        positions: list[int],
        values: list[int],
        index_component_type: int,
        value_component_type: int,
    ) -> dict:
        idx_fmt = {5121: "B", 5123: "H", 5125: "I"}[index_component_type]
        val_fmt = {5121: "B", 5123: "H", 5125: "I"}[value_component_type]
        idx_view = self.add_buffer_view(
            struct.pack(f"<{len(positions)}{idx_fmt}", *positions)
        )
        val_view = self.add_buffer_view(
            struct.pack(f"<{len(values)}{val_fmt}", *values)
        )
        return {
            "count": len(positions),
            "indices": {"bufferView": idx_view, "componentType": index_component_type},
            "values": {"bufferView": val_view},
        }

    def add_position_accessor(self, count: int) -> int:
        return self.add_accessor(
            {"componentType": 5126, "count": count, "type": "VEC3"}
        )

    def add_mesh(self, primitives: list[dict], name: str | None = None) -> None:
        mesh: dict = {"primitives": primitives}
        if name is not None:
            mesh["name"] = name
        self.meshes.append(mesh)

    def build(self, path: str) -> None:
        document = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": len(self.bin)}],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
            "meshes": self.meshes,
        }
        document.update(self.extra)
        json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_bytes += b" " * ((-len(json_bytes)) % 4)
        bin_bytes = bytes(self.bin) + b"\x00" * ((-len(self.bin)) % 4)
        total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, total))
            fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
            fh.write(json_bytes)
            fh.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
            fh.write(bin_bytes)
