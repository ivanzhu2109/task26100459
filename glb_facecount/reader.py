"""GLB 容器解析与索引 accessor 元数据读取。

基础索引数据不整体读入内存，只记录其在文件中的位置，
由调用方按需分块读取；sparse 覆盖（规模有界）则直接读为数组。
"""
from __future__ import annotations

import json
import os
import struct
from typing import BinaryIO

import numpy as np

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MiB
MAX_JSON_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB

_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN = 0x004E4942  # "BIN\0"

# glTF 索引 componentType -> (numpy 小端 dtype, 字节宽)
COMPONENT_DTYPES = {
    5121: np.dtype("<u1"),  # UNSIGNED_BYTE
    5123: np.dtype("<u2"),  # UNSIGNED_SHORT
    5125: np.dtype("<u4"),  # UNSIGNED_INT
}


class GlbError(ValueError):
    """GLB 结构不合法时抛出。"""


class GlbFile:
    """已打开的 GLB 文件：gltf JSON 与 BIN 块在文件内的位置。"""

    def __init__(self, fp: BinaryIO, gltf: dict, bin_offset: int, bin_length: int) -> None:
        self.fp = fp
        self.gltf = gltf
        self.bin_offset = bin_offset
        self.bin_length = bin_length

    def read_bin(self, offset: int, size: int) -> bytes:
        """读取 BIN 块内 [offset, offset+size) 的字节。"""
        if offset < 0 or size < 0 or offset + size > self.bin_length:
            raise GlbError("BIN 读取越界")
        self.fp.seek(self.bin_offset + offset)
        data = self.fp.read(size)
        if len(data) != size:
            raise GlbError("BIN 读取不足")
        return data

    def close(self) -> None:
        self.fp.close()

    def __enter__(self) -> "GlbFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_glb(path: str) -> GlbFile:
    """打开 GLB 文件，解析头部与 JSON 块，记录 BIN 块位置（不读入内存）。"""
    file_size = os.path.getsize(path)
    if file_size > MAX_FILE_SIZE:
        raise GlbError(f"文件超过 500 MiB 上限: {file_size}")
    fp = open(path, "rb")
    try:
        header = fp.read(12)
        if len(header) != 12:
            raise GlbError("文件头不完整")
        magic, version, total_length = struct.unpack("<III", header)
        if magic != _GLB_MAGIC:
            raise GlbError("不是 GLB 文件")
        if version != _GLB_VERSION:
            raise GlbError(f"不支持的 glTF 版本: {version}")
        if total_length != file_size:
            raise GlbError("文件长度与头部声明不一致")

        json_chunk = _read_chunk_header(fp)
        if json_chunk is None or json_chunk[1] != _CHUNK_JSON:
            raise GlbError("缺少 JSON 块")
        json_length = json_chunk[0]
        if json_length > MAX_JSON_CHUNK_SIZE:
            raise GlbError(f"JSON 块超过 1 MiB 上限: {json_length}")
        gltf = json.loads(fp.read(json_length))

        bin_chunk = _read_chunk_header(fp)
        if bin_chunk is None or bin_chunk[1] != _CHUNK_BIN:
            raise GlbError("缺少 BIN 块")
        bin_offset = fp.tell()
        return GlbFile(fp, gltf, bin_offset, bin_chunk[0])
    except Exception:
        fp.close()
        raise


def _read_chunk_header(fp: BinaryIO) -> tuple[int, int] | None:
    data = fp.read(8)
    if not data:
        return None
    if len(data) != 8:
        raise GlbError("块头不完整")
    return struct.unpack("<II", data)


class IndexSource:
    """一个索引 accessor 的逻辑索引来源（含 sparse 覆盖）。"""

    def __init__(
        self,
        count: int,
        component_type: int,
        has_base: bool,
        base_file_offset: int,
        sparse_positions: np.ndarray | None,
        sparse_values: np.ndarray | None,
    ) -> None:
        self.count = count
        self.component_type = component_type
        self.has_base = has_base
        self.base_file_offset = base_file_offset  # 绝对文件偏移
        self.sparse_positions = sparse_positions  # int64 升序数组或 None
        self.sparse_values = sparse_values  # 无符号整型数组或 None

    @property
    def dtype(self) -> np.dtype:
        return COMPONENT_DTYPES[self.component_type]


def read_index_source(glb: GlbFile, accessor_index: int) -> IndexSource:
    """按 accessor 定义构造索引来源，sparse 覆盖直接读为数组。"""
    accessors = glb.gltf.get("accessors") or []
    try:
        accessor = accessors[accessor_index]
    except IndexError:
        raise GlbError(f"accessor 下标越界: {accessor_index}") from None

    component_type = accessor.get("componentType")
    if component_type not in COMPONENT_DTYPES:
        raise GlbError(f"索引 accessor 的 componentType 非法: {component_type}")
    count = int(accessor.get("count", 0))

    has_base = "bufferView" in accessor
    base_file_offset = 0
    if has_base:
        base_file_offset = glb.bin_offset + _buffer_view_offset(
            glb, accessor["bufferView"]
        ) + int(accessor.get("byteOffset", 0))

    sparse_positions = None
    sparse_values = None
    sparse = accessor.get("sparse")
    if sparse is not None:
        sparse_count = int(sparse["count"])
        indices_def = sparse["indices"]
        values_def = sparse["values"]
        pos_dtype = COMPONENT_DTYPES.get(indices_def.get("componentType"))
        if pos_dtype is None:
            raise GlbError("sparse.indices 的 componentType 非法")
        pos_offset = glb.bin_offset + _buffer_view_offset(
            glb, indices_def["bufferView"]
        ) + int(indices_def.get("byteOffset", 0))
        val_offset = glb.bin_offset + _buffer_view_offset(
            glb, values_def["bufferView"]
        ) + int(values_def.get("byteOffset", 0))
        sparse_positions = np.frombuffer(
            glb.read_bin(pos_offset - glb.bin_offset, sparse_count * pos_dtype.itemsize),
            dtype=pos_dtype,
        ).astype(np.int64)
        sparse_values = np.frombuffer(
            glb.read_bin(val_offset - glb.bin_offset, sparse_count * COMPONENT_DTYPES[component_type].itemsize),
            dtype=COMPONENT_DTYPES[component_type],
        )
        if sparse_count > 1 and not np.all(sparse_positions[:-1] <= sparse_positions[1:]):
            order = np.argsort(sparse_positions, kind="stable")
            sparse_positions = sparse_positions[order]
            sparse_values = sparse_values[order]

    return IndexSource(
        count=count,
        component_type=component_type,
        has_base=has_base,
        base_file_offset=base_file_offset,
        sparse_positions=sparse_positions,
        sparse_values=sparse_values,
    )


def _buffer_view_offset(glb: GlbFile, buffer_view_index: int) -> int:
    buffer_views = glb.gltf.get("bufferViews") or []
    try:
        buffer_view = buffer_views[buffer_view_index]
    except IndexError:
        raise GlbError(f"bufferView 下标越界: {buffer_view_index}") from None
    return int(buffer_view.get("byteOffset", 0))
