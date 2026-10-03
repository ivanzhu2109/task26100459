"""GLB 容器解析：读取头部、JSON 块与 BIN 块。

约束（按业务规则，不做通用格式校验）：
- 文件不超过 500 MiB，JSON 块不超过 1 MiB；
- 恰好一个 JSON 块跟一个 BIN 块，无外部资源。
BIN 块通过 mmap 映射为 memoryview，不把数据拷进内存。
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass

MAX_FILE_BYTES = 500 * 1024 * 1024
MAX_JSON_BYTES = 1 * 1024 * 1024

_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN = 0x004E4942  # "BIN\0"
_HEADER = struct.Struct("<III")
_CHUNK_HEADER = struct.Struct("<II")


class GlbFormatError(ValueError):
    """输入不满足本库假定的 GLB 结构。"""


@dataclass(frozen=True)
class GlbFile:
    """解析后的 GLB 视图。持有文件对象以维持 mmap 生命周期。"""

    document: dict
    bin_chunk: memoryview
    _fileobj: object

    def close(self) -> None:
        self.bin_chunk.release()
        self._fileobj.close()

    def __enter__(self) -> "GlbFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def parse_glb(path: str) -> GlbFile:
    """解析 GLB 文件，返回 JSON 文档与 BIN 块的内存映射视图。"""
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise GlbFormatError(f"文件超过 500 MiB 上限: {size} 字节")
    if size < _HEADER.size + _CHUNK_HEADER.size:
        raise GlbFormatError("文件太小，不是合法的 GLB")

    fileobj = open(path, "rb")
    try:
        mapped = mmap.mmap(fileobj.fileno(), 0, access=mmap.ACCESS_READ)
    except Exception:
        fileobj.close()
        raise

    try:
        magic, version, total_length = _HEADER.unpack_from(mapped, 0)
        if magic != _GLB_MAGIC:
            raise GlbFormatError("缺少 glTF 魔数")
        if version != _GLB_VERSION:
            raise GlbFormatError(f"不支持的 glTF 版本: {version}")
        if total_length != size:
            raise GlbFormatError("头部声明长度与文件大小不一致")

        offset = _HEADER.size
        json_length, json_type = _CHUNK_HEADER.unpack_from(mapped, offset)
        offset += _CHUNK_HEADER.size
        if json_type != _CHUNK_JSON:
            raise GlbFormatError("第一个块不是 JSON 块")
        if json_length > MAX_JSON_BYTES:
            raise GlbFormatError(f"JSON 块超过 1 MiB 上限: {json_length} 字节")
        json_end = offset + json_length
        if json_end + _CHUNK_HEADER.size > size:
            raise GlbFormatError("JSON 块越界")
        document = json.loads(mapped[offset:json_end].decode("utf-8"))

        bin_length, bin_type = _CHUNK_HEADER.unpack_from(mapped, json_end)
        if bin_type != _CHUNK_BIN:
            raise GlbFormatError("第二个块不是 BIN 块")
        bin_start = json_end + _CHUNK_HEADER.size
        bin_end = bin_start + bin_length
        if bin_end > size:
            raise GlbFormatError("BIN 块越界")

        bin_chunk = memoryview(mapped)[bin_start:bin_end]
        return GlbFile(document=document, bin_chunk=bin_chunk, _fileobj=fileobj)
    except Exception:
        mapped.close()
        fileobj.close()
        raise
