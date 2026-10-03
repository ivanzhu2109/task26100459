# glb-facecount

GLB（glTF 2.0 二进制）模型三角面数检查库，供资产导入流程直接调用。
输入本地 GLB 文件，结果直接返回内存，无接口层、无持久化。

## 用法

```python
from glb_facecount import count_faces

result = count_faces("model.glb")
for mesh in result.meshes:
    print(mesh.name, mesh.face_count)
print("total:", result.total_face_count)
```

## 统计规则

- 每个 mesh 的每个 primitive 统计一次；不按场景节点实例重复计数，未引用的 mesh 也计入。
- 支持 mode 4（TRIANGLES，每 3 项成组）、5（TRIANGLE_STRIP，连续三项）、
  6（TRIANGLE_FAN，首项与相邻对成面）；其他 mode 计 0 面。
- 候选面中任意两个顶点索引相同则不计入；坐标重合/共线不影响。
- 无 indices 时逻辑索引为 0..POSITION.count-1；支持 8/16/32 位无符号索引、
  bufferView 与 accessor 各自偏移、小端字节序，BIN 填充字节不计入。
- 支持 sparse 覆盖：有基础 bufferView 时先取基础值再替换；无基础数据时基础项全 0；
  覆盖是替换而非追加，且只影响所属 accessor。

## 性能（本机实测，pytest -s 记录）

| 场景 | 耗时 | 峰值额外内存 |
| --- | --- | --- |
| 10 亿项隐式零索引扇形（2 项覆盖） | <0.001 s | ~0 MiB |
| 10 万项 sparse 覆盖（1000 万索引扇形） | ~0.07 s | ~18 MiB |
| 约 500 MiB 文件（1.3 亿 u32 索引） | ~0.23 s | ~43 MiB |

无基础 bufferView 的稀疏索引按覆盖点解析求解，时间与覆盖数成正比；
有基础数据时分块流式读取，内存与文件大小无关。

## 测试

```bash
python3 -m pytest tests/ -q -s
```
