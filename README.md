# glb-face-count

GLB（glTF 2.0 二进制）模型三角面数检查库。输入本地 GLB 文件路径，
结果以内存对象返回，无接口层、不做持久化，供资产导入流程直接调用。

## 用法

```python
from glb_face_count import count_glb_faces

report = count_glb_faces("model.glb")
report.total_faces            # 文件总面数
for mesh in report.meshes:    # 每个 mesh 统计一次，不按节点实例重复计数
    mesh.mesh_index, mesh.name, mesh.faces, mesh.primitive_faces
```

## 计数规则

- 统计 mode 4/5/6（TRIANGLES / TRIANGLE_STRIP / TRIANGLE_FAN），其余为 0；
- 候选面中任意两个顶点索引相同即退化，不计入；坐标重合、共线不影响；
- 无 `indices` 时逻辑索引为 `0..POSITION.count-1`；
- 索引支持无符号 8/16/32 位、小端、accessor 与 bufferView 各自偏移，
  BIN 对齐填充字节不计入；
- 支持 sparse 覆盖：先取基础 bufferView（无则基础项全 0），再按下标替换；
  覆盖只作用于所属 accessor，即使共享同一段 BIN。

## 非功能约束

- 文件 ≤ 500 MiB，JSON 块 ≤ 1 MiB，超限抛 `GlbFormatError`；
- BIN 经 mmap 访问、索引分块（每块 4M 项）流式统计，单次检查峰值额外
  内存远低于 96 MiB；
- 索引 accessor 无基础 bufferView 时走解析式路径：只枚举包含至少两个
  被覆盖项的候选面，十亿级逻辑索引毫秒级完成，面数精确。

## 测试

```bash
python3 -m pytest tests -q
```

规模测试（十亿隐式零索引扇形、十万覆盖、近 500 MiB 文件）的耗时与
峰值额外内存记录在 `scale_report.json`。
