# CryoAgent MCP Server V2 输入字段详细表

本文档是 [MCP_SERVER_INPUT_FORMAT.md](/home/lisongyang/cryoagent/MCP_SERVER_INPUT_FORMAT.md:1) 的补充版本，专门用于和下游 MCP server 逐字段对齐 V2 输入 schema。

建议联调时按下面顺序确认：

1. 顶层字段是否齐全
2. `dataset_info` 的静态数据能否稳定提供
3. `current_state` 的状态字段能否稳定提供
4. `last_node_info` 的结构化抽取是否稳定
5. 空值、布尔值、数值是否保持 JSON 原生类型

## 1. 顶层字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `schema_version` | `string` | 是 | 服务端常量 | 当前固定为 `"2.0"` | `"2.0"` |
| `task_type` | `string` | 是 | 服务端常量 | 当前固定为 `"workflow_decision"` | `"workflow_decision"` |
| `dataset_info` | `object` | 是 | workflow 标签 + EMDB XML + workflow 检索 | 数据集级别背景信息 | `{...}` |
| `current_state` | `object` | 是 | workflow 运行状态 + 日志结构化抽取 | 当前决策锚点 | `{...}` |

## 2. `dataset_info` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `dataset_info.empiar_id` | `string` | 是 | workflow 文件名 / 标签文件 | 统一格式建议 `EMPIAR-xxxxx` | `"EMPIAR-12099"` |
| `dataset_info.emdb_id` | `string \| null` | 否 | EMDB XML / workflow 中参考 map | 统一格式建议 `EMD-xxxxx` | `"EMD-50426"` |
| `dataset_info.resolution` | `array \| null` | 否 | `workflow_label.json` | 标签中的分辨率列表 | `[3.0]` |
| `dataset_info.input_type` | `string \| null` | 否 | `workflow_label.json` | 如 `micrograph` / `particle` | `"particle"` |
| `dataset_info.macromolecules_type` | `string \| null` | 否 | `workflow_label.json` | 如 `protein` / `ribosome` | `"protein"` |
| `dataset_info.num_of_maps` | `integer \| null` | 否 | `workflow_label.json` | map 数量 | `1` |
| `dataset_info.abstract` | `string \| null` | 否 | EMDB XML 摘要 | 紧凑的背景摘要 | `"Primary citation title..."` |
| `dataset_info.known_workflow_steps` | `array[object] \| null` | 否 | 已知 workflow 检索结果 | 已知时提供完整参考流程，未知时传 `null` | `[]` 或 `null` |

### 2.1 `dataset_info.known_workflow_steps[]`

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `dataset_info.known_workflow_steps[].step_index` | `integer` | 是 | workflow 排序结果 | 参考流程中的稳定顺序 | `0` |
| `dataset_info.known_workflow_steps[].node_id` | `string` | 是 | workflow JSON | 逻辑节点 ID | `"J2028"` |
| `dataset_info.known_workflow_steps[].action` | `string` | 是 | workflow JSON | job type | `"import_volumes"` |
| `dataset_info.known_workflow_steps[].title` | `string \| null` | 否 | workflow JSON | 节点标题 | `"EMD-50426"` |
| `dataset_info.known_workflow_steps[].description` | `string \| null` | 否 | workflow JSON | 节点描述 | `""` |
| `dataset_info.known_workflow_steps[].upstream_node_ids` | `array[string]` | 是 | workflow DAG | 上游依赖节点 ID 列表 | `[]` |
| `dataset_info.known_workflow_steps[].parameter_template` | `object` | 是 | workflow JSON | 参考流程该步骤的模板参数 | `{ "volume_blob_path": "..." }` |

说明：

- 已知 workflow 场景下，建议直接传完整流程
- 未知 workflow 场景下，整字段传 `null`
- 后续评测探索能力时，可以人为屏蔽这部分字段

## 3. `current_state` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_id` | `string \| null` | 推荐 | workflow 当前执行状态 | 上一步逻辑节点 ID；未开始可为 `null` | `"J2280"` |
| `current_state.last_action` | `string \| null` | 推荐 | 上一步 job 类型 | 上一步动作类型 | `"patch_ctf_estimation_multi"` |
| `current_state.last_node_status` | `string` | 是 | runtime / 日志 / 服务端状态机 | 推荐枚举：`not_started` / `running` / `completed` / `failed` / `waiting` | `"completed"` |
| `current_state.last_node_info` | `object` | 是 | `Log/<EMPIAR_ID>/jXX.json` 结构化抽取 | 当前最核心状态对象 | `{...}` |

## 4. `last_node_info` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.job_type` | `string \| null` | 推荐 | job 文档 | 上一步 job type | `"import_micrographs"` |
| `current_state.last_node_info.job_uid` | `string \| null` | 推荐 | job 文档 | 上一步实际 job ID | `"J2279"` |
| `current_state.last_node_info.job_title` | `string \| null` | 推荐 | job 文档 | job 标题 | `"New Job J2279"` |
| `current_state.last_node_info.project_uid` | `string \| null` | 推荐 | job 文档 | cryoSPARC project ID | `"P1"` |
| `current_state.last_node_info.status` | `string` | 是 | job 文档 | 当前 job 状态 | `"completed"` |
| `current_state.last_node_info.timestamps` | `object` | 推荐 | job 文档 | 创建、启动、完成时间 | `{...}` |
| `current_state.last_node_info.inputs` | `object` | 推荐 | job 文档 | 输入组结构化摘要 | `{ "groups": [...] }` |
| `current_state.last_node_info.parameters` | `object` | 推荐 | job 文档参数区 | 上一步实际参数 | `{...}` |
| `current_state.last_node_info.outputs` | `object` | 推荐 | job 文档输出区 | 输出组结构化摘要 | `{ "groups": [...] }` |
| `current_state.last_node_info.metrics` | `object` | 推荐 | 输出摘要 / 日志解析 | 对决策最有用的数值指标 | `{ "micrograph_count": 509 }` |
| `current_state.last_node_info.runtime` | `object` | 推荐 | worker log 解析 | 资源、节点、工作目录等 | `{...}` |
| `current_state.last_node_info.evidence_text` | `array[string]` | 推荐 | worker log 摘要 | 短句证据列表 | `["Job ready to run"]` |
| `current_state.last_node_info.warning_lines` | `array[string]` | 推荐 | worker log 摘要 | 警告摘要列表 | `[]` |
| `current_state.last_node_info.image_refs` | `object` | 推荐 | 日志关联图片 / tile 图 | 图像引用摘要 | `{...}` |
| `current_state.last_node_info.recent_batch_node_ids` | `array[string]` | 否 | 服务端聚合 | 最近一批完成节点 ID；并行批次时有用 | `["J2278", "J2279"]` |

## 5. `timestamps` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.timestamps.created_at` | `string \| null` | 否 | job 文档 | 创建时间 | `"Wed, 06 Aug 2025 02:26:19 GMT"` |
| `current_state.last_node_info.timestamps.started_at` | `string \| null` | 否 | job 文档 | 启动时间 | `"Wed, 06 Aug 2025 02:27:26 GMT"` |
| `current_state.last_node_info.timestamps.completed_at` | `string \| null` | 否 | job 文档 | 完成时间 | `"Wed, 06 Aug 2025 02:30:24 GMT"` |

## 6. `inputs.groups[]` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.inputs.groups[].name` | `string \| null` | 否 | `input_slot_groups` | 输入组内部名称 | `"movies"` |
| `current_state.last_node_info.inputs.groups[].title` | `string \| null` | 否 | `input_slot_groups` | 输入组展示标题 | `"Source Movies"` |
| `current_state.last_node_info.inputs.groups[].count_min` | `integer \| null` | 否 | `input_slot_groups` | 最小连接数 | `0` |
| `current_state.last_node_info.inputs.groups[].count_max` | `integer \| null` | 否 | `input_slot_groups` | 最大连接数；无穷时建议转 `null` | `null` |
| `current_state.last_node_info.inputs.groups[].slot_names` | `array[string]` | 否 | `input_slot_groups.slots` | 输入槽名列表 | `["movie_blob"]` |
| `current_state.last_node_info.inputs.groups[].connected_job_uids` | `array[string]` | 否 | `input_slot_groups.connections` | 已连接上游 job ID 列表 | `[]` |

## 7. `outputs.groups[]` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.outputs.groups[].name` | `string \| null` | 否 | `output_result_groups` | 输出组名称 | `"imported_micrographs"` |
| `current_state.last_node_info.outputs.groups[].description` | `string \| null` | 否 | `output_result_groups` | 输出组描述 | `"Imported micrographs."` |
| `current_state.last_node_info.outputs.groups[].num_items` | `integer \| null` | 否 | `output_result_groups` | 输出项数量 | `509` |
| `current_state.last_node_info.outputs.groups[].field_names` | `array[string]` | 否 | `contains` | 输出字段名列表 | `["micrograph_blob", "mscope_params"]` |
| `current_state.last_node_info.outputs.groups[].scalar_stats` | `object` | 否 | `latest_summary_stats` 过滤结果 | 标量型统计摘要 | `{}` |
| `current_state.last_node_info.outputs.groups[].summary_stat_keys` | `array[string]` | 否 | `latest_summary_stats` | 原始统计键列表 | `[]` |

## 8. `metrics` 字段表

这是给模型看的“高价值摘要”，不要求所有任务都齐全。

| 字段路径 | 类型 | 必填 | 说明 | 示例 |
| --- | --- | --- | --- | --- |
| `current_state.last_node_info.metrics.micrograph_count` | `integer` | 否 | 微图数量 | `509` |
| `current_state.last_node_info.metrics.particle_count` | `integer` | 否 | 粒子数量 | `87716` |
| `current_state.last_node_info.metrics.selected_particle_count` | `integer` | 否 | 被选中的粒子数 | `53214` |
| `current_state.last_node_info.metrics.rejected_particle_count` | `integer` | 否 | 被剔除粒子数 | `12450` |
| `current_state.last_node_info.metrics.class_count` | `integer` | 否 | class 数量 | `50` |
| `current_state.last_node_info.metrics.volume_count` | `integer` | 否 | volume 数量 | `1` |
| `current_state.last_node_info.metrics.mask_count` | `integer` | 否 | mask 数量 | `1` |

## 9. `runtime` 字段表

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.runtime.work_dir` | `string \| null` | 否 | worker log | 工作目录 | `"/home/share/cryoSPARC/P1/J2279"` |
| `current_state.last_node_info.runtime.lane` | `string \| null` | 否 | worker log | 执行 lane | `"default"` |
| `current_state.last_node_info.runtime.worker_hostname` | `string \| null` | 否 | worker log | 执行节点 | `"A100a0"` |
| `current_state.last_node_info.runtime.allocated_cpu` | `array \| string \| null` | 否 | worker log | 分配 CPU 信息 | `[8]` |
| `current_state.last_node_info.runtime.allocated_gpu` | `array \| string \| null` | 否 | worker log | 分配 GPU 信息 | `[]` |
| `current_state.last_node_info.runtime.allocated_ram` | `array \| string \| null` | 否 | worker log | 分配 RAM 信息 | `[6, 7, 8]` |
| `current_state.last_node_info.runtime.allocated_ssd` | `boolean \| string \| null` | 否 | worker log | 是否使用 SSD | `false` |
| `current_state.last_node_info.runtime.micrograph_count_logged` | `integer` | 否 | worker log 正则提取 | 日志里记录的微图数量 | `509` |
| `current_state.last_node_info.runtime.import_file_count_logged` | `integer` | 否 | worker log 正则提取 | 导入文件数 | `509` |
| `current_state.last_node_info.runtime.extracted_particle_count_logged` | `integer` | 否 | worker log 正则提取 | 日志里记录的提取粒子数 | `87716` |

## 10. `image_refs` 字段表

当前仍按文本接口处理图像，只保留引用摘要。

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.image_refs.ui_tile_images` | `array[object]` | 否 | job 文档 | cryoSPARC tile 图引用 | `[]` |
| `current_state.last_node_info.image_refs.output_group_images` | `object` | 否 | job 文档 | 输出组与图像 fileid 映射 | `{}` |
| `current_state.last_node_info.image_refs.event_images` | `array[object]` | 否 | 日志图片引用 | 事件图片摘要列表 | `[]` |
| `current_state.last_node_info.image_refs.event_image_count` | `integer` | 否 | 统计结果 | 事件图片总数 | `3` |
| `current_state.last_node_info.image_refs.event_image_kind_counts` | `object` | 否 | 统计结果 | 各类型图片计数 | `{ "raw_data": 3 }` |

### 10.1 `image_refs.event_images[]`

| 字段路径 | 类型 | 必填 | 推荐来源 | 说明 | 示例 |
| --- | --- | --- | --- | --- | --- |
| `current_state.last_node_info.image_refs.event_images[].filename` | `string` | 是 | 日志图片引用 | 图片文件名 | `"J2279_raw_data_xxx.png"` |
| `current_state.last_node_info.image_refs.event_images[].path` | `string \| null` | 否 | 日志图片引用 | 图片路径 | `"/hdd1/.../J2279_raw_data_xxx.png"` |
| `current_state.last_node_info.image_refs.event_images[].bytes` | `integer \| null` | 否 | 日志图片引用 | 图片大小 | `867876` |
| `current_state.last_node_info.image_refs.event_images[].kind` | `string` | 是 | 规则分类 | 图片类型摘要 | `"raw_data"` |

## 11. 空值与类型约定

### 11.1 空值约定

| 场景 | 推荐表示 |
| --- | --- |
| 未知 `emdb_id` | `null` |
| 未知 `resolution` | `null` |
| 没有已知 workflow | `known_workflow_steps = null` |
| workflow 尚未开始 | `last_node_id = null` |
| 没有警告 | `warning_lines = []` |
| 没有图片 | `event_images = []` |
| 没有参数 | `parameters = {}` |
| 没有输出 | `outputs = { "groups": [] }` |

### 11.2 类型约定

必须保持 JSON 原生类型：

- 布尔值用 `true` / `false`
- 数值用 `number`
- 空值用 `null`
- 不要把布尔值写成 `"True"` / `"False"`
- 不要把数值整体写成字符串

推荐：

```json
{
  "allocated_ssd": false,
  "num_of_maps": 1,
  "resolution": [3.0]
}
```

不推荐：

```json
{
  "allocated_ssd": "False",
  "num_of_maps": "1",
  "resolution": "3.0"
}
```

## 12. 最小可联调 V2 输入

```json
{
  "schema_version": "2.0",
  "task_type": "workflow_decision",
  "dataset_info": {
    "empiar_id": "EMPIAR-12099",
    "emdb_id": null,
    "resolution": null,
    "input_type": null,
    "macromolecules_type": null,
    "num_of_maps": null,
    "abstract": null,
    "known_workflow_steps": null
  },
  "current_state": {
    "last_node_id": null,
    "last_action": null,
    "last_node_status": "not_started",
    "last_node_info": {
      "job_type": null,
      "job_uid": null,
      "job_title": "EMPIAR-12099",
      "project_uid": null,
      "status": "not_started",
      "timestamps": {},
      "inputs": {
        "groups": []
      },
      "parameters": {},
      "outputs": {
        "groups": []
      },
      "metrics": {},
      "runtime": {},
      "evidence_text": [
        "Workflow has not started yet."
      ],
      "warning_lines": [],
      "image_refs": {
        "ui_tile_images": [],
        "output_group_images": {},
        "event_images": [],
        "event_image_count": 0,
        "event_image_kind_counts": {}
      }
    }
  }
}
```
