# CryoAgent MCP Server I/O Schema Alignment (Current v23)

本文档用于和 MCP server 侧对齐当前模型实际使用的输入输出 schema。

这份文档描述的是截至当前训练与评估所使用的真实版本：

- 输入 schema: `2.1`
- 输出 schema: `3.0`
- 输出风格: `minimal_v3`
- 当前训练设置: `no-workflow`
  - 即 `dataset_context.dataset_metadata.known_workflow_steps` 当前固定为 `null`

本文档优先回答四个问题：

1. MCP server 需要给模型传什么输入
2. 哪些字段是当前版本真正用到的
3. 模型会返回什么输出
4. MCP server 在上线时需要做哪些校验

## 1. Overall Contract

完整链路建议如下：

1. MCP server 收集数据集静态信息和当前运行状态
2. MCP server 组装单个 JSON payload 作为模型输入
3. 模型只返回严格 JSON 决策
4. MCP server 校验输出 schema
5. MCP server 将输出映射为实际工具调用

当前版本中：

- 模型输入是一个 JSON 对象
- 模型输出也是一个 JSON 对象
- 不再向模型显式提供 `candidate_actions`
- 不要求模型输出 `workflow_node_id`
- 当前输出只保留最小决策信息：
  - `decision_type`
  - `selected_actions[].job_type`
  - `selected_actions[].parameters`

## 2. Input Schema

### 2.1 Top-level Structure

当前实际输入结构如下：

```json
{
  "schema_version": "2.1",
  "task_type": "workflow_decision",
  "dataset_context": {},
  "current_state": {}
}
```

### 2.2 Top-level Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | `string` | Yes | Fixed to `"2.1"` |
| `task_type` | `string` | Yes | Fixed to `"workflow_decision"` |
| `dataset_context` | `object` | Yes | Dataset-level static context |
| `current_state` | `object` | Yes | Current step decision context |

## 3. `dataset_context`

### 3.1 Structure

```json
{
  "dataset_metadata": {},
  "dataset_parameter_facts": {},
  "dataset_parameter_facts_by_job_type": {}
}
```

### 3.2 `dataset_metadata`

这是数据集级别的静态背景信息，理论上在整个 workflow 决策过程中都可见。

当前结构：

```json
{
  "empiar_id": "EMPIAR-10059",
  "emdb_id": "EMD-8117",
  "resolution": [3.038],
  "input_type": "particle",
  "macromolecules_type": "protein",
  "num_of_maps": 1,
  "abstract": "TRPV1 structures in nanodiscs reveal mechanisms of ligand and lipid action...",
  "known_workflow_steps": null,
  "label_empiar_id": 10059
}
```

字段说明：

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `empiar_id` | `string` | Yes | 数据集 ID，建议固定格式 `EMPIAR-xxxxx` |
| `emdb_id` | `string \| null` | Recommended | 对应 EMDB ID，建议固定格式 `EMD-xxxxx` |
| `resolution` | `array[number] \| null` | Recommended | 标签文件中的分辨率列表 |
| `input_type` | `string \| null` | Recommended | 如 `micrograph` / `particle` |
| `macromolecules_type` | `string \| null` | Recommended | 如 `protein` / `helix` |
| `num_of_maps` | `integer \| null` | Recommended | map 数量 |
| `abstract` | `string \| null` | Recommended | 由 EMDB XML 摘要出的数据集背景文本 |
| `known_workflow_steps` | `array \| null` | Yes | 当前版本固定为 `null` |
| `label_empiar_id` | `integer \| null` | Optional | 标签文件中的原始数字 ID |

说明：

- `known_workflow_steps` 虽然字段保留，但当前 `v23` 训练和评估中是屏蔽状态
- 也就是说，当前这版模型学习的是“无已知 workflow 参考”的自主决策
- 如果后续恢复已知 workflow 输入，建议仍沿用同一字段，不要再设计另一套接口

### 3.3 `dataset_parameter_facts`

这是数据集事实型参数。

设计原则：

- 它们属于“输入事实”，不是模型自由猜测的目标
- 只要在整个任务过程中都成立，就应当放在这里
- 模型在任意步骤做决策时都可以看到这些信息

示例：

```json
{
  "accel_kv": 300,
  "blob_exists": true,
  "cs_mm": 2,
  "ctf_exists": true,
  "enable_validation": true,
  "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
  "psize_A": 1.2156,
  "volume_blob_path": "/hdd1/msai/db/emdb/emd_8117.map"
}
```

常见字段包括：

- `accel_kv`
- `blob_paths`
- `blob_exists`
- `cs_mm`
- `ctf_exists`
- `enable_validation`
- `particle_meta_path`
- `particle_blob_path`
- `psize_A`
- `total_dose_e_per_A2`
- `volume_blob_path`

说明：

- 这些字段不要求每个数据集都齐全
- 缺失时建议直接不写该 key，或在必须保留固定 key 时使用 `null`
- 保持 JSON 原生类型，不要把数值或布尔值转成字符串

### 3.4 `dataset_parameter_facts_by_job_type`

这是按 job type 分组后的数据集事实参数，作用是让模型更容易知道“某类 job 需要哪些事实参数”。

示例：

```json
{
  "import_volumes": {
    "volume_blob_path": "/hdd1/msai/db/emdb/emd_8117.map"
  },
  "import_particles": {
    "accel_kv": 300,
    "blob_exists": true,
    "cs_mm": 2,
    "ctf_exists": true,
    "enable_validation": true,
    "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
    "psize_A": 1.2156
  }
}
```

建议：

- 这是当前版本中很有价值的辅助字段，建议 MCP server 保留
- 它可以由 `dataset_parameter_facts` 进一步归并得到
- 逻辑上仍属于静态上下文，而不是动态状态

## 4. `current_state`

### 4.1 Structure

```json
{
  "last_node_id": "J2279",
  "last_action": "import_micrographs",
  "last_node_status": "completed",
  "last_node_info": {},
  "state_features": {}
}
```

### 4.2 Core Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `last_node_id` | `string \| null` | Yes | 上一步逻辑节点 ID；未开始时为 `null` |
| `last_action` | `string \| null` | Yes | 上一步 job type；未开始时为 `null` |
| `last_node_status` | `string` | Yes | 如 `not_started` / `completed` |
| `last_node_info` | `object` | Yes | 上一步任务的结构化状态摘要 |
| `state_features` | `object` | Yes | 从 `last_node_info` 中稳定提取出的摘要特征 |

说明：

- 当前决策时，模型只能看到“当前步之前”的状态
- 这是为了贴近实际落地场景
- `dataset_context` 是全局静态信息
- `current_state` 是到当前时刻为止的动态信息

## 5. `last_node_info`

### 5.1 Structure

```json
{
  "job_type": "import_micrographs",
  "job_uid": "J2279",
  "job_title": "New Job J2279",
  "project_uid": "P1",
  "status": "completed",
  "timestamps": {},
  "inputs": { "groups": [] },
  "parameters": {},
  "outputs": { "groups": [] },
  "metrics": {},
  "runtime": {},
  "evidence_text": [],
  "warning_lines": [],
  "image_refs": {},
  "recent_batch_node_ids": ["J2278", "J2279"]
}
```

### 5.2 Field Summary

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `job_type` | `string \| null` | Yes | 上一步 job type |
| `job_uid` | `string \| null` | Yes | 上一步真实 job ID |
| `job_title` | `string \| null` | Recommended | job 标题 |
| `project_uid` | `string \| null` | Recommended | cryoSPARC project ID |
| `status` | `string` | Yes | 任务状态 |
| `timestamps` | `object` | Yes | 创建/开始/完成时间 |
| `inputs` | `object` | Yes | 输入组摘要 |
| `parameters` | `object` | Yes | 上一步实际参数 |
| `outputs` | `object` | Yes | 输出组摘要 |
| `metrics` | `object` | Yes | 高价值计数/统计摘要 |
| `runtime` | `object` | Yes | 运行环境和资源摘要 |
| `evidence_text` | `array[string]` | Yes | 文本证据摘要 |
| `warning_lines` | `array[string]` | Yes | 告警摘要 |
| `image_refs` | `object` | Yes | 图像引用摘要 |
| `recent_batch_node_ids` | `array[string]` | Optional | 最近一批已完成节点，主要用于并行 branch 场景 |

### 5.3 `inputs.groups[]`

每个输入组建议包含：

```json
{
  "name": "movies",
  "title": "Source Movies",
  "count_min": 0,
  "count_max": null,
  "slot_names": ["movie_blob"],
  "connected_job_uids": []
}
```

### 5.4 `outputs.groups[]`

每个输出组建议包含：

```json
{
  "name": "imported_micrographs",
  "description": "Imported micrographs.",
  "num_items": 509,
  "field_names": ["micrograph_blob", "mscope_params"],
  "scalar_stats": {},
  "summary_stat_keys": []
}
```

### 5.5 `metrics`

这是给模型看的高价值数字摘要。

常见字段：

- `micrograph_count`
- `particle_count`
- `selected_particle_count`
- `rejected_particle_count`
- `volume_count`
- `class_count`
- `mask_count`

### 5.6 `runtime`

常见字段：

- `work_dir`
- `lane`
- `worker_hostname`
- `allocated_cpu`
- `allocated_gpu`
- `allocated_ram`
- `allocated_ssd`
- `import_file_count_logged`
- 其他可稳定解析出的运行时信息

### 5.7 `image_refs`

当前模型输入仍然是文本 JSON，不直接喂图像张量。

因此这里仅保留图像引用摘要，例如：

- `ui_tile_images`
- `output_group_images`
- `event_images`
- `event_image_count`
- `event_image_kind_counts`

当前版本里，图像本身还没有直接进入模型原生多模态输入。

## 6. `state_features`

这是 `v23` 新补充并已经验证有效的稳定状态摘要。

原则：

- 只保留“可稳定从字段直接得到”的特征
- 不加入规则派生、弱 hint、推测性标签

当前结构：

```json
{
  "has_templates": false,
  "has_ctf": false,
  "has_ctf_stats": false,
  "has_initial_volume": false,
  "has_selected_particles": false,
  "has_particle_alignments_2d": false,
  "has_particle_alignments_3d": false,
  "has_filament_metadata": false,
  "micrograph_count": null,
  "particle_count": null,
  "selected_particle_count": null,
  "rejected_particle_count": null,
  "volume_count": null,
  "class_count": null,
  "mask_count": null,
  "last_output_group_names": [],
  "last_output_field_names": []
}
```

字段说明：

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `has_templates` | `boolean` | Yes | 上一步输出中是否已有 templates |
| `has_ctf` | `boolean` | Yes | 上一步输出字段中是否含 `ctf` |
| `has_ctf_stats` | `boolean` | Yes | 是否含 `ctf_stats` |
| `has_initial_volume` | `boolean` | Yes | 是否已有初始 volume/map |
| `has_selected_particles` | `boolean` | Yes | 是否已有 `particles_selected` |
| `has_particle_alignments_2d` | `boolean` | Yes | 是否已有 `alignments2D` |
| `has_particle_alignments_3d` | `boolean` | Yes | 是否已有 `alignments3D` |
| `has_filament_metadata` | `boolean` | Yes | 是否已有 filament 相关字段 |
| `micrograph_count` | `integer \| null` | Yes | 来自 `metrics` |
| `particle_count` | `integer \| null` | Yes | 来自 `metrics` |
| `selected_particle_count` | `integer \| null` | Yes | 来自 `metrics` |
| `rejected_particle_count` | `integer \| null` | Yes | 来自 `metrics` |
| `volume_count` | `integer \| null` | Yes | 来自 `metrics` |
| `class_count` | `integer \| null` | Yes | 来自 `metrics` |
| `mask_count` | `integer \| null` | Yes | 来自 `metrics` |
| `last_output_group_names` | `array[string]` | Yes | 上一步输出组名列表 |
| `last_output_field_names` | `array[string]` | Yes | 上一步输出字段名列表 |

说明：

- MCP server 侧如果能稳定得到这些字段，建议保留
- 不建议在这一层加入例如“应该进入 refine 阶段”之类派生结论

## 7. Output Schema

### 7.1 Top-level Structure

当前模型输出必须是严格 JSON，且只允许下面三个顶层字段：

```json
{
  "schema_version": "3.0",
  "decision_type": "forward",
  "selected_actions": []
}
```

### 7.2 Top-level Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | `string` | Yes | Fixed to `"3.0"` |
| `decision_type` | `string` | Yes | One of `forward`, `branch`, `stop`, `request_input` |
| `selected_actions` | `array[object]` | Yes | 下一步动作列表 |

### 7.3 `decision_type`

定义：

- `forward`
  - 下一步执行一个动作
- `branch`
  - 下一步需要并行执行多个动作
- `stop`
  - 当前 workflow 应停止，无需后续动作

### 7.4 `selected_actions[]`

每个 action 只允许两个字段：

```json
{
  "job_type": "import_particles",
  "parameters": {
    "accel_kv": 300,
    "blob_exists": true,
    "cs_mm": 2,
    "ctf_exists": true,
    "enable_validation": true,
    "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
    "psize_A": 1.2156
  }
}
```

字段说明：

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `job_type` | `string` | Yes | 要执行的 cryoSPARC job type |
| `parameters` | `object` | Yes | 该 job 的参数字典，可为空对象 |

## 8. Forbidden Output Fields

当前版本明确不允许模型输出以下字段：

- `decision`
- `next_action`
- `action`
- `reason`
- `explanation`
- `confidence`
- `evidence`
- `rollback_target`
- `branch_plan`
- `workflow_node_id`

MCP server 侧建议直接做 schema 校验。

如果出现这些额外字段，建议视为格式不合规输出。

## 9. MCP Server Responsibilities

建议 MCP server 负责：

1. 收集并标准化输入字段
2. 组装输入 JSON
3. 调用模型
4. 校验模型输出是否满足 `schema_version=3.0` 的最小输出格式
5. 将 `selected_actions` 映射为实际工具调用
6. 在执行前补做安全检查

特别说明：

- 模型当前不输出 `workflow_node_id`
- 所以下游执行不应依赖模型去命名新节点
- 更合理的做法是：
  - MCP server 根据 `job_type + parameters` 创建实际任务
  - 新产生的运行时节点 ID 由系统侧分配

## 10. Recommended Validation Rules

MCP server 建议至少做以下检查：

### 10.1 Input-side

- `schema_version == "2.1"`
- `task_type == "workflow_decision"`
- `dataset_context` 存在
- `current_state` 存在
- `current_state.last_node_info` 存在
- `current_state.state_features` 存在

### 10.2 Output-side

- 输出可被解析为单个 JSON object
- `schema_version == "3.0"`
- `decision_type in {"forward", "branch", "stop"}`
- `selected_actions` 是 array
- 每个 action 仅有：
  - `job_type`
  - `parameters`
- `stop` 时建议 `selected_actions = []`
- `forward` 时建议 `selected_actions` 长度为 `1`
- `branch` 时建议 `selected_actions` 长度大于等于 `2`

## 11. Current Example

### 11.1 Input Example

```json
{
  "schema_version": "2.1",
  "task_type": "workflow_decision",
  "dataset_context": {
    "dataset_metadata": {
      "empiar_id": "EMPIAR-10059",
      "emdb_id": "EMD-8117",
      "resolution": [3.038],
      "input_type": "particle",
      "macromolecules_type": "protein",
      "num_of_maps": 1,
      "abstract": "TRPV1 structures in nanodiscs reveal mechanisms of ligand and lipid action. Sample: TRPV1 ion channel in complex with DkTx and RTX. Resolution: 2.95 A.",
      "known_workflow_steps": null,
      "label_empiar_id": 10059
    },
    "dataset_parameter_facts": {
      "accel_kv": 300,
      "blob_exists": true,
      "cs_mm": 2,
      "ctf_exists": true,
      "enable_validation": true,
      "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
      "psize_A": 1.2156,
      "volume_blob_path": "/hdd1/msai/db/emdb/emd_8117.map"
    },
    "dataset_parameter_facts_by_job_type": {
      "import_volumes": {
        "volume_blob_path": "/hdd1/msai/db/emdb/emd_8117.map"
      },
      "import_particles": {
        "accel_kv": 300,
        "blob_exists": true,
        "cs_mm": 2,
        "ctf_exists": true,
        "enable_validation": true,
        "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
        "psize_A": 1.2156
      }
    }
  },
  "current_state": {
    "last_node_id": null,
    "last_action": null,
    "last_node_status": "not_started",
    "last_node_info": {
      "job_type": null,
      "job_uid": null,
      "job_title": "10059-8117(2.95A)-5irx",
      "project_uid": null,
      "status": "not_started",
      "timestamps": {},
      "inputs": { "groups": [] },
      "parameters": {},
      "outputs": { "groups": [] },
      "metrics": {},
      "runtime": {},
      "evidence_text": ["Workflow has not started yet."],
      "warning_lines": [],
      "image_refs": {
        "ui_tile_images": [],
        "output_group_images": {},
        "event_images": [],
        "event_image_count": 0,
        "event_image_kind_counts": {}
      }
    },
    "state_features": {
      "has_templates": false,
      "has_ctf": false,
      "has_ctf_stats": false,
      "has_initial_volume": false,
      "has_selected_particles": false,
      "has_particle_alignments_2d": false,
      "has_particle_alignments_3d": false,
      "has_filament_metadata": false,
      "micrograph_count": null,
      "particle_count": null,
      "selected_particle_count": null,
      "rejected_particle_count": null,
      "volume_count": null,
      "class_count": null,
      "mask_count": null,
      "last_output_group_names": [],
      "last_output_field_names": []
    }
  }
}
```

### 11.2 Output Example

```json
{
  "schema_version": "3.0",
  "decision_type": "branch",
  "selected_actions": [
    {
      "job_type": "import_volumes",
      "parameters": {
        "volume_blob_path": "/hdd1/msai/db/emdb/emd_8117.map"
      }
    },
    {
      "job_type": "import_particles",
      "parameters": {
        "accel_kv": 300,
        "blob_exists": true,
        "cs_mm": 2,
        "ctf_exists": true,
        "enable_validation": true,
        "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
        "psize_A": 1.2156
      }
    }
  ]
}
```

## 12. Practical Notes

当前这版最关键的接口共识可以压缩为三条：

1. 静态事实放进 `dataset_context`
2. 当前状态放进 `current_state`
3. 模型只输出最小 JSON 决策，不输出解释，不输出节点 ID

如果后续我们恢复“已知 workflow 参考输入”或扩展到其他 EM 工具，优先建议继续复用这套顶层结构，而不是重新定义一套完全不同的接口。


## 10. Missing Input and Exploratory Particle Picking

When a required decision fact is unavailable, the model may return a non-executing `request_input` decision:

```json
{
  "schema_version": "2.0",
  "decision_type": "request_input",
  "requested_inputs": ["particle_diameter_A"],
  "reason": "Blob picking requires a particle diameter that is not available.",
  "confidence": 0.95,
  "risk_flags": ["needs_human_input"],
  "evidence": []
}
```

MCP must not create or queue a job for `request_input`.

If the particle size is uncertain but a broad exploratory search is justified, the model may choose `blob_picker_gpu` with both `diameter` and `diameter_max`, and must include `exploratory_parameter_range` in `risk_flags`. MCP still validates the numeric range and input connection before queueing.
