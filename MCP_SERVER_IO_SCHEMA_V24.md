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
- MCP 只在上游给定的 `data_root` / project scope 内检索，不进行服务器全局盲搜
- 文件不能仅根据文件名或后缀判断可用，必须检查存在性、可读性、可解析性及必要字段
- 只有经过 MCP 实际确认的信息才写入 `dataset_parameter_facts`
- 无法确认的字段使用 `null` 或直接省略，禁止 MCP / Model 根据经验猜测
- `.star` 不是所有数据集必有文件，仅在实际存在且验证可用时填写 `particle_meta_path`
- 参数来源、验证状态等 provenance 应由 MCP 内部日志保存；当前 Model Schema 不新增 provenance 字段

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
- 上游 / 人工至少提供 `empiar_id`、`data_root`，或由系统配置 `data_root`；`data_root` 是 MCP 检索范围约束，不要求新增到当前 Model Schema
- `particle_meta_path` 可由人工提供 hint；否则 MCP 在 `data_root` 范围内自动发现 `.star` / `.cs` 等 particle metadata，并解析确认其确为可用 particle metadata 后再填写
- `volume_blob_path` 由 MCP 根据 `emdb_id` 推导候选文件，在约定的本地 EMDB 数据目录中查找，并确认实际存在、可读后再填写
- `blob_exists` 由 MCP 检查 particle metadata 引用的实际 particle / blob 文件是否存在、可读后给出
- `ctf_exists` 由 MCP 解析 `.star` / `.cs` 等 metadata，根据实际 CTF 字段判断
- `accel_kv`、`cs_mm`、`psize_A` 允许人工明确提供；否则优先由 MCP 从 STAR / XML / CS / MRC header / 已有 metadata 等可靠来源提取，其中 `psize_A` 建议在可用时交叉验证
- `enable_validation` 属于 MCP / 系统执行策略配置，不是数据集天然事实；如果写入此处，只表示后续执行时采用该策略

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
- 归并时同样只能使用 MCP 已确认或系统明确配置的信息，不能为某个 job type 补写未确认字段

## 4. `current_state`

### 4.1 Structure

```json
{
  "last_node_id": "J2279",
  "last_action": "import_micrographs",
  "last_node_status": "completed",
  "last_node_info": {},
  "state_features": {},
  "recent_job_history": []
}
```

### 4.2 Core Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `last_node_id` | `string \| null` | Yes | 上一步逻辑节点 ID；未开始时为 `null` |
| `last_action` | `string \| null` | Yes | 上一步 job type；未开始时为 `null` |
| `last_node_status` | `string` | Yes | 归一化状态：`not_started` / `completed` / `failure` |
| `last_node_info` | `object` | Yes | 上一步任务的结构化状态摘要 |
| `state_features` | `object` | Yes | 从 `last_node_info` 中稳定提取出的摘要特征 |
| `recent_job_history` | `array[object]` | Yes | 最近 job 尝试历史，最多 6 条，成功/失败都可记录 |

说明：

- 当前决策时，模型只能看到“当前步之前”的状态
- 这是为了贴近实际落地场景
- `dataset_context` 是全局静态信息
- `current_state` 是到当前时刻为止的动态信息
- `last_node_status == "failure"` 表示上一 job 节点或上一轮 MCP 执行失败，具体错误必须放在 `last_node_info.error_info`
- `last_node_status` 是 MCP-to-model 的归一化状态，不是 CryoSPARC 原始状态；CryoSPARC 原始状态如 `failed` / `killed`，以及 system error、model validation error 等执行失败，都应由 MCP 在这一层归一化为 `failure`
- `last_node_info.status` 保留上一节点或上一轮执行的底层具体状态，可以是 CryoSPARC 原始状态如 `failed` / `killed`，也可以是 MCP 自定义具体状态如 `mcp_file_validation_failed` / `model_output_validation_failed` / `system_call_failed`
- 因此失败场景下允许出现：`current_state.last_node_status == "failure"`，同时 `current_state.last_node_info.status == "failed"` / `"killed"` / 其他具体失败状态
- 模型做高层决策时优先使用 `last_node_status`；需要区分具体失败来源时再读取 `last_node_info.status`、`error_info.source` 和 `error_info.error_code`

### 4.3 `recent_job_history[]`

`recent_job_history` 用于给模型提供最近 job 尝试历史，最多保留 6 条。它是 `current_state` 的字段，不属于单个 `last_node_info`。

排序规则：按时间从旧到新排列，即越靠后的记录越接近当前状态。这样模型可以按执行轨迹理解最近尝试。

每条建议包含：

```json
{
  "job_uid": "J2280",
  "job_type": "import_particles",
  "attempt_index": 1,
  "status": "failure",
  "key_parameters": {
    "particle_meta_path": "/path/to/particles.star",
    "psize_A": 1.2156
  },
  "output_summary": {
    "output_group_names": [],
    "output_field_names": [],
    "num_items_by_group": {}
  },
  "error_summary": {
    "error_code": "IMPORT_METADATA_PARSE_FAILED",
    "message": "Missing required particle image field."
  }
}
```

字段说明：

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `job_uid` | `string \| null` | Yes | CryoSPARC job UID；非 CryoSPARC 错误可为 `null` |
| `job_type` | `string \| null` | Yes | job type 或 MCP action type |
| `attempt_index` | `integer` | Yes | 同类尝试序号，从 1 开始 |
| `status` | `string` | Yes | `completed` 或 `failure` |
| `key_parameters` | `object` | Yes | 关键参数摘要 |
| `output_summary` | `object` | Yes | 输出摘要 |
| `error_summary` | `object \| null` | Yes | 失败摘要；成功时为 `null` |

`output_summary` 固定结构：

```json
{
  "output_group_names": [],
  "output_field_names": [],
  "num_items_by_group": {}
}
```

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
  "recent_batch_node_ids": ["J2278", "J2279"],
  "error_info": null
}
```

### 5.2 Field Summary

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `job_type` | `string \| null` | Yes | 上一步 job type |
| `job_uid` | `string \| null` | Yes | 上一步真实 job ID |
| `job_title` | `string \| null` | Recommended | job 标题 |
| `project_uid` | `string \| null` | Recommended | cryoSPARC project ID |
| `status` | `string` | Yes | 上一节点或上一轮执行的底层具体状态，不要求与 `current_state.last_node_status` 相同 |
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
| `error_info` | `object \| null` | Yes | 原始错误信息；仅当 `last_node_status == "failure"` 时非 `null` |

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

### 5.8 `error_info`

`error_info` 用于记录上一 job 节点或上一轮 MCP 执行失败时的事实型错误信息。成功或未开始时固定为 `null`；当 `current_state.last_node_status == "failure"` 时必须为对象。

结构：

```json
{
  "error_type": "ToolExecutionError",
  "error_code": "IMPORT_METADATA_PARSE_FAILED",
  "message": "Failed to parse particle metadata.",
  "raw_error_text": "Missing required field _rlnImageName in particles.star.",
  "raw_error_truncated": false,
  "stderr_tail": null,
  "stderr_truncated": false,
  "log_path": "/path/to/mcp/internal/job/J2280.log",
  "source": "cryosparc_job_log"
}
```

字段说明：

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `error_type` | `string \| null` | Recommended | 错误类型，例如 `ToolExecutionError` |
| `error_code` | `string` | Yes | 机器可读错误码 |
| `message` | `string` | Yes | 给模型看的短错误说明 |
| `raw_error_text` | `string \| null` | Yes | 原始错误文本，最多 2000 字符；超出时保留 tail |
| `raw_error_truncated` | `boolean` | Yes | `raw_error_text` 是否被截断 |
| `stderr_tail` | `string \| null` | Optional | stderr 尾部摘要，最多 2000 字符；超出时保留 tail |
| `stderr_truncated` | `boolean` | Optional | `stderr_tail` 是否被截断 |
| `log_path` | `string \| null` | Optional | MCP 内部或 CryoSPARC 日志路径 |
| `source` | `string` | Yes | 错误来源 |

`source` 建议枚举：

- `cryosparc_job_log`
- `mcp_file_validation`
- `model_output_validation`
- `system_call`
- `mcp_tool_execution`
- `unknown`

## 6. `state_features`

这是 `v23` 新补充并已经验证有效的稳定状态摘要。2026-08-08 对杭州服务器 `P2/W4`、`P2/W8`、`P2/W9`、`P2/W10` 的既有 CryoSPARC workflow 做过只读测试：这些字段可以由 CryoSPARC job 的 `outputs.groups[].name`、`outputs.groups[].field_names` 和 `num_items` 直接读取或简单派生。

原则：

- 只保留“可稳定从字段直接得到”的特征
- 不加入规则派生、弱 hint、推测性标签
- 布尔 `has_*` 字段表示“从上一步输出组名/字段名中确认是否存在”
- 布尔 `has_* == false` 只表示 MCP 在当前节点输出组名/字段名中未观察到该特征，不等价于该数据集或整个 workflow 全局不存在该特征
- 计数字段表示“当前节点对应输出组的数量”，只在对应 workflow 阶段有意义；没有对应输出时填 `null`

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
| `has_templates` | `boolean` | Yes | 上一步输出组名/字段名中是否含 templates，例如 `templates`、`templates_selected`、`class_averages` |
| `has_ctf` | `boolean` | Yes | 上一步输出字段中是否含 `ctf`；表示已有 CTF 参数 |
| `has_ctf_stats` | `boolean` | Yes | 上一步输出字段中是否含 `ctf_stats`；表示已有 CTF 估计质量统计 |
| `has_initial_volume` | `boolean` | Yes | 上一步输出组名/字段名中是否已有 volume/map，例如 `volume`、`volume_class_0`、`volumes_all_classes`、`map` |
| `has_selected_particles` | `boolean` | Yes | 上一步输出组名中是否已有 `particles_selected` / `selected_particles` |
| `has_particle_alignments_2d` | `boolean` | Yes | 上一步输出字段中是否已有 `alignments2D` |
| `has_particle_alignments_3d` | `boolean` | Yes | 上一步输出字段中是否已有 `alignments3D` / `alignments3D_multi` |
| `has_filament_metadata` | `boolean` | Yes | 上一步输出组名/字段名中是否已有 filament / helix / helical 相关字段 |
| `micrograph_count` | `integer \| null` | Yes | micrograph/exposure 类输出数量；当前节点没有对应输出时为 `null` |
| `particle_count` | `integer \| null` | Yes | particle 类输出数量；当前节点没有对应输出时为 `null` |
| `selected_particle_count` | `integer \| null` | Yes | selected particle 类输出数量；通常只在 `select_2D` 等选择步骤后出现 |
| `rejected_particle_count` | `integer \| null` | Yes | rejected/excluded/unused particle 类输出数量；当前节点没有对应输出时为 `null` |
| `volume_count` | `integer \| null` | Yes | volume/map 类输出数量；通常在 ab initio/refine 阶段出现 |
| `class_count` | `integer \| null` | Yes | class/template 类输出数量；通常在 2D classification / selection 阶段出现 |
| `mask_count` | `integer \| null` | Yes | mask 类输出数量；通常在 refine 阶段出现 |
| `last_output_group_names` | `array[string]` | Yes | 上一步输出组名列表，直接来自 CryoSPARC outputs |
| `last_output_field_names` | `array[string]` | Yes | 上一步输出字段名列表，直接来自 CryoSPARC output results/fields |

说明：

- 目前测试显示，`last_output_group_names` 和 `last_output_field_names` 是最稳定的原始事实字段，应优先保留。
- 上述布尔 `has_*` 字段可以从 CryoSPARC 的 output group names / field names 直接或简单派生，MCP server 侧可稳定生成。
- 计数字段适合保留，但必须允许 `null`：它们只在对应 job 阶段出现。例如 refine 阶段有 `volume_count` / `mask_count`，import/CTF 阶段通常没有；`select_2D` 阶段才有 `selected_particle_count`。
- `has_ctf` 和 `has_ctf_stats` 必须分开：CTF 阶段和早期 particle 阶段通常能看到 `ctf_stats`；到 `class_2D` / refine 阶段，通常还能看到 `ctf`，但不一定还有 `ctf_stats`。
- 目前测试过的 `P2/W4`、`P2/W8`、`P2/W9`、`P2/W10` 中，`has_filament_metadata` 全部为 `false`。只能说明这些样本不是 filament/helical workflow；如需验证该字段，需要额外使用真实 filament/helical 样本。
- `P2/W10` 的完整流程观察：`import_micrographs` 输出 `imported_micrographs`；`patch_ctf_estimation_multi` 输出 `ctf` 和 `ctf_stats`；`class_2D_new` 输出 `alignments2D`；`select_2D` 输出 `particles_selected` / `templates_selected`；`homo_abinit` 和 `homo_refine_new` 输出 `alignments3D` / `map` / `volume` / `mask`。
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
| `decision_type` | `string` | Yes | One of `forward`, `branch`, `stop` |
| `selected_actions` | `array[object]` | Yes | 下一步动作列表 |

### 7.3 `decision_type`

定义：

- `forward`
  - 下一步执行一个动作
- `branch`
  - 下一步需要并行执行多个动作
- `stop`
  - 当前 workflow 应停止，无需后续动作
- `rollback`
  - 当前不允许rollback

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
- `confidence`（后续考虑）
- `evidence`
- `rollback_target`
- `branch_plan`
- `workflow_node_id`
- `connections`

MCP server 侧建议直接做 schema 校验。

如果出现这些额外字段，建议视为格式不合规输出。

## 9. MCP Server Responsibilities

建议 MCP server 负责：

1. 接收上游 / 人工提供的最小数据集入口信息，例如 `empiar_id`、`data_root`，以及可选 hint
2. 在 `data_root` / project scope 内发现、解析并验证数据集文件，不进行服务器全局盲搜
3. 收集并标准化输入字段
4. 对 `particle_meta_path`、`volume_blob_path`、`blob_exists`、`ctf_exists`、`accel_kv`、`cs_mm`、`psize_A` 等字段执行来源解析和可用性检查
5. 组装输入 JSON
6. 调用模型
7. 校验模型输出是否满足 `schema_version=3.0` 的最小输出格式
8. 将 `selected_actions` 映射为实际工具调用
9. 在执行前补做安全检查
10. 根据当前 DAG、上游输出和目标 job input schema 自动解析 CryoSPARC input connections；模型不得提供 `connections`

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
- `current_state.last_node_status in {"not_started", "completed", "failure"}`
- `current_state.last_node_info` 存在
- `current_state.state_features` 存在
- `current_state.recent_job_history` 存在且最多 6 条
- 当 `current_state.last_node_status == "failure"` 时，`current_state.last_node_info.error_info` 必须非 `null`
- 当 `current_state.last_node_status != "failure"` 时，`current_state.last_node_info.error_info` 应为 `null`
- `current_state.last_node_info.status` 是底层具体状态，不要求等于 `current_state.last_node_status`

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
    "last_node_id": "J2280",
    "last_action": "import_particles",
    "last_node_status": "failure",
    "last_node_info": {
      "job_type": "import_particles",
      "job_uid": "J2280",
      "job_title": "10059-8117(2.95A)-5irx",
      "project_uid": "P1",
      "status": "failed",
      "timestamps": {
        "created_at": "2026-08-08T10:00:00+08:00",
        "queued_at": "2026-08-08T10:01:00+08:00",
        "started_at": "2026-08-08T10:02:00+08:00",
        "completed_at": null,
        "failed_at": "2026-08-08T10:03:00+08:00",
        "updated_at": "2026-08-08T10:03:00+08:00"
      },
      "inputs": { "groups": [] },
      "parameters": {
        "accel_kv": 300,
        "blob_exists": true,
        "cs_mm": 2,
        "ctf_exists": true,
        "enable_validation": true,
        "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
        "psize_A": 1.2156
      },
      "outputs": { "groups": [] },
      "metrics": {},
      "runtime": {
        "work_dir": "/path/to/mcp/internal/job/J2280",
        "lane": null,
        "worker_hostname": null,
        "allocated_cpu": null,
        "allocated_gpu": null,
        "allocated_ram": null,
        "allocated_ssd": null
      },
      "evidence_text": [
        "Job J2280 (import_particles) status: failure.",
        "Particle metadata parsing failed before usable outputs were produced."
      ],
      "warning_lines": [
        "Missing required particle image field."
      ],
      "image_refs": {
        "ui_tile_images": [],
        "output_group_images": {},
        "event_images": [],
        "event_image_count": 0,
        "event_image_kind_counts": {}
      },
      "recent_batch_node_ids": [],
      "error_info": {
        "error_type": "ToolExecutionError",
        "error_code": "IMPORT_METADATA_PARSE_FAILED",
        "message": "Failed to parse particle metadata.",
        "raw_error_text": "Missing required field _rlnImageName in particles.star.",
        "raw_error_truncated": false,
        "stderr_tail": null,
        "stderr_truncated": false,
        "log_path": "/path/to/mcp/internal/job/J2280.log",
        "source": "cryosparc_job_log"
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
    },
    "recent_job_history": [
      {
        "job_uid": "J2280",
        "job_type": "import_particles",
        "attempt_index": 1,
        "status": "failure",
        "key_parameters": {
          "particle_meta_path": "/home/share/empiar/10059/data/particles/particles.star",
          "psize_A": 1.2156
        },
        "output_summary": {
          "output_group_names": [],
          "output_field_names": [],
          "num_items_by_group": {}
        },
        "error_summary": {
          "error_code": "IMPORT_METADATA_PARSE_FAILED",
          "message": "Missing required particle image field."
        }
      }
    ]
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
