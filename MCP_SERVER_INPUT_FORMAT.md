# CryoAgent MCP Server V2 输入格式约定

本文档面向下游 MCP server 联调，说明 V2 版本的 **模型输入 payload** 应该如何构造、各字段从哪里来、哪些字段是必填/推荐/可选，以及已知 workflow 与未知 workflow 两种场景如何统一表示。

目标是让 MCP server 和模型侧对齐：

- MCP server 负责从 workflow / 日志 / EMDB XML / 标签文件中收集状态
- MCP server 负责把状态组装成一个 JSON payload
- 模型只负责读取该 payload 并输出严格 JSON 决策
- MCP server 再去校验模型输出并执行实际工具调用

## 1. 结论先行

V2 输入不再围绕 `candidate_actions` 构造，而是统一采用：

- `dataset_info`
- `current_state`

模型任务定义为：

- 输入：数据集总体信息 + 当前处理状态
- 输出：下一步动作和对应参数

这套 schema 同时覆盖两种场景：

- 已知 workflow：输入里提供 `known_workflow_steps`
- 未知 workflow：输入里把 `known_workflow_steps` 置为 `null`

也就是说，模型始终只有一套接口，不拆成两套模型。

## 2. 顶层输入结构

模型输入是一个单独的 JSON 对象，建议结构如下：

```json
{
  "schema_version": "2.0",
  "task_type": "workflow_decision",
  "dataset_info": {},
  "current_state": {}
}
```

## 3. 顶层字段说明

### 3.1 必填字段

- `schema_version`
  - 固定为 `"2.0"`
- `task_type`
  - 固定为 `"workflow_decision"`
- `dataset_info`
- `current_state`

### 3.2 不再保留的 V1 字段

以下旧字段不再作为 V2 主接口的一部分：

- `dataset`
- `reference_data`
- `workflow`
- `candidate_actions`
- `constraints`

如果后续下游服务端内部仍需要这些对象，可以在 MCP server 内部维护，但不建议继续直接暴露给模型。

## 4. `dataset_info`

`dataset_info` 表示当前数据集的总体背景，是相对稳定的静态上下文。

推荐结构：

```json
{
  "empiar_id": "EMPIAR-12099",
  "emdb_id": "EMD-50426",
  "resolution": [3.0],
  "input_type": "particle",
  "macromolecules_type": "protein",
  "num_of_maps": 1,
  "abstract": "Primary citation title... Sample: ... Resolution: ...",
  "known_workflow_steps": []
}
```

### 4.1 字段来源

- `empiar_id`
  - 来源：workflow 文件名或标签文件
- `emdb_id`
  - 来源：`emd-xxxx.xml` 或 workflow 中的参考 map 路径
- `resolution`
  - 来源：`workflow_label.json`
- `input_type`
  - 来源：`workflow_label.json`
- `macromolecules_type`
  - 来源：`workflow_label.json`
- `num_of_maps`
  - 来源：`workflow_label.json`
- `abstract`
  - 来源：EMDB XML 中的 citation / sample / resolution 摘要
- `known_workflow_steps`
  - 来源：已知 workflow JSON

### 4.2 字段含义

- `empiar_id`
  - string
  - 格式建议固定为 `EMPIAR-xxxxx`

- `emdb_id`
  - string 或 `null`
  - 格式建议固定为 `EMD-xxxxx`

- `resolution`
  - array 或 `null`
  - 保留原始标签中的分辨率列表

- `input_type`
  - string 或 `null`
  - 如 `micrograph` / `particle`

- `macromolecules_type`
  - string 或 `null`
  - 如 `protein` / `ribosome`

- `num_of_maps`
  - integer 或 `null`

- `abstract`
  - string 或 `null`
  - 推荐保留紧凑摘要，不要塞整篇文献

- `known_workflow_steps`
  - array 或 `null`
  - 如果 MCP server 能检索到该数据集对应的已知 workflow，则提供完整有序步骤列表
  - 如果检索不到，则传 `null`

## 5. `known_workflow_steps`

这部分是 V2 的关键设计点。

对于已知数据集，`known_workflow_steps` 提供的是 **完整已知参考流程**，不是仅到当前时刻为止的历史步骤摘要。

示例：

```json
{
  "step_index": 0,
  "node_id": "J2028",
  "action": "import_volumes",
  "title": "EMD-50426",
  "description": "",
  "upstream_node_ids": [],
  "parameter_template": {
    "volume_blob_path": "/hdd1/msai/db/emdb/emd_50426.map"
  }
}
```

字段说明：

- `step_index`
  - integer
  - 在参考 workflow 中的稳定顺序编号

- `node_id`
  - string
  - workflow 逻辑节点 ID

- `action`
  - string
  - cryoSPARC job type

- `title`
  - string 或 `null`

- `description`
  - string 或 `null`

- `upstream_node_ids`
  - array[string]
  - 该节点依赖的上游逻辑节点

- `parameter_template`
  - object
  - 该参考 workflow 中该步骤的模板参数

说明：

- 已知 workflow 场景下，MCP server 可以把完整流程作为参考上下文直接传给模型
- 未知 workflow 场景下，`known_workflow_steps` 统一传 `null`
- 后续如果要测试模型探索能力，可以在评测时主动屏蔽这部分字段，而不必改模型接口

## 6. `current_state`

`current_state` 是当前时刻的决策锚点，即模型“现在应该怎么做”的依据。

推荐结构：

```json
{
  "last_node_id": "J2280",
  "last_action": "patch_ctf_estimation_multi",
  "last_node_status": "completed",
  "last_node_info": {}
}
```

### 6.1 字段来源

- `last_node_id`
  - 来源：上一步完成的 workflow 节点 ID
- `last_action`
  - 来源：上一步 job type
- `last_node_status`
  - 来源：上一步任务状态
- `last_node_info`
  - 来源：`Log/<EMPIAR_ID>/jXX.json` 的结构化抽取结果

### 6.2 字段含义

- `last_node_id`
  - string 或 `null`
  - 未开始时传 `null`

- `last_action`
  - string 或 `null`
  - 未开始时传 `null`

- `last_node_status`
  - string
  - 推荐枚举：
    - `not_started`
    - `running`
    - `completed`
    - `failed`
    - `waiting`

- `last_node_info`
  - object
  - 当前最核心的结构化状态对象

## 7. `last_node_info`

`last_node_info` 用来把原始日志、参数、输出和图像引用压缩成一个适合模型消费的结构化对象。

推荐字段：

- `job_type`
- `job_uid`
- `job_title`
- `project_uid`
- `status`
- `timestamps`
- `inputs`
- `parameters`
- `outputs`
- `metrics`
- `runtime`
- `evidence_text`
- `warning_lines`
- `image_refs`
- `recent_batch_node_ids`

说明：

- 这是“上一步任务实际发生了什么”的结构化摘要
- 不建议把整份原始日志直接塞给模型
- 优先保留有决策价值的字段，如输出数量、分辨率、运行状态、警告信息、图像摘要

### 7.1 `timestamps`

示例：

```json
{
  "created_at": "Wed, 06 Aug 2025 02:26:19 GMT",
  "started_at": "Wed, 06 Aug 2025 02:27:26 GMT",
  "completed_at": "Wed, 06 Aug 2025 02:30:24 GMT"
}
```

### 7.2 `inputs`

推荐最小格式：

```json
{
  "groups": [
    {
      "name": "movies",
      "title": "Source Movies",
      "count_min": 0,
      "count_max": null,
      "slot_names": ["movie_blob"],
      "connected_job_uids": []
    }
  ]
}
```

### 7.3 `parameters`

说明：

- 这里表示“上一步任务实际使用过的参数”
- 不是下一步动作参数模板

### 7.4 `outputs`

推荐最小格式：

```json
{
  "groups": [
    {
      "name": "imported_micrographs",
      "description": "Imported micrographs.",
      "num_items": 509,
      "field_names": ["micrograph_blob", "mscope_params"],
      "scalar_stats": {},
      "summary_stat_keys": []
    }
  ]
}
```

### 7.5 `metrics`

说明：

- 存放对决策最有用的数值摘要
- 例如：
  - `micrograph_count`
  - `particle_count`
  - `selected_particle_count`
  - `class_count`
  - `volume_count`

### 7.6 `runtime`

说明：

- 存放运行环境和日志中抽取出的资源信息
- 例如：
  - `work_dir`
  - `lane`
  - `worker_hostname`
  - `allocated_cpu`
  - `allocated_gpu`
  - `allocated_ram`
  - `allocated_ssd`

### 7.7 `evidence_text`

说明：

- 推荐只保留短句摘要
- 不建议直接传整段 worker log

示例：

```json
[
  "Working in directory: /home/share/cryoSPARC/P1/J2279",
  "Running on lane default",
  "Job ready to run",
  "Importing 509 files"
]
```

### 7.8 `warning_lines`

说明：

- 只保留短警告摘要
- 没有则传 `[]`

### 7.9 `image_refs`

说明：

- 当前 V2 文本接口里，先传图像引用摘要，不直接传图像张量
- 后续如果切多模态接口，再升级为真实图像输入

推荐结构：

```json
{
  "ui_tile_images": [],
  "output_group_images": {},
  "event_images": [],
  "event_image_count": 0,
  "event_image_kind_counts": {}
}
```

## 8. 已知 workflow 与未知 workflow 的统一约定

MCP server 建议采用如下逻辑：

### 8.1 已知 workflow

如果上游能检索到该数据集的已知 workflow：

- `dataset_info.known_workflow_steps` 填完整有序流程
- `current_state` 仍然按当前真实状态填写

### 8.2 未知 workflow

如果上游检索不到该数据集的已知 workflow：

- `dataset_info.known_workflow_steps` 传 `null`
- 其它字段保持不变

这样模型始终只面对一套输入 schema。

## 9. 最小可联调输入

如果要先跑通最小闭环，建议最少提供：

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

## 10. 类型与空值约定

联调时最容易出问题的是类型不一致。

### 10.1 空值约定

| 场景 | 推荐表示 |
| --- | --- |
| 未知 `emdb_id` | `null` |
| 没有已知 workflow | `null` |
| workflow 尚未开始 | `last_node_id = null` |
| 没有警告 | `[]` |
| 没有参数 | `{}` |
| 没有输出组 | `{ "groups": [] }` |

### 10.2 类型约定

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
  "resolution": [3.0],
  "num_of_maps": 1
}
```

不推荐：

```json
{
  "allocated_ssd": "False",
  "resolution": "3.0",
  "num_of_maps": "1"
}
```

## 11. 关于图像输入

当前基座模型支持多模态能力，但本次 V2 接口先按文本结构化输入设计。

因此当前建议是：

- 在 `last_node_info.image_refs` 中保留图像引用摘要
- 真正的图像张量、base64 或多图消息格式暂不纳入本版 MCP 契约

后续如果需要升级成多模态接口，可以在不改核心字段语义的前提下，增加图像载荷层。
