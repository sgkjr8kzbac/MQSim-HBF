# Trace 输入格式与公共数据集说明

本模拟器支持两类外部 trace 输入，以及保留的全部合成工作负载预设（`chat`、`long32k`、`agentic`、`reasoning`）和合成 token 选择模式（`dense`、`uniform_tokens`、`uniform_pages`、`sink_recent` 等）。外部请求级 trace 与合成 token 选择模式可以自由组合（例如真实数据集的到达流与前缀共享 + 合成稀疏读取选择）。

> 所有输入经前端转换后，均输出统一的地址轨迹（`addr.npz`、`layers.npz`、`addr.json`），再交给 MQSim、轮数估算或 HBFSim 后端。R7 不改变地址轨迹格式。

---

## 一、请求级 Trace（`requests.csv` 与 Mooncake 公共数据集）

### 1.1 统一 CSV 格式

`hbf/frontend/trace/requests.py` 定义了统一的请求级 CSV 格式，列头为：

```csv
arrival_s,prompt_tokens,output_tokens
```

或带共享前缀编号（可选第 4 列，以及可选第 5 列每组共享前缀长度）：

```csv
arrival_s,prompt_tokens,output_tokens,prefix_id
arrival_s,prompt_tokens,output_tokens,prefix_id,prefix_tokens
```

| 列名 | 必填 | 类型 | 语义与校验规则 |
|---|---|---|---|
| `arrival_s` | 是 | float | 请求到达时间（秒），有限且 `>= 0`；若输入未按到达时间排序，读取时稳定排序 |
| `prompt_tokens` | 是 | int | 完整提示词长度（token 数），`>= 1` |
| `output_tokens` | 是 | int | 解码输出长度（token 数），`>= 1` |
| `prefix_id` | 否 | int / 空 | 共享前缀分组 ID（非负整数）；留空或 `-1` 表示无共享前缀 |
| `prefix_tokens` | 否 | int / 空 | 该前缀分组的共享 token 数，`0 <= prefix_tokens <= prompt_tokens`；仅当存在 `prefix_id` 时允许出现。若未提供此列，可在导入时通过 `prefix_tokens` 参数（或 `--prefix-tokens`）为带 `prefix_id` 的请求指定固定前缀长度 |

转换为 `Workload` / `Request` 时的语义：
- 当 `prefix_id` 有效且共享长度 `pt > 0` 时，请求拆分为共享前缀 `pt` tokens（由同组请求复用，空闲后保留 `retain_s` 秒）与私有未命中后缀 `input = prompt_tokens - pt` tokens，解码生成 `output_tokens` tokens；
- `prompt_tokens` 在 CSV 中始终保留完整原始提示词长度；
- 多小时寿命模拟中，若 `--hours` 超过 trace 持续时间，默认按 trace 周期循环回放（可用 `--requests-no-loop` 关闭）。

### 1.2 Mooncake FAST'25 公共数据集

- **来源仓库**：`kvcache-ai/Mooncake`（FAST'25 发布版），路径 `FAST25-release/traces/conversation_trace.jsonl`
- **下载地址**：`https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl`
- **许可证**：Apache License 2.0
- **校验和（SHA-256）**：
  - `conversation_trace.jsonl`：`b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df`

数据集文件**不提交进 git 仓库**，使用下载脚本拉取到 `.build/traces/mooncake/`（支持幂等校验）：

```bash
scripts/download_mooncake.sh               # 默认下载并校验 conversation_trace.jsonl
scripts/download_mooncake.sh toolagent     # 可选下载 toolagent_trace.jsonl
scripts/download_mooncake.sh all           # 全部下载
```

#### 原始 JSONL 字段与统计

每行是一条 JSON 对象，字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | int | 到达时间戳（毫秒，导入时转为秒 `arrival_s`） |
| `input_length` | int | 提示词长度（token 数，映射为 `prompt_tokens`） |
| `output_length` | int | 输出长度（token 数，映射为 `output_tokens`） |
| `hash_ids` | list[int] | 按 **512-token KV 块**切分的前缀哈希序列，`len(hash_ids) == ceil(input_length / 512)` |

`conversation_trace.jsonl` 基本统计（实测 12,031 条请求）：
- **时间跨度**：0 – 3,536.999 s（约 59.0 分钟），单调递增，平均到达率 **3.401 req/s**；
- **输入长度**：最小 891，中位数 6,909，均值 12,035.1，P90 27,367，P99 85,400，最大 126,195；
- **输出长度**：最小 1，中位数 350，均值 342.6，P90 597，P99 1,118.5，最大 2,000；
- **前缀共享特征**：
  - 全部 12,031 条请求共享首个 512-token 系统提示块（`hash_ids[0] == 0`）；
  - 6,917 / 12,031（57.5%）条请求与其他请求共享 `>= 2` 个连续前缀块（`>= 1,024` tokens），分布在 2,259 个多轮对话分支；
  - 因果块级前缀命中上限为总输入 token 的 **37.36%**；多轮对话轮间间隔中位数 123.0 s，78.6% `<= 300 s`，94.6% `<= 600 s`。

#### `hash_ids` 到 `prefix_id` 的分组规则

`import_mooncake_jsonl(..., prefix_mode=...)` 支持三种模式：
1. `conversation`（默认）：对每个请求的 `hash_ids` 前缀树计数，跳过被全部请求共享的单一根块（若存在更深的 `>= min_prefix_count` 共享分支），沿当前分支计数不变的平台延伸至最长公共前缀。在 `conversation_trace.jsonl` 上得到 2,260 个分组（最小组大小 2、无孤立组、组内 `prefix_tokens` 完全一致），覆盖 81.2% 的请求命中和 **32.05%** 的总输入 token 缓存复用。
2. `longest`：取出现次数 `>= min_prefix_count` 的最长前缀哈希链。
3. `none`：忽略 `hash_ids`（`prefix_id = None`），用于关闭前缀共享的对照实验。

### 1.3 使用示例

**Python API 转换与调用：**

```python
from hbf.frontend.trace.requests import (
    import_mooncake_jsonl, write_requests_csv, read_requests_csv, workload_from_requests,
)

records = import_mooncake_jsonl(".build/traces/mooncake/conversation_trace.jsonl")
write_requests_csv("runs/mooncake_requests.csv", records)
wl = workload_from_requests(records, name="mooncake", retain_s=300.0)
```

**命令行直接运行寿命模拟（含与合成稀疏模式组合）：**

```bash
# 直接读取 Mooncake JSONL（开启 300 s 前缀保留）
python -m hbf lifetime \
  --requests-csv .build/traces/mooncake/conversation_trace.jsonl \
  --requests-retain-s 300 \
  --hours 24 --warmup-h 1 --rd-threshold 1e5,1e6 \
  --output runs/mooncake-dense

# 真实请求流 + 合成稀疏 token 选择（例如 uniform_tokens 10%）
python -m hbf lifetime \
  --requests-csv .build/traces/mooncake/conversation_trace.jsonl \
  --requests-retain-s 300 \
  --mode uniform_tokens --fraction 0.1 \
  --hours 24 --warmup-h 1 --rd-threshold 1e5,1e6 \
  --output runs/mooncake-sparse
```

TOML 配置文件同样支持 `[trace]` 下的 `requests_csv`、`requests_format`、`requests_retain_s`、`requests_prefix_mode`、`requests_max`、`requests_no_loop`。

---

## 二、Token 级 Trace（`access.csv`）

Token 级外部输入沿用 `hbf/frontend/trace/tokens.py` 定义的 `access.csv` 格式，由 `hbf run --access-csv PATH`（或配置文件 `[trace] access_csv = "..."`）直接读取，经 KV 格式与布局映射为地址轨迹。

### 2.1 CSV 列头与校验规则

严格列头（8 列，顺序不可更改）：

```csv
step,layer,kv_head,kind,op,token_start,token_end,bits
```

| 列名 | 类型 | 有效范围与说明 |
|---|---|---|
| `step` | int | 解码步编号，`>= -1`；`-1` 表示 prefill 写入阶段，`0, 1, ...` 表示各解码步 |
| `layer` | int | Transformer 层编号，`0 <= layer < shape.layers` |
| `kv_head` | int | KV head 编号，`0 <= kv_head < shape.kv_heads` |
| `kind` | str | 张量名称，必须存在于当前 KV 格式的 `layout.specs` 中（例如 BF16/FP8 为 `K`、`V`；KIVI 量化还包含 `K_scale`、`K_zero`、`V_scale`、`V_zero` 等） |
| `op` | str | 操作类型，仅允许 `R`（HBF 读）、`W`（HBF 写）、`H`（HBM 缓存命中，不发 HBF 页访问，仅记入 `layers.npz` 需求列） |
| `token_start` | int | 半开区间起点（含），须满足 `window_lo <= token_start < token_end <= window_hi` |
| `token_end` | int | 半开区间终点（不含） |
| `bits` | int | 每元素位宽，必须与 `layout.specs[kind].bits` 一致（例如 BF16 为 `16`，FP8 为 `8`，INT4/KIVI4 为 `4`） |

示例片段（Llama-3.1-8B BF16，第 0 步第 0 层读取第 0 个 KV head 的 K/V 前 128 个 token）：

```csv
step,layer,kv_head,kind,op,token_start,token_end,bits
0,0,0,K,R,0,128,16
0,0,0,V,R,0,128,16
```

### 2.2 真实推理截取脚本（待 GPU 环境就绪）

从真实 LLM 推理（Hugging Face `transformers` 注意力 / KV-cache hook）在线截取 `access.csv` 的脚本（规划编号 `T-C1`）将在具备 GPU 硬件后实现并放置在 `hbf/frontend/trace/capture/` 目录下，输出严格遵循上述 `access.csv` 格式。在无 GPU 环境下，可使用内置合成 token 选择模式（`dense`、`uniform_tokens`、`uniform_pages`、`sink_recent`）或离线生成的 `access.csv` 驱动前后端完整流程。
