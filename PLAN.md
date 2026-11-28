# 开发计划与分工

本文件记录研究目标、阶段安排、各模块状态和可以认领的任务，供人和 agent 协作时使用。实验结果写在 [REPORT.md](REPORT.md)，用法写在 [README.md](README.md)。设计讨论的原始记录在 [docs/design/](docs/design/)，参数来源在 [docs/research/params.md](docs/research/params.md)。

最后更新：第 2 阶段 B（一）的实验已在本地跑完并写入 REPORT；事件模拟（T-B1d）由子代理开发中（分支 `claude/hbf-kv-study`）。第 3 阶段的设计已定，在分支 `claude/hbf-kv-terms` 上开发。测试和实验在本地运行，见 [docs/local-runs.md](docs/local-runs.md)。

---

## 一、研究目标

研究问题：把 LLM 推理的 KV cache 放在 HBF（High Bandwidth Flash，依据 OCP HBF 规范 v0.7.0）上，读带宽和寿命各是什么情况，压缩能带来多少收益，以及现有 KV cache 方案直接用上去会遇到哪些冲突。

| # | 要测的问题 | 对应阶段 | 状态 |
|---|---|---|---|
| 1 | HBF 带宽不足的表现 | 2A | ✅ 已有结果（REPORT"第 2 阶段 A"） |
| 2 | KV cache 压缩后带宽能提高多少 | 2A | ✅ 已有结果 |
| 3 | KV cache 直接存在 HBF 上的寿命 | 2B | 🔄 开发中 |
| 4 | KV cache 压缩后的寿命收益 | 2B | 🔄 开发中 |
| 5 | 现有 KV cache 方案直接用于 HBF 的效果与冲突 | 2A（读路径）、2B（写路径）、2C（真实方法） | 读路径 ✅；写路径部分完成 |

---

## 二、阶段安排

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | MQSim 验证：页级读时序、T0–T10 验收、可复现性 | ✅ 完成 |
| 1 | HBF 化改造：机制分析模式、真实 die/plane 层级（T-PLANE）、HBF 命令模型（T-CMD）、cache read（T-CACHE）、OCP 拓扑与地址映射、协议约束（MOCS、64 B 读、UCIe） | ✅ 完成 |
| 2A | 读路径：numpy 分析引擎（M7）、读时间估计（M0）、压缩格式（M3）、GPU 与系统模型（M1）、扫描脚本；第 1、2 项实验与第 5 项读路径 | ✅ 完成 |
| 2B | 写路径与寿命：请求放置与写顺序检查（M2）、写入与刷新对读的干扰（M5）、寿命模型（M4）；第 3、4 项实验与第 5 项写路径 | 🔄 进行中 |
| 2C | 真实访问轨迹：真实模型导出（M6）与方法适配器 | ⏳ 未开始（需要 GPU） |
| 3 | 按 KV 大小公式逐项压缩：页格式（MLA、DSA、量化参数同页）、按层/按头分组（混合注意力、跨层共享、DuoAttention、PyramidKV）、模型预设、token 淘汰与写入流、部分通道读、前缀共享；分支 `claude/hbf-kv-terms` | 🔄 K1、K2 完成；接手说明见 [docs/handoff.md](docs/handoff.md) |

### 第 2 阶段 B 的进度

设计见 [docs/design/phase2b-design.md](docs/design/phase2b-design.md)，定案和分工见 [docs/design/phase2b-decisions.md](docs/design/phase2b-decisions.md)。

| 模块 | 文件 | 状态 | 说明 |
|---|---|---|---|
| M2 请求放置 | `hbf/batch.py`、`tests/test_batch.py` | ✅ 已提交，审查问题已修 | 四种分配：`layer_major`、`dedicated`、`packed`、`vllm_pool`，可按通道或 die 分区；快照的写入历史；`check_write_order` 按 §5.4.1 检查跳页、覆盖写和擦除有效块；CLI `--alloc` 等 |
| M5 写入干扰 | `hbf/interference.py`、`tests/test_interference.py` | ✅ 已提交 | fifo、gaps、suspend、random、correlated 五种模式；`python -m hbf.interference RUN_DIR` |
| M4a 时钟 | `hbf/endurance.py`（Clock、calibrate） | ✅ 已提交 | 用拟合的估算加 `system.evaluate` 代替每步跑分析；与阶段 A 的 TPOT 相差 < 0.01% |
| M4b 解析模型 | `hbf/endurance.py`（analytic）、`hbf/workload.py` | ✅ 已提交 | `python -m hbf.endurance --analytic`：写入速率、读干扰刷新、数据保持刷新、擦除放大、到达 P/E 上限的天数；参数可逗号分隔批量扫描 |
| M4c 事件模拟 | `hbf/endurance.py`（simulate）、`hbf/workload.py` | 🔄 子代理开发中（独立 worktree，完成后合并） | 段级状态、按量子推进：到达、写回、读计数、刷新、回收、磨损均衡；未测试的草稿在 `docs/design/drafts/` |
| 扫描接入 | `hbf/sweep.py`（`"tool"` 键） | ✅ | `"tool": "endurance"` 让扫描脚本驱动寿命模型，每点一个组合，共用标定缓存 |
| 实验 | `evidence/phase2b/` | 🔄 | E3.1、E4.1、E5.1、E5.2、E5.6 已在本地跑完；放置实验需在 T-B7 后重跑；其余等事件模拟 |
| 报告 | REPORT"第 2 阶段 B" | 🔄 | "第 2 阶段 B（一）"已写；（二）等事件模拟 |

第 2 阶段 B 的实验矩阵（详见设计文档 §9）：

| 编号 | 问题 | 依赖 |
|---|---|---|
| E3.1 | 寿命曲线（解析）：模型 × 负载 × 读干扰阈值 X × P/E 上限 × 每块页数 | M4b |
| E3.2–E3.4 | 寿命（模拟）、块大小、数据保持刷新 | M4c |
| E4.1–E4.2 | 压缩和稀疏选择对寿命的影响 | M4b、M4c |
| E5.1 | 各分配方式的写顺序合法性 | M2（已可跑） |
| E5.2 | 各分配方式的读并行度 | M2（已可跑） |
| E5.3–E5.4 | 回收粒度、vLLM 空闲链表 | M4c |
| E5.5 | HBM 暂存与未完成写单元数 | 解析 |
| E5.6 | 写入干扰对 TPOT 的影响 | M5（已可跑） |
| E5.7 | 共享前缀的读干扰热点 | M4b、M4c |
| E5.8 | 写入干扰的 MQSim 小规模核对（可选） | M5 |

### 第 2 阶段 C：真实访问轨迹（未开始）

- **M6 导出脚本**：在 HF transformers 中挂钩 attention，按 (step, layer, KV head) 记录选中的 token 区间，写成 `access.csv`；张量规格用 `--format` 或 `--tensors`。需要 GPU，在用户的机器上运行。
- **方法适配器**：KIVI、StreamingLLM、SnapKV、Quest、ThinK、Quest + 4 bit。其中 KIVI 和 StreamingLLM 的读位置是确定的，不需要 GPU 就能先生成。
- **验收**：真实轨迹与对应合成模式（`uniform_tokens`、`sink_recent`）的轮数、读放大、每块读次数对比，并写进 REPORT。

### 第 3 阶段：按 KV 大小的各项压缩（分支 `claude/hbf-kv-terms`）

KV 大小 = 2（K、V）× 层数 × KV 头数 × head_dim × token 数 × 每元素字节 × batch。单纯按字节比例缩小的方法（fp8、int、KIVI、ThinK、裁剪 KV 头）已经支持；这一阶段补上对 HBF 的影响不止字节比例的方法。它们大多通过三种途径起作用：
- 页粒度：一页装几行、选择单位、量化参数是否同页；
- 地址别名与放置：按层、按头的不等大区域，共享区域，独立写前沿；
- 只能顺序写的设备上的 token 流：淘汰或重写的 token 留下死页，而 HBF 没有 GC。

设计见 [docs/design/phase3-kv-terms.md](docs/design/phase3-kv-terms.md)（各项方法、抽象、里程碑、手算验收值、实验矩阵），定案见 [docs/design/phase3-decisions.md](docs/design/phase3-decisions.md)。顺序：K1 与 K2 并行，然后 K3 与 K5，事件模拟合并后做 K4、K7，最后 K6。

### 其他候选方向

- **HBF 感知的布局与压缩格式协同设计**：第 2 项实验显示，16 token 的选择块遇上压缩后的大页，压缩反而更慢；量化参数单独成页，又让读页数翻倍。可以研究的方向：
  - 量化参数与数据同页；
  - 选择块与页对齐；
  - 页摘要留在 HBM。
- **GPU 侧能否直接消费 HBF 的页**（stream）而不经 HBM 暂存：这决定了 8 个 stack 的带宽能不能叠加到 HBM 之上。
- **权重放 HBF**（规范 §13）：权重每步都要读，读干扰刷新会一直存在。
- **多 stack 的地址交织**：目前是按请求划分到各 stack，单个请求跨 stack 交织的情况没有建模。
- **MQSim 的 HBF 写路径**：顺序写、第 0 页自动擦除、无 GC，用于核对 M5 和刷新的精确时序。目前上游 MQSim 一遇到 GC 就会卡死，所以测试里断言不出现擦除。
- **GPU 实测时间表接入**：参照 HBF-EventLab 的做法。

---

## 三、可以认领的任务

每个任务都写明了要改的文件，方便并行。认领前先在本文件的表里写上名字，做完把状态改为 ✅。

| ID | 任务 | 要改的文件 | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-B1 | 寿命模型 M4：时钟、解析、CLI | `hbf/endurance.py`、`hbf/workload.py`、`tests/test_endurance.py` | M5 | 设计 §7 的测试 8–9；E3.1 的解析曲线 | ✅ |
| T-B1d | 寿命模型 M4：事件模拟 | 同上，草稿在 `docs/design/drafts/` | T-B1 | 设计 §7 的测试 10–16；与解析模型在稳态下一致 | 🔄 子代理开发中，计划写在 `docs/design/phase2b-sim-plan.md` |
| T-B7 | `packed` 分配的请求起点错开：原来每个请求从行边界开始，短上下文时各请求同一层落在同一批 bank 上（KIVI-2、2K 时轮数是理想的 5.3 倍） | `hbf/batch.py` | — | 黄金文件不变；短上下文的 κ 接近 1 | ✅ 默认 `--align stagger`，同一点 κ 5.33 → 1.67 |
| T-B2 | 扫描脚本支持 `"tool": "endurance"` | `hbf/sweep.py`、`tests/test_sweep.py` | T-B1 的 CLI | 一个寿命网格能跑出 summary.csv | ✅ |
| T-B3 | E5.1、E5.2 实验：各分配方式的写顺序合法性与读并行度 | `evidence/grids/phase2b-placement.json` | M2 | 本地运行（docs/local-runs.md 的 `placement`），汇总表与 REPORT 段落 | ✅ 已本地运行，写入 REPORT；T-B7 后需重跑 |
| T-B4 | E5.6 实验：写入干扰对 TPOT 的影响（tPROG、tBERS、模式扫描） | `evidence/scripts/interference_scan.py` | M5 | 本地运行（`interference`），表格与 REPORT 段落 | ✅ 已本地运行，写入 REPORT |
| T-B4e | E3.1、E4.1 实验：寿命解析曲线与压缩的寿命收益 | `evidence/scripts/run_local.sh endurance` | T-B1 | 本地运行，表格与 REPORT 段落 | ✅ 已本地运行，写入 REPORT |
| T-B5 | E3.x、E4.x、E5.3–E5.7 实验与报告 | 网格、REPORT | T-B1、T-B2 | REPORT"第 2 阶段 B" | ⏳ |
| T-B6 | 第 2 阶段 B 的审查（代码与报告） | 只读 | T-B1–T-B5 | 按严重程度排序的问题列表，附复现方法 | ⏳ |
| T-C1 | M6 导出脚本（HF transformers，GPU） | 新目录 `frontends/` | — | 对一个小模型导出 `access.csv`，能被 `--access-csv` 读入 | ⏳ 可认领 |
| T-C2 | KIVI、StreamingLLM 的确定性适配器 | 新文件 | — | 与 `--format kivi2`、`--mode sink_recent` 的结果一致 | ⏳ 可认领 |
| T-3a | 量化参数与数据同页的布局 | 并入 T-K1 | — | — | ➡️ T-K1 |
| T-K1 | 页格式：`TensorSpec` 加 `attn_dims`、`logical_dims`；预设 `mla[:split,indexer]`、`mla_fp8`，int/kivi 的 `fused=1[,pow2=1]` | `hbf/access.py`（TensorSpec）、`hbf/formats.py`、`hbf/pages.py`、`tests/test_formats.py` | — | 设计 §3 K1 的手算值；黄金文件和引擎对照不变 | ✅ 主代理完成，待子代理审查 |
| T-K2 | KV 分组：`hbf/kvspec.py`、`GroupedLayout`、`--kv-spec`（hybrid、ssm、cla、yoco、duo、pyramid）、无读层补零行 | `hbf/kvspec.py`（新）、`hbf/cli.py`、`tests/test_kvspec.py` | — | 设计 §3 K2 的手算值；不加 `--kv-spec` 时逐位不变 | ✅ 主代理完成，待子代理审查 |
| T-K3 | 模型预设：MLA、混合注意力、MoE（只计线性层时间） | `hbf/model.py`、`hbf/system.py`、`hbf/configs/models.json` | T-K2 的 `from_model` 接口 | 参数总量与公开值相差 < 1%，出处写进 params.md；现有预设的 TPOT 不变 | ⏳ |
| T-K5 | 选择的真实性：存储密度、静态选择、相邻步复用、GQA 并集 | `hbf/access.py`（generate）、`hbf/pages.py`、`hbf/controller.py`（拒绝） | T-K1 | 设计 §3 K5 的手算值 | ⏳ |
| T-K4 | token 流（写入、死亡、重写）与寿命模型接口 | `hbf/kvspec.py`；事件模拟合并后改 `hbf/endurance.py` | T-K2、T-B1d | 设计 §3 K4 的手算值；现有寿命结果不变 | ⏳ |
| T-K6 | 部分通道读：`loki`、`sparq[:dual]`、`channel_select` | `hbf/formats.py`、`hbf/access.py` | T-K5 | 设计 §3 K6 的手算值 | ⏳ |
| T-K7 | 分组放置、按类别分写前沿、前缀别名、滞留页统计 | `hbf/batch.py`、`tests/test_batch.py` | T-K2、T-K4 | 设计 §3 K7 的手算值 | ⏳ |

---

## 四、协作规则（人和 agent 都适用）

**分支与提交**
- 在第 2 阶段的分支 `claude/hbf-kv-study` 上工作，或者从它切出自己的分支，再由用户决定合并。不要直接推 master。
- 提交说明用英文，写清改了什么、为什么。
- 仓库文件里不要写 AI 模型的名称或版本号。
- 不要提交临时文件和大的运行目录。实验结果只保存汇总 CSV，放在 `evidence/` 下。

**不改变默认行为**
- 新功能一律默认关闭。默认输出必须和改动前逐字节一致，由 `tests/test_golden.py` 守护。
- 只有在有意改变默认结果时才用 `HBF_REGEN_GOLDEN=1` 重新生成黄金文件，并在提交说明里写明原因。
- numpy 引擎和纯 Python 参考实现必须逐行一致，由 `tests/test_engine.py` 守护。

**测试**
- 开发中只跑相关的测试文件，例如 `python -m pytest -q tests/test_batch.py`。
- 合并前跑一次完整测试 `python -m pytest -q`。完整测试约 15 分钟，其中会跑 MQSim，日志放进 `evidence/pytest/`。
- 1,024 plane 的 MQSim 进程峰值内存约 4.2 GB，本机 15 GB，**同一时间只跑一个**。分析模式（`--analysis-only`）可以并行，扫描脚本最多 3 个进程。

**建模与文档**
- 每条依据规范的规则注明章节号，例如 §5.4.1。
- 每个假设的数值都要标为假设，并注明来源（见 [docs/research/params.md](docs/research/params.md)）。规范对 tR、编程和擦除时间、块大小、P/E 上限、读干扰阈值都写"由产品自定"，这些只能做参数扫描，结论写成"在某个参数范围内成立"。
- README、REPORT 和本文件用中文。代码注释、文档字符串和设计笔记用英文。

**流程**
- 每个模块开工前先讨论设计。设计笔记放 `docs/design/`，写清楚输入输出接口。
- 完成后交给另一个人或 agent 审查，修正后在 REPORT 中记录审查表。
- 多个 agent 并行时按文件划分，避免同时改同一个文件。共享接口（函数签名、CSV 列名）在设计笔记中先定下来。

---

## 五、主要接口（新模块请通过这些接口接入）

| 接口 | 位置 | 说明 |
|---|---|---|
| `build_parser()`、`configure(args)`、`pipeline(...)` | `hbf/cli.py` | 命令行解析与一次运行；扫描脚本和校准都复用它们 |
| `iter_layer_pages`、`place`、`LayerPages` | `hbf/pages.py` | 访问记录 → 每层去重后的页数组及其位置 |
| `analyze_accesses(..., timing=True)` | `hbf/analysis.py` | 每层的轮数、读时间估计、需求列（`attended_elements`、`hbf_elements`、`hbm_bytes`、`dequant_ops`）、每步的 `max_block_reads` |
| `estimate_ns`、`block_ids` | `hbf/timing.py` | 读时间估算；擦除块编号 |
| `evaluate(layer_rows, model, gpu, options, meta)` | `hbf/system.py` | 每层、每步的 TPOT 与 HBM 基线；输入约定见 [docs/design/phase2a-decisions.md](docs/design/phase2a-decisions.md) |
| `parse_format`、`TensorSpec`（window、tier、role、dequant_ops） | `hbf/formats.py`、`hbf/access.py` | KV 存储格式 |
| `BatchSpec`、`PageTableLayout`、`check_write_order` | `hbf/batch.py` | 请求放置与写顺序检查 |
| `interference(step_inputs, mode, ...)` | `hbf/interference.py` | 写入与刷新对一步的拖慢 |
| `python -m hbf.sweep GRID.json` | `hbf/sweep.py` | 批量运行与汇总 |

---

## 六、新 agent 的上手顺序

1. 读 README 的"能力范围"和"建模假设"，以及 REPORT 的"结论"和"第 2 阶段 A"。
2. 读本文件，认领一个任务，再读它对应的设计笔记。
3. 安装 numpy 和 pytest，然后跑 `python -m pytest -q tests/test_golden.py tests/test_engine.py`，确认环境正常（约 30 秒）。
4. 用 `--analysis-only` 先跑小例子，例如 README"快速开始"里的命令。
5. 按上面的协作规则开发、测试、提交，并更新本文件的状态。
