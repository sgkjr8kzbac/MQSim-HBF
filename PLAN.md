# 开发计划与分工

本文件记录研究目标、阶段安排、各模块状态和后续可认领的任务，供人与 agent 协作时使用。实验结果写在 [REPORT.md](REPORT.md)，用法写在 [README.md](README.md)，设计讨论记录在 [docs/design/](docs/design/)，参数来源在 [docs/research/params.md](docs/research/params.md)。

最后更新：第 4 阶段整理与重构（[docs/design/restructure.md](docs/design/restructure.md) 的 **R0–R9**）已在 `refactor/modular` 分支全部完成（含 R7 Mooncake 公共请求级轨迹、R9 HBFSim 第二后端评估与 R8 文档收尾）。**后端分工已定案**：MQSim-HBF 保留为单 stack 读干扰/刷新/寿命的主验证后端，轮数模型负责 MOCS/协议与参数扫描，**HBFSim 作为独立交叉核对后端并承接后续多 stack、HBM4 与 D2D 链路实验**。本地批量运行方法见 [docs/local-runs.md](docs/local-runs.md)。

---

## 一、研究目标

研究问题：把 LLM 推理的 KV cache 放在 HBF（High Bandwidth Flash，依据 OCP HBF 规范 v0.7.0）上，读带宽和寿命各是什么情况，压缩能带来多少收益，以及现有 KV cache 方案直接用上去会遇到哪些冲突。

| # | 要测的问题 | 对应阶段 | 状态 |
|---|---|---|---|
| 1 | HBF 带宽不足的表现 | 2A、R9 交叉核对 | ✅ 已有结果（REPORT"第 2 阶段 A"与 R9 三方对照） |
| 2 | KV cache 压缩后带宽能提高多少 | 2A、R9 交叉核对 | ✅ 已有结果 |
| 3 | KV cache 直接存在 HBF 上的寿命 | 2B、R6、R7 | ✅ 已有解析、MQSim 缩小核对、全尺寸事件模拟与 Mooncake 真实轨迹结果 |
| 4 | KV cache 压缩后的寿命收益 | 2B、R7 | ✅ 已有格式压缩与真实请求流 + 稀疏选页结果 |
| 5 | 现有 KV cache 方案直接用于 HBF 的效果与冲突 | 2A（读路径）、2B/R5（写路径）、2C（真实模型） | 读路径 ✅；写路径（含 MQSim 实测核对）✅；真实 GPU 导出 ⏳ |

---

## 二、阶段安排

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | MQSim 验证：页级读时序、T0–T10 验收、可复现性 | ✅ 完成 |
| 1 | HBF 化改造：机制分析模式、真实 die/plane 层级（T-PLANE）、HBF 命令模型（T-CMD）、cache read（T-CACHE）、OCP 拓扑与地址映射、协议约束（MOCS、64 B 读、UCIe） | ✅ 完成 |
| 2A | 读路径：numpy 分析引擎、读时间估计、压缩格式、GPU 与系统模型、扫描脚本；第 1、2 项实验与第 5 项读路径 | ✅ 完成 |
| 2B | 写路径与寿命：请求放置与写序检查（M2）、写入干扰（M5）、寿命解析与事件模拟（M4）、R5 MQSim 写补丁核对、R6 时间线与全尺寸外推 | ✅ 核心完成（vLLM 空闲链表事件模拟与部分扫描实验待按需补全） |
| 3 | 按 KV 大小公式逐项压缩（MLA、kvspec 分组、K4–K7、Loki/SparQ）；已在 R4 抽离为可选扩展 `hbf/ext/kv_terms/` | ✅ 代码完成并迁入 `hbf/ext/kv_terms/`（独立实验与报告待按需跑） |
| 4 | 整理与重构（R0–R9）：基线冻结（`baseline-pre-refactor`）、R1 正确性修复与 `T-PERF`、R2 规范行为作默认、R3 分包与地址轨迹唯一接口、R4 微内核与扩展分离、R5 MQSim HBF 写补丁（`T-HBFW1`–`T-HBFW4`）、R6 寿命 MQSim 核对与全尺寸外推、R7 Mooncake 请求级轨迹接入、R9 HBFSim 第二独立后端评估、R8 文档重写 | ✅ **全部完成**（`579 passed, 3 xfailed`） |
| 2C / 后续 | GPU 真实模型 `access.csv` 导出（`hbf/frontend/trace/capture/`）；基于 HBFSim 的多 stack + HBM4 + D2D 混存实验 | ⏳ 可认领 |

### 重构后核心模块状态

| 模块 | 位置 | 状态 | 说明 |
|---|---|---|---|
| 器件参数与静态映射 | `hbf/device/{config,address}.py` | ✅ | 参数与假设出处、拓扑预设（默认 `ocp`，含 `ocp-8ch`）、§4.6 静态地址映射唯一实现 |
| 前端轨迹输入 | `hbf/frontend/trace/{tokens,synthetic,requests}.py` | ✅ | `access.csv`、合成生成（含分组挂接）、请求流与 Mooncake FAST'25 导入器（`docs/traces.md`） |
| KV 格式、布局与放置 | `hbf/frontend/kv/{tensors,formats,layouts,placement}.py` | ✅ | 核心压缩格式、4 种布局、多请求快照与写序检查（多请求默认 `packed`） |
| 地址轨迹与时间线 | `hbf/frontend/{pages,refresh,timeline,addrtrace}.py` | ✅ | 前后端唯一接口（`addr.npz`/`layers.npz`/`addr.json`）、刷新插入、请求流时间线展开、§5.4.1/§5.4.3 校验 |
| MQSim 后端 | `hbf/backends/mqsim/{build,schedule,run,parse}.py` | ✅ | 14 个补丁（含 `T-PERF` 与 `T-HBFW1`–`T-HBFW4`），单 stack 主验证本体 |
| 轮数后端 | `hbf/backends/rounds/{analysis,timing}.py` | ✅ | numpy 机制分析、读时间估算与 MOCS/UCIe 协议约束 |
| HBFSim 后端 | `hbf/backends/hbfsim/{build,convert,run,parse}.py` | ✅ | 锁定 `8ab183cc`、`raw-physical` 转换器、`hbfsim-ocp.cfg`、`0001-die-blocking.patch` |
| 寿命模型 | `hbf/lifetime/{geometry,clock,counts,analytic,interference,report,cli}.py`、`sim/` | ✅ | 核心 `ReadTimeClock`、MQSim 计数换算、解析模型、段级事件模拟与 M5 干扰 |
| 统一入口与注册表 | `hbf/{registry,run,cli,sweep,models}.py` | ✅ | 统一插件注册表、TOML/JSON 配置与 `--set`、统一子命令 `python3 -m hbf` |
| 可选扩展 | `hbf/ext/{system,kv_terms}/` | ✅ | 可整目录删除不影响核心运行（`tests/test_core_without_ext.py` 守卫） |

---

## 三、后续可以认领的任务

每个任务写明主要涉及的文件，方便并行。认领前请先在本表标明状态。

| ID | 任务 | 涉及文件 | 依赖 | 验收标准 | 状态 |
|---|---|---|---|---|---|
| T-R46-REV | R4–R9 代码与报告独立审查 | 只读 | R4–R9 | 按严重程度列出问题并附复现方法 | ⏳ 可认领 |
| T-MS1 | 基于 HBFSim 的多 stack + HBM4 + D2D 实验（KV 与权重分层、流式消费 vs HBM 暂存、热节流） | `hbf/backends/hbfsim/`、新脚本、`REPORT.md` | R9（已定案由 HBFSim 承接） | 多 stack（2/4/8）与 HBM4 混存对比表写入 REPORT | ⏳ 可认领 |
| T-C1 | M6 真实模型导出脚本（HF transformers attention hook，需 GPU） | `hbf/frontend/trace/capture/`（新） | GPU 环境 | 对小模型导出 `access.csv`，经 `hbf run --access-csv` 跑通并对比合成模式 | ⏳ 待 GPU |
| T-C2 | KIVI、StreamingLLM 确定性适配器 | `hbf/frontend/trace/capture/`（新） | — | 输出与 `--format kivi2`、`--mode sink_recent` 一致 | ⏳ 可认领 |
| T-TR2 | 补充更多公共请求级数据集导入器（Azure LLM inference trace / BurstGPT / Mooncake `toolagent_trace.jsonl`） | `hbf/frontend/trace/requests.py`、`scripts/`、`docs/traces.md` | R7 | 统一为 `requests.csv`，附下载脚本与 SHA-256，跑通 `hbf lifetime --requests-csv` | ⏳ 可认领 |
| T-B1v | 事件模拟补全 `vllm_pool` 空闲链表策略（设计测试 16）与高并发 die 分区提速 | `hbf/lifetime/sim/`、`tests/test_endurance.py` | — | 设计测试 16 通过，旧寿命测试零回退 | ⏳ 可认领 |
| T-B5 | 阶段 2B / 第 3 阶段扩展实验补跑（T-B7 后重跑放置 κ、E3.x/E4.x/E5.3–E5.7 与 `ext/kv_terms` 对比） | `evidence/grids/`、`REPORT.md` | R4–R7 | 汇总 CSV 入 `evidence/`，结论写入 REPORT | ⏳ 可认领 |

---

## 四、协作规则（人与 agent 均适用）

**分支与提交**
- 在 `refactor/modular` 或从它切出的分支上工作，由用户决定合并与推送；不要直接推 `master`。
- 提交说明用英文，写清改了什么、为什么；每个提交要么是**纯搬移**（固定种子输出逐字节相同，注明 `move only`），要么是**有意改变行为**（写清原因，必要时用 `HBF_REGEN_GOLDEN=1` 更新黄金文件），两者不混在一个提交里。
- 仓库文件里不要写 AI 模型的名称或版本号。
- 不要提交临时文件、数据集本体或大型运行目录；实验结果只保存汇总 CSV 于 `evidence/` 下。

**默认行为与多后端原则**
- 规范行为（OCP HBF v0.7.0）一律是默认值，不再为旧默认加兼容开关；旧报告数字靠 tag `baseline-pre-refactor` 复现。
- 单 stack 读干扰/刷新/寿命以 **MQSim-HBF** 为主验证本体，轮数模型公式改动须附 MQSim 对照；多 stack / HBM4 / D2D 实验采用 **HBFSim** 后端，且不得破坏 MQSim 后端结果。
- 前后端唯一接口为地址轨迹（`addr.npz`/`layers.npz`/`addr.json`）；如确需新增字段，须同步更新 MQSim、轮数与 HBFSim 三个后端。

**测试与资源约束**
- 开发中先跑相关测试文件，例如 `python3 -m pytest -q tests/test_hbfsim.py`。
- 里程碑完成前跑一次完整单进程测试 `python3 -m pytest -q`，日志存入 `evidence/pytest/`。
- 1,024 plane 的 MQSim 进程峰值内存约 **4.2 GB**（当前本地宿主机内存 7.6 GiB），**同一时间只跑一个**。轮数分析（`--analysis-only`）与 HBFSim（约 0.4 GB）可并行。

**建模与文档**
- 每条依据规范的规则注明章节号（如 §4.6、§5.4.1、§5.4.3、§11.5）。
- 每个假设数值均标明为假设并注明来源（[docs/research/params.md](docs/research/params.md)），器件产品参数以扫描范围给出结论。
- `README.md`、`REPORT.md`、`PLAN.md` 与面向用户的 `docs/*.md` 用中文；代码注释、docstring 与设计笔记用英文。

---

## 五、主要接口（新模块请通过以下接口接入）

| 接口 | 位置 | 说明 |
|---|---|---|
| `REGISTRY`、`load_extensions()` | `hbf/registry.py` | 统一插件注册表（格式、布局、合成模式、放置、时钟与评估器） |
| `load_config`、`apply_overrides`、`config_to_*_argv` | `hbf/run.py` | TOML/JSON 配置解析、`--set` 覆盖与 `check.json` 汇总 |
| `main()`、`build_parser()`、`pipeline()` | `hbf/cli.py` | 统一子命令入口（`run`/`check`/`sweep`/`lifetime`/`system`/`interference`） |
| `place`、`device_lpa`、`block_page`、`block_ids` | `hbf/device/address.py` | §4.6 与 CWDP 静态物理地址映射唯一实现 |
| `read_requests_csv`、`import_mooncake_jsonl`、`workload_from_requests` | `hbf/frontend/trace/requests.py` | 统一请求级轨迹与 Mooncake 导入，构建 `Workload`/`Stream` |
| `iter_trace`、`write_trace`、`read_trace`、`validate` | `hbf/frontend/addrtrace.py` | 前后端唯一接口（`addr.npz`/`layers.npz`/`addr.json`）及规范校验 |
| `build_timeline`、`insert_refreshes` | `hbf/frontend/timeline.py`、`refresh.py` | 请求流按时间展开为 LPA 级读写/释放/刷新地址轨迹 |
| `simulate`、`batches_from_trace`、`block_counts` | `hbf/backends/mqsim/` | MQSim 后端执行、地址轨迹转批与 `T-HBFW4` 每块计数解析 |
| `analyze_trace`、`estimate_ns` | `hbf/backends/rounds/` | 轮数与机制分析、读时间估算与主机协议约束 |
| `build`、`convert_trace`、`simulate_trace`、`HbfSimSession` | `hbf/backends/hbfsim/` | HBFSim 第二独立后端构建、地址转换与 `raw-physical` 执行 |
| `StepClock`、`ReadTimeClock`、`counts_from_trace`、`simulate`、`analytic` | `hbf/lifetime/` | 核心寿命时钟接口、MQSim 计数换算、解析模型与段级事件模拟 |
| `python3 -m hbf sweep GRID.json` | `hbf/sweep.py` | 配置键笛卡儿积批量运行与汇总 |

---

## 六、新贡献者 / agent 的上手顺序

1. 读 [README.md](README.md) 了解安装、快速开始、地址轨迹格式、三后端分工与扩展机制。
2. 读 [REPORT.md](REPORT.md) 的「结论」、第 2 阶段 A/B 以及 R5–R9 章节，掌握现有结论与基线差异。
3. 读本文件第三节认领任务，并查阅 `docs/design/` 中对应的设计文档与 [docs/traces.md](docs/traces.md)。
4. 跑一次冒烟测试 `python3 -m pytest -q tests/test_golden.py tests/test_engine.py tests/test_quickstart.py` 确认环境正常（约 40 秒）。
5. 按第四节协作规则开发、测试、提交，并更新本文件状态。
