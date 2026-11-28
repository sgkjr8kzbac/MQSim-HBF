# 模拟器整理与重构方案

> 执行者：本地 agent。审查：云端 agent（每个里程碑完成后）。
> 起点：分支 `claude/hbf-kv-terms`，提交 `bd3b91c` 之后的文档提交。第一版经子代理对照代码审查后修订（第七节）。
> 本文件取代 PLAN.md 里所有尚未开始的功能任务。R0–R8 完成前，不再增加新的研究功能。

## 一、已经定下的决定

以下是和用户逐条确认过的边界。执行中遇到冲突时，以这一节为准；拿不准的先问用户，不要自行扩展。

| 问题 | 决定 |
|---|---|
| 最终产出 | 结论优先，其次把代码整理到别人能用：能跑自己的 trace、能加自己的方法 |
| 模拟器本体 | **MQSim-HBF**。Python 轮数模型保留，作为快速通道，用于大范围扫描和把寿命外推到天、年。凡是进 REPORT 的 Python 结果，都要能用 MQSim 抽查 |
| 前后端分工 | 前端（Python）把 trace 经 KV 格式和布局变成**地址轨迹**，这是前后端之间唯一的接口。MQSim 和轮数模型读同一份地址轨迹 |
| 写入规则 | 短期用少量 MQSim 补丁实现：前端决定写什么、写到哪、什么时候刷新，MQSim 负责执行、计时、计数（R5）。把完整的 HBF 写入规则做进 MQSim 的 FTL 是长期目标，不在本方案内 |
| 寿命 | 用 MQSim 核对 Python 的计数和时序，由 Python 外推到天、年。Python 事件模拟保留（R6 写明了具体做法） |
| 旧代码和旧结果 | 打 tag 冻结，**不需要兼容**。旧 REPORT 的数字靠 tag 复现。新默认就是 OCP HBF 规范 v0.7.0 的行为 |
| GPU 计算和 TPOT | 作为内置扩展保留，默认关闭 |
| 第 3 阶段的方法 | MLA、kvspec 分组、loki、sparq 等作为内置扩展保留，默认关闭。短期取数据要用，所以不删；README 写明删掉 `hbf/ext/` 不影响模拟器工作 |
| 数据 | 请求级用公共数据集；token 级等有 GPU 后用自己截取的 trace。两种都要能跑 |
| MQSim 修改 | 短期尽量少改。只加 `T-PERF`（提速，不改变结果）和 R5 的写入补丁 |

另一份重构计划（"三轮独立审查终稿"，不在仓库里）的处理：
- 第一阶段的 bug 修复并入本方案的 R1，用到的细节已抄进 R1；
- 它的 `legacy` 档、三态开关、旧名别名、历史 `meta.json` 重放等兼容设计都不采用，因为和"不需要兼容"的决定冲突；
- 拆分 `endurance.py` 的思路可以参考，但按本方案的分包放置。

## 二、现状

- **代码规模**：`hbf/*.py` 共 10,927 行。最大的三个文件是 `endurance.py`（3,120 行）、`batch.py`（1,303 行）和 `kvspec.py`（1,098 行）。测试共 5,610 行，分在 18 个文件里。MQSim 有 9 个补丁。
- **选项**：`hbf.cli` 有 71 个命令行选项，`hbf.endurance` 另有 68 个。
- **默认值不是规范行为**：
  - 拓扑默认 `64x16`；
  - HBF 命令模型（`--read-plane-ns 0 --coalesce-ns 1 --cache-read`）要手动打开。
  
  原因是旧规则"新功能一律默认关闭"，该规则已在 PLAN.md 第四节换掉。
- **同一件事有两套实现**：
  - 读引擎：`analysis.py` 里的 numpy 引擎和纯 Python 参考引擎（`--engine`）。注意 numpy 引擎在部分情况下会调用参考实现的函数，见 R2；
  - 访问生成：`access.generate` 和 `kvspec.generate_grouped`。
- **依赖缠绕**：
  - `endurance.py` 直接导入 `cli`（用 `phase_a` 跑校准）、`system`、`model`、`kvspec`；
  - 第 3 阶段的代码嵌在核心文件里，例如 `access.generate`、`pages.layer_pages`、`batch.py`、`cli.py`。
  
  第 3.2 节列出了具体位置。
- **MQSim 慢的原因**：在 `ocp` 上用 gprof 剖析了一次（2 层、128K 上下文，先 prefill 再读，共 52 万次页操作）：

  | 部分 | 占比 |
  |---|---:|
  | `Input_Stream_Manager_NVMe::Handle_serviced_request` 里的 `std::list::remove`，O(n²) | 82% |
  | `AddressMappingDomain` 初始化：1.34 亿个物理页的页级映射表 | 8% |
  | 真正的闪存模拟：调度、通道、chip | 约 5% |

  修复见 R1 的 `T-PERF`。

## 三、目标结构

```
hbf/
  util.py                 写表、解析重叠方式等通用小函数（← system.write_table、system.parse_overlap）
  device/                 HBF 器件
    config.py             ← config.py：Config、拓扑、器件参数、资源路径常量（CONFIG_DIR 等）
    address.py            静态地址映射（§4.6）：LPA ↔ (channel, die, plane, block, page)
                          收拢现在散在 pages.place、timing.block_ids、batch 里的同一个公式
  frontend/
    trace/
      tokens.py           token 级 trace：读写 access.csv（← access.read_access、write_access）
      synthetic.py        合成模式（← access.generate 的通用部分），含第 0 阶段的模式
      requests.py         请求级 trace：请求流（← workload.py）和数据集导入（R7）
    kv/
      tensors.py          ← access.Shape、TensorSpec、default_tensors、parse_tensors、tensors_meta
      formats.py          核心格式：bf16、fp8、int4、int2、kivi4、kivi2、think、quest，以及 fused/pow2 参数
      layouts.py          ← access.Layout 和各布局
      placement.py        多请求放置的快照、写前沿、写序检查（← batch.py 的核心部分）
    timeline.py           请求流 → 按时间展开的 LPA 级读写和释放（新，R6a）
    addrtrace.py          地址轨迹：数据结构、读写文件、校验（新，前后端唯一接口）
  backends/
    mqsim/                本体
      build.py run.py parse.py   ← simulator.py
      schedule.py         地址轨迹 → MQSim 批和请求（← controller.py）
      （C++ 源码和补丁留在 hbf/mqsim、hbf/patches 原位置）
    rounds/               快速通道
      pages.py analysis.py timing.py   ← 同名文件；只保留 numpy 引擎和它依赖的 protocol_rounds、locate
  lifetime/
    clock.py              每步时间的接口：读时间（MQSim 实测或轮数估算）加常数项；TPOT 时钟由扩展提供
    counts.py             每块计数 → 天、年（新，R6）
    analytic.py sim/ report.py cli.py   ← endurance.py 拆分
    interference.py       ← interference.py：解析公式，R5 之后作为 MQSim 结果的快速近似
  models.py               模型形状预设：layers、kv_heads、head_dim、q_heads（← model.py 的形状部分）
  ext/                    扩展，默认关闭，可整体删除
    system/               ← system.py、model.py 的计算部分、gpus.json：GPU roofline、TPOT、prefill 速率
    kv_terms/             ← kvspec.py；mla、mla_fp8、DSA indexer；loki、sparq、channel_select；
                             K5 的选择真实性；K7 的分组放置
  registry.py             格式、布局、合成模式、放置、指标的注册表；启动时按需导入 hbf.ext
  run.py                  一次运行：读配置 → 前端 → 后端 → 指标
  cli.py                  子命令：run、sweep、lifetime、check（MQSim 抽查）
  sweep.py
tests/
  reference/              纯 Python 参考引擎（analyze、analyze_group），只作为测试里的对照
```

### 3.1 地址轨迹（前后端唯一接口）

地址轨迹由三部分组成：
- `addr.npz`：逐页的读写记录；
- `layers.npz`：逐 (step, layer) 的旁表；
- `addr.json`：元数据。

两个后端都只读这三样东西。

**`addr.npz`（逐页）**

| 列 | 类型 | 说明 |
|---|---|---|
| `step`、`layer`、`subbatch` | int | 归属与提交顺序。`subbatch` 对应现在 `controller.schedule` 和 `--batch-pages` 的子批：轮数后端的 `batch_rounds`、`protocol_rounds`、`est_batches_ns` 都按子批求和，MQSim 要求批的到达时间严格递增（`simulator.py:238`） |
| `request` | int | 请求编号（多请求放置时） |
| `op` | `R`/`W`/`D` | `D` 表示释放：这一页的数据作废，不发器件命令，只更新存活状态。擦除不单列：按 §5.4.1，写块的第 0 页时器件自动擦除 |
| `lpa` | int | 逻辑页地址，物理位置由 `device.address` 唯一决定 |
| `useful_bytes`、`burst_bytes`、`burst_runs` | int | 现有 `controller.Request` 的三个字段（64 B 读、突发） |
| `cls` | 枚举 | `kv`、`prefill`、`flush`、`refresh`、`relocate`，用于把写入和擦除归因到来源 |

- **不放到达时间**：到达时间由后端按 (step, layer, subbatch) 的顺序赋值，与现在 `controller.py:97、126` 的做法一致。
- **顺序规则**：同一 (step, layer) 内写在读之前，与 MQSim TSU 同时刻读优先、§5.4.3 的约定相同。

**`layers.npz`（逐 step × layer，每个组合一行，包括没有 HBF 页的层）**
- 需求列：`hbm_bytes`、`attended_tokens`、`attended_elements`、`hbf_elements`、`dequant_ops`、`tensor_tokens`（来自 `pages.py:20-45、199`）。HBM 层的张量和 `op="H"` 的缓存命中不产生页，只出现在这里（`controller.py:71`）。`system.py` 和报告要用这些列。
- 没有 HBF 页的层和步照样输出零行，取代现在的 `_pad_timing_rows`（`cli.py:161`）和 `analysis.py:418` 的特殊处理。

**`addr.json`**
- 器件配置的摘要哈希、布局元数据、生成它的配置；
- 协议选项：MOCS、读突发、主机发射顺序、UCIe 档位、分组感测。这些属于后端的计时选项，但要随轨迹记录，便于复现；
- `resident`：运行开始时已在 HBF 中的 LPA 区间。只读运行默认是整个布局区间，与现在"数据已在 HBF 中"（`cli.py:300-302`）的假设一致。生成了 prefill 写的运行，`resident` 为空。

**校验 `addrtrace.validate()`**（默认开启，失败就报错）
- LPA 越界；
- 读未写页：不在 `resident` 里，也没有被写过（§5.4.3，错误 0x7）；
- 块内顺序写（§5.4.1），按存活规则判断：重写某块第 0 页时，该块所有页都必须已经 `D`（释放）或从未写过，否则就是擦掉了有效数据。

  不能直接复用 `batch.check_write_order`。它把每次写第 0 页都算作 `live_erases`（`batch.py:1237-1249`），刷新和回收的轨迹会被全部判为非法。

**内存**：按 (step, layer) 流式生成和消费，不在内存里拼整份轨迹（placement 网格每个进程约 5 GB）。`sweep` 的 `--skip-access-csv` 同样跳过地址轨迹文件。

token 级 trace（access.csv）和请求级 trace（requests.csv）是前端的输入，不是这个接口。

### 3.2 扩展机制

**注册表**
- 核心模块从不在顶层导入 `hbf.ext`。`registry.load_extensions()` 在 `run.py` 启动时逐个导入 `hbf.ext.*`。只有 `hbf.ext` 本身不存在（ModuleNotFoundError，且模块名就是 `hbf.ext`）时才跳过，其余导入错误照常报出。
- 扩展通过注册表接入的点（挂接点）：
  - 格式预设、布局、合成模式和选择方式；
  - 放置策略；
  - kv-spec 分组；
  - 额外的需求列和指标列；
  - 寿命时钟；
  - 报告段落。
- 配置里用到某个未注册的名字时，报错要说明它属于哪个扩展。

**测试** `tests/test_core_without_ext.py`：把 `hbf/` 复制到临时目录并删掉 `ext/`，在其中跑核心冒烟测试。用小拓扑跑 MQSim，复用 `.build` 缓存；另跑一次轮数估算、一次寿命外推。

**现有代码里要拆开的依赖**（审查时逐条核实过，R4 要全部处理）：

| 位置 | 依赖 | 处理 |
|---|---|---|
| `endurance.py:261、320、353`（Clock） | 用 `system` 算每步时间 | 核心定义 `lifetime/clock.py` 接口（读时间加常数项）；TPOT 时钟由 `ext/system` 注册 |
| `endurance.py:627-640`（`prefill_tok_s`） | 用 `gpu.flops` 和模型参数 | 核心改为接收 prefill 速率参数；按 GPU 计算的版本放进扩展 |
| `endurance.py:221`（`geometry_for`） | 非 GQA 模型时导入 kvspec | 改成挂接点 |
| `endurance.py:451` | 校准读 `system_steps.csv` | 改读轮数后端的每层时间，见 R3.5 |
| `interference.py:219`（`hbf_gaps`） | 导入 `system.parse_overlap` | `parse_overlap` 移到 `util.py` |
| `sweep.py:45` | 顶层导入 `system.write_table` | `write_table` 移到 `util.py` |
| `access.generate`；`pages.py:127、144`（density 和 `H` 分支）；`cli.py:195-197、596-603`（自动填充）；`KVFormat` 的字段 | K5、K6 嵌在核心里 | 改成选择方式和需求列的挂接点 |
| `batch.py` 中 `kv_spec`、`frontiers`、`prefix` 相关的约 128 行 | K7 嵌在核心里 | 改成放置策略的挂接点 |

**功能归属**（判断规则：第 1–5 项的基本结论不涉及具体压缩方法；它们需要用到的功能就算核心）：

| 核心 | 扩展 |
|---|---|
| 格式 bf16、fp8、int4、int2、kivi4、kivi2、think、quest；量化参数张量；int/kivi 的 `fused`、`pow2` 参数 | mla、mla_fp8、DSA indexer |
| 布局 layer_contiguous、striped、staggered、vllm_interleaved | kvspec 分组（hybrid、ssm、cla、yoco、duo、pyramid）、`from_model` |
| 合成模式 dense、uniform_pages、uniform_tokens、sink_recent 及第 0 阶段的模式（`--unit`、`--count`、热点模式） | density、select-static、reuse、query-groups、full-tensors（K5），loki、sparq、channel_select（K6） |
| 放置 packed、dedicated、vllm_pool、hbm_decode 和写序检查 | K7 的分组放置、按类别分写前沿、前缀别名 |
| 寿命：时钟接口、计数外推、解析模型、事件模拟 | K4 的 token 流接入（随 kvspec 进扩展）、TPOT 时钟 |
| 模型形状预设 | GPU roofline、TPOT、MoE 线性层时间 |

这张表在 R4 开工前由用户确认一次。

### 3.3 配置与命令行

- 一次运行由一个 TOML 或 JSON 配置文件描述，分为 `[device]`、`[trace]`、`[kv]`、`[placement]`、`[backend]`、`[lifetime]`、`[ext.*]` 几节。
- 命令行只保留常用的覆盖项，例如 `python -m hbf run CONFIG --set kv.format=kivi2`。
- 每次运行把解析后的完整配置写进输出目录的 `config.json`，`meta.json` 只放结果的元数据。
- 默认值全部取规范行为。
  - HBF 命令模型的参数写进 `ocp` 预设（`hbf/configs/parameters.json`），**不要**改 `Config` 字段的默认值。否则 `Config.__post_init__` 会拒绝 chip 当 plane 的拓扑（`config.py:69-74`），第 1 阶段的验证测试就跑不了。
  - 器件参数规范没有给出的（tR、tPROG、tBERS、块大小、P/E 上限等），在预设里注明"假设"和出处（`docs/research/params.md`）。

## 四、里程碑

**每个里程碑的通用要求**：
- 开工前读本节对应部分；完成后跑全量测试，日志放进 `evidence/pytest/`。
- 提交说明写清"纯搬移"还是"有意改变行为"，交给云端 agent 审查后再进入下一个。
- 一次只跑一个 1,024 plane 的 MQSim 进程。
- 做完 R1、R3 后停下来等审查；R4、R7 开工前先问用户。

### R0 基线

1. 拉取本次提交，跑 `python -m pytest -q`，确认通过数与本文件第六节一致。
2. 打 tag `baseline-pre-refactor` 并推送，旧 REPORT 的数字从此只靠这个 tag 复现。
3. 从这个 tag 切出 `refactor/modular` 分支，重构都在这个分支上做。也可以继续用 `claude/hbf-kv-terms`，由用户决定。

### R1 正确性修复（在旧结构上做，每条一个提交，按编号顺序）

| # | 问题 | 改法 | 验收 |
|---|---|---|---|
| 1 | MQSim 完成请求时 O(n²) 查找 | 加 `hbf/patches/T-PERF.patch`（全文见附录 A），追加到 `series` 末尾。`tests/test_plane.py:176` 的断言同时检查了补丁数（9 → 10）和最后三个补丁的名字，两处都要改 | MQSim 测试全部通过（云端在 `6bc6e23` 上验证过：106 项通过、3 项 xfail，唯一失败就是这条断言）；`ocp` 核对点的墙钟约降到原来的 1/3 |
| 2 | `config.load()` 在拓扑预设覆盖 `aggregate_gb_s` 时报 `TypeError`（关键字参数重复） | 先合并字典再构造：`replace(Config(**(params.pop("calibration") \| params \| fields)), **overrides)` | 带 `aggregate_gb_s` 的拓扑预设能加载。第 3 条的新预设依赖这一条 |
| 3 | UCIe 档位 1 允许 16 个主机通道（§5.1.2.1：档位 1 最多 8 个模块，合计 0.384 TB/s；REPORT 第 350 行引用过） | ① `analysis.py` 抽出 `check_protocol(cfg, protocol)`，档位 1 要求 `cfg.host_channels <= 8`，在 `protocol_rounds()`、`analyze()`、`analyze_pages()`（紧挨 `check_host_channels`）和 `cli.configure()` 调用，numpy 的闭式路径 `_protocol_part()` 也要经过它。② 新预设 `ocp-8ch`：`channels=16, chips=2, planes=16, address_map="ocp", plane_rule="independent", aggregate_gb_s=512`（512 bank，每条核心通道仍是 32 GB/s）。③ `cli.topology_name()` 区分 16 和 8 个主机通道 | 档位 1 配 16 通道报错。以下测试和网格要改用 `ocp-8ch` 或档位 2：`test_engine.py:54`、`test_timing.py:53`、`test_sweep.py:28-34`、`evidence/grids/item1-layout-protocol.json`（否则档位 1 那一行会在扫描里被静默记成不可行）、`evidence/scripts/protocol_credit_scan.py`。用新预设重跑后订正 REPORT 第 2 阶段 A 的"稠密读慢 33%"，**两种对比基准都写出来**：同一 8 通道几何下的档位 2/3，以及 16 通道 `ocp`。寿命模型里 `Device(banks=1024)` 和 `blocking_units(dies=64)` 写死了 16 通道，`ocp-8ch` 暂时不能用于寿命模型，在 REPORT 的局限里写明 |
| 4 | `prefill_batches` 只检查 `base` 是单元数的倍数 | 改为块第 0 页对齐：`layout.base % (cfg.units * cfg.pages_per_block) == 0`。同时改 `controller.py:104-110` 的 docstring：MQSim 的数据写前沿依次用块 0、3、4……（块 1、2 留给 translation 和 GC 写前沿，`Flash_Block_Manager_Base.cpp:53-57`），所以在 R5 的 `T-HBFW1` 之前，MQSim 的物理块号本来就不等于静态块号；这里只保证块内页偏移一致 | 不对齐的 base 报错的测试 |
| 5 | `unwritten_reads` 不知道块被擦除 | 改为 `unwritten_reads(cfg, batches)`：按 (unit, block) 记录已写页，写到某块第 0 页时把该块重置为只有第 0 页已写；批内先查读、再按写更新。调用处 `cli.py:294` 和 `simulator.py:225` 一起改 | 先写满一块、再写它的第 0 页、再读第 1 页，应计为读未写页 |
| 6 | `endurance._analytic_rows`（`endurance.py:2800-2817`）调 `analytic()` 时没有传 `align` | 补上 `align=config["align"]` | `--analytic --align none` 与默认结果不同的测试 |
| 7 | T-PLANE 的 independent 规则（`hbf_joins_command` 的 case 1）让写和擦除也能按不同页号合成多 plane 命令；§11.2.2 要求同页号 | 只有读允许不同页号；写和擦除保持上游规则（同块号、同页号）。另一份计划还提到"首个事务命中时没有更新基准 pageID/blockID"，审查发现它只在队首是带 RelatedRead 的写时才会出现（`TSU_Base.cpp:75-77`），目前走不到，不改，在补丁注释里记一句 | 负例测试：MQSim 的 FTL 为每个 plane 维护独立的写前沿，所以先在批 0 只写 plane 0，把它的前沿推到第 1 页（plane 1 仍在第 0 页）；再在批 1 同时写 plane 0 和 plane 1。断言 `Issued_Flash_Multiplane_Program_CMD == 0`、`Issued_Flash_Program_CMD == 3`。这个测试的写序本身不合法，需要绕过写序检查 |
| 8 | REPORT 第 2 阶段 B（第 730 行附近）把 `gaps` 写成"只占用 bank 空闲时间" | 改为"按 die 阻塞；优先放进读间隙，放不下的拉长 TPOT" | — |

### R2 新默认 = 规范行为（删除兼容）

**改默认值**
- `ocp` 预设加上 HBF 命令模型：`read_plane_ns=0`、`coalesce_ns=1`、`cache_read=True`。写在预设里，见第 3.3 节。
- 命令行和配置的默认拓扑改为 `ocp`。
- 放置方式本来就默认 `static`，不用改。
- 单请求、多 plane、不导入 `--access-csv` 的 MQSim 运行自动 prefill。
- **多请求放置的默认值暂时不改。**
  - 原因：现在只要配了多请求放置（`BatchSpec`），就只能用 `--analysis-only`，而且不能加 `--prefill`、`--access-csv`（`cli.py:655-657`），K5 的选择方式和 density < 1 也与它互斥（`cli.py:206-212`）。默认改成 `packed` 会让所有 MQSim 运行报错。
  - 等 R5 打通"放置 → 地址轨迹 → MQSim"之后，再把默认改为 `packed`（R5 第 6 步）。
- `endurance.phase_a` 只在策略不是 `layer_major` 时才传 `--alloc`（`endurance.py:440`），改为总是显式传。
- 寿命校准缓存的键里没有代码版本（`endurance.py:483-488`），改默认后会复用旧结果。把键里的 `version` 加 1。

**删除的命令行选项**（只删命令行选项；`Config` 里的 `onfi`、`dbsy_ns` 等字段保留，T-ONFI、T-PLANE 的测试还在用，见 `test_validation.py:308-314`、`test_plane.py:27-40`）
- `--engine`：只留 numpy 引擎。
  - 移到 `tests/reference/` 的只有 `analyze`、`analyze_group`；
  - `protocol_rounds`、`locate` 和 `controller.Request` 留在轮数后端：`_protocol_part` 在 `group != 1` 或额度不均（`wanted` 读）时会调用它们（`analysis.py:378-381`）；
  - `tests/test_engine.py` 改为对照 `tests/reference/`；
  - `evidence/scripts/protocol_credit_scan.py:4`、`selection_granularity.py:5` 导入了 `analysis.analyze`，改用 `analyze_accesses`，或按下面"脚本"一条冻结。
- `--timing`：总是计算。`endurance.py:434`、`sweep.py:118` 和 evidence 的网格、脚本里传这个选项的地方一起删。
- `--placement mqsim`。
- `--dbsy-ns`。

**保留的选项**（第一版方案曾列为删除，审查发现仍在使用）
- `--count`：它就是 sink_recent 的 sink token 数（`access.py:682`），`item2-compression.json:6` 和 `test_golden.py:29` 都在用；
- `--unit` 和第 0 阶段的合成模式：T3–T10 的验证测试在用（`test_validation.py:126-163、436`，`test_timing.py:26、80`）。
- 命名拓扑 `64x16`、`16x64`、`256x4` 也保留，只是不再作为默认值。`parse_topology` 本来就接受任意 CxN（`config.py:286-291`），删掉命名没有意义；`test_validation.py:105` 还会检查 `configs/{topology}.xml`。

**脚本和文档**
- `time_calibration.py:25-29` 用当前解析器重放旧运行的 `argv`，R2 之后结果会变。
- 决定：只为复现旧 REPORT 数字服务的脚本（`time_calibration.py`、`phase2a_tables.py`、`protocol_credit_scan.py`、`selection_granularity.py`、`interference_scan.py`），在文件开头注明"只在 tag `baseline-pre-refactor` 上使用"，不再维护。
- 继续使用的 `mqsim_confirm.sh`、`run_local.sh`、`evidence/grids/*.json`、`docs/local-runs.md` 和 README 的命令，在 R3.6 换成新命令行时一起改写。

**黄金文件**：用 `HBF_REGEN_GOLDEN=1` 按新默认重新生成，提交说明写明原因。

**验收**
- 不带任何选项运行（单请求），MQSim 和分析模式都得到合规的 `ocp` 结果；
- 全量测试通过；
- 选 12 个点（第 2 阶段 A 的 MQSim 核对点和各项主结论各取几个，都是 tag 上能运行的组合），新代码不加选项的结果，与 tag 上加上相应显式选项的结果一致。

### R3 分包与地址轨迹

每一步一个提交，每一步测试全绿。第 1、5 步是纯搬移，第 2、3、6、7 步提交说明里写清行为是否变化。

1. **先收拢资源路径，再搬文件**。
   - 现在 `config.py:8-10`、`interference.py:50`、`model.py:7`、`workload.py:26`、`endurance.py:2610` 各自按 `__file__` 找 `configs/`，`simulator.py` 按 `PACKAGE` 找 `mqsim/` 和 `patches/`。先统一成 `device/config.py` 导出的常量。
   - 然后建目录、搬文件、改导入，不保留旧路径的转发模块。`evidence/scripts` 和 `tests` 的导入一起改。
   - `system.write_table`、`system.parse_overlap` 移到 `util.py`。
2. **`device/address.py`**：把静态映射公式收拢成一处，`pages.place`、`timing.block_ids`、`batch` 都改为调用它。
   - 测试对照标量版本 `Config.device_lpa` / `physical`，以及 MQSim I2 补丁输出的物理位置（prefill 后的随机读）。
3. **地址轨迹**：`frontend/addrtrace.py` 实现第 3.1 节的接口。
   - `controller.schedule` 改为"地址轨迹 → MQSim 批"；
   - 轮数后端改为读地址轨迹和 `layers.npz`；
   - `cli.pipeline` 拆成"前端生成地址轨迹"和"后端消费地址轨迹"两步。
   - 验收：固定种子下，搬移前后的 `layers.csv`、`steps.csv`、`analysis_*.csv`、`system_*.csv`、`batches.csv`、`report.md` 逐字节相同。只允许新增 `addr.npz`、`layers.npz`、`addr.json` 三个文件。
4. **（移到 R4）** 合并两套访问生成。原因：R4 要把 kvspec 移进扩展，提前把它合并进核心会自相矛盾。
5. **拆 `endurance.py`**：放进 `lifetime/`。
   - 先纯搬移：固定种子的事件模拟和解析模型 CSV 逐行相同。`test_m5_hook_per_quantum` 用 monkeypatch 替换 `endurance.interference`，替换目标要跟着搬到新模块。
   - 再切断它对 `cli` 的依赖：校准改为直接调用前端和轮数后端，读每层时间，不再读 `system_steps.csv`。
   - 加入 `lifetime/clock.py` 的时钟接口。
   - 每个文件尽量不超过 600 行，不能有循环导入。加一个测试，逐个导入所有模块。
6. **配置文件和子命令**（第 3.3 节）。验收：
   - 写出旧入口到新子命令的对照表：`hbf.cli`、`hbf.endurance`、`hbf.sweep`，以及 `hbf.system`、`hbf.interference` 的 `__main__`；
   - 所有用 `main([...])` 直接传选项的测试都改成新形式；
   - README 快速开始里的每条命令由一个测试实际运行；
   - `evidence/grids/*.json` 改写成配置键的笛卡儿积，`mqsim_confirm.sh`、`run_local.sh`、`docs/local-runs.md` 换成新命令。
7. **打包**：`pyproject.toml` 改用 `packages.find`；用 wheel 解包确认 `configs/`、`patches/`、`mqsim/` 都在。

### R4 扩展分离

1. 用户确认第 3.2 节的功能归属表。
2. 实现 `registry.py` 和第 3.2 节列出的挂接点，把扩展搬进 `hbf/ext/`。第 3.2 节"要拆开的依赖"表里的每一行都要处理掉。
3. **合并两套访问生成**：核心的 `generate` 通过挂接点支持分组（kvspec 注册到核心）。
   - 如果随机数的抽取顺序变了，这个提交算"有意改变行为"；
   - 保留"单个分组与原布局逐页相同"的测试；
   - 另用固定种子比较每层的选中 token 数和页数分布。
4. 加 `tests/test_core_without_ext.py`。
5. README 新增"扩展"一节：列出每个扩展的作用、配置项，并写明删除 `hbf/ext/` 不影响模拟器。

验收：
- 删掉 `ext/` 后核心测试通过；
- 带扩展时，第 3 阶段的测试（`test_kvspec`、`test_selection`、`test_formats` 的 K 部分、`test_batch` 的 K7 部分）全部通过。

### R5 MQSim 的 HBF 写入补丁

原则：前端负责策略（写什么、写到哪、何时刷新），MQSim 负责机制（执行命令、计时、计数）。

四个补丁都和现有补丁一样用 `HBF_*` 环境变量打开（`simulator.py:263-270`），测试能单独开关它们。

| 补丁 | 内容 | 要改的位置 | 验收 |
|---|---|---|---|
| `T-HBFW1` 静态写与原地重写 | 读和写都按静态公式从 LPA 算出 PPA（与 `device/address.py` 相同），不用写前沿和空闲块池；块擦除后可以从第 0 页重新顺序写 | `Address_Mapping_Unit_Page_Level::translate_lpa_to_ppa`、`allocate_page_in_plane_for_user_write`、`online_create_entry_for_reads`；`Flash_Block_Manager::Allocate_block_and_page_in_plane_for_user_write`。另外：<br>① `Store_mapping_table_on_flash_at_start` 会把 translation 页写进写前沿块（`Address_Mapping_Unit_Page_Level.cpp:423-440`），静态模式下关掉；<br>② `Invalidate_page_in_block` 会检查块的 `Stream_id`（`Flash_Block_Manager.cpp:113-118`），从未分配过的块会报错；<br>③ `Free_pages_count == 0` 时 `Check_bookkeeping_correctness` 报错（`Flash_Block_Manager_Base.cpp:122-128`） | 不 prefill 时，I2 输出的物理位置与 Python 静态映射逐页相同（随机合法写序列）；同一块写满、擦除、再写一轮成功；写满整个布局区间不报错（逻辑容量扣了 7% OP，见 `config.py:162-167`，"写满"指布局区间，不是整个器件） |
| `T-HBFW2` 页 0 自动擦除 | 写到已写过的块的第 0 页时，先对该块发一个擦除事务（`NVM_Transaction_Flash_ER`），编程要等擦除完成才能发出 | ① TSU 按读、写、擦的顺序服务（`TSU_Base.h:93-100`），擦除进 `GCEraseTRQueue`（`TSU_Priority_OutOfOrder.cpp:268-269`）。不加依赖的话，编程会先于擦除发出，所以要改 `transaction_is_ready`（`TSU_Base.h:103-116`）。<br>② 擦除完成后的处理要新写：来源为 USERIO 的擦除，现在会直接 PRINT_ERROR（`GC_and_WL_Unit_Base.cpp:47-60`）；走 GC_WL 路径又会把块放回空闲池（`:130-143`），两种都不对。<br>③ `simulator.py:285-286` 一见擦除队列或 GC 队列有事务就报错（`gc_enqueued` 统计的就是 GC_Erase 队列，`:119`），写入模式下要改成只检查 GC | 见下方"读和写的先后" |
| `T-HBFW3` 关掉 GC | `GC_and_WL_Unit_Page_Level::Check_gc_required` 不触发。静态磨损均衡已经在 XML 里关掉（`config.py:307-308`），不用改 | 写满布局区间并反复重写，不出现 GC 事务；补丁开启时不再需要 `check_gc_budget` |
| `T-HBFW4` 每块计数 | 每块的读次数、编程次数、擦除次数，运行结束时导出 CSV。`Block_Pool_Slot_Type::Erase` 只在 `Add_erased_block_to_pool` 里调用（`Flash_Block_Manager.cpp:135-146`），所以 `T-HBFW2` 的擦除要单独计数 | `Flash_Block_Manager_Base::Read_transaction_serviced`、`Program_transaction_serviced`，以及 `T-HBFW2` 的擦除完成处理 | 与 Python 从同一份地址轨迹数出的计数逐块相同 |

**读和写的先后**：这直接决定写入干扰的结果，要单独定，做成开关 `HBF_WRITE_ORDER`，两种都实现。
- `read_first`：MQSim 现有行为，同 die 上已排队的读先于擦除、编程执行，对应 `interference.py` 的 `gaps` 思路；
- `fifo`：按到达顺序执行，对应 `fifo` 模式。

验收用写明到达时刻的手算用例。设擦除和编程在 t0 发往 die d，一个读在 t0 + δ（0 < δ < tBERS）到达同一 die：
- `fifo` 下读的完成时间为 t0 + tBERS + tPROG + 读本身的时间；
- `read_first` 下为 t0 + tBERS + 读本身的时间，编程排在读之后。

另加一例：读在 t0 之前到达，两种模式下读都先完成。

**其他**
- T-CACHE 遇到写和擦除时保持上游行为，这种组合从没测过，加一个测试。
- `simulate` 的子批重叠检查（`simulator.py:297`）会挡住"带写轨迹对比只读"的测量，写入模式下要放开。
- 规范没有编程、擦除挂起，`T-SUSP` 只作为非规范对照。

**前端配套**
1. prefill、decode flush、刷新搬移的写都进地址轨迹，标上 `cls`。数据作废时写 `D`。
2. 读干扰刷新和保持期刷新的触发由前端从 trace 算出（每块读次数、经过的时间），插入相应的读、写和释放。
3. 写序由 `addrtrace.validate()` 把关。
4. 写入干扰直接在 MQSim 里测：同一份轨迹，带写和只读各跑一次，对比每层、每步的时间。选 6–10 个点，与 `interference.py` 的 `fifo`、`gaps` 模式对比，差异写进 REPORT。
5. 多请求放置的轨迹能交给 MQSim 之后，去掉 `cli.py:655-657` 的"只能分析模式"限制。
6. 第 5 步做完后，把多请求放置的默认值改为 `packed`（R2 推迟到这里），并重新生成黄金文件。

### R6 寿命：用 MQSim 核对 Python，由 Python 外推

**为什么不能直接用 MQSim 窗口外推**
- 窗口太短，刷新还没发生：第一次读干扰刷新要等 X / (ppb × share) 步（`endurance.py:644-646`），X = 1e5、ppb = 256 时是 390 步。long32k 每个请求输出 338 个 token，并发 8 时 100 步内只有约 2 个请求完成。
- 窗口没有物理时间基准：每步时间（TPOT）来自 `system.evaluate`（`endurance.py:313-321、347-354`），属于扩展；MQSim 配置里的 `interval_ns` = 100 ms 也不是物理时间。

**做法**
1. **R6a 时间线前端**（`frontend/timeline.py`）：把请求流展开成按时间排列的 LPA 级操作，包括 prefill 写、decode flush、请求结束时的释放（`D`）、刷新和搬移。
   - 现在没有这样的模块：`batch.py` 只做快照（`batch.py:1-10`），事件模拟按段而不是按页工作（`endurance.py:1114-1120`）。
   - 写法上复用事件模拟的策略逻辑，按页展开。
   - 输出就是地址轨迹，可以交给 MQSim。
2. **R6b 缩小参数核对**：把读干扰阈值 X、每块页数 ppb、请求长度按比例缩小，使刷新和请求更替都在 MQSim 能跑完的窗口内发生，窗口取 max(阈值步数, 请求寿命步数) 的数倍。在同一组缩小参数下：
   - MQSim（`T-HBFW1–4`）输出每块的读、写、擦计数和每步时间；
   - Python 事件模拟输出同样的量；
   - 两者的每块计数应相同，每天的 P/E 次数、刷新与 KV 写入之比的差异应在 5% 以内。超过的要查清原因再写进 REPORT。
3. **R6c 全尺寸外推**：核对通过后，由 Python 事件模拟在真实参数下算每天的 P/E 次数和到 P/E 上限的天数。
   - 每步时间来自 `lifetime/clock.py`：读时间取 MQSim 实测或轮数估算，加上一个常数项（GPU 计算时间作为参数）；
   - 打开 `ext/system` 时，改用 TPOT 时钟。
4. `lifetime/counts.py` 负责从 MQSim 每块计数算速率，供 R6b 对比用。
5. 从 long32k 和 agentic 两个工作负载开始，已有解析值（见 REPORT 第 2 阶段 B（一））。

### R7 trace 输入

1. **请求级公共数据集**：在 `frontend/trace/requests.py` 加导入器，字段统一成 `arrival_s, prompt_tokens, output_tokens[, prefix_id]`。
   - 先做一个：Azure LLM inference trace、BurstGPT 或 Mooncake trace，由用户选；Mooncake 带前缀哈希，可以看前缀共享。
   - 数据集不进仓库，只提交下载脚本、校验和以及字段说明（含许可）。
2. **token 级 trace**：沿用现有 `access.csv` 格式作为外部输入格式，写清字段说明。截取脚本（T-C1，HF transformers 的 hook）等用户有 GPU 后再做，放在 `frontend/trace/capture/`。
3. 验收：一个公共数据集端到端跑通。请求 → 时间线 → 地址轨迹 → MQSim 和轮数估算 → 寿命外推，并和合成工作负载的结果对比。

### R8 文档

- README 按新结构重写：安装、快速开始（默认即规范行为）、配置文件、地址轨迹格式、扩展。
- REPORT 开头加一段说明：`baseline-pre-refactor` 之前的数字来自旧默认，复现方式是检出该 tag。
- PLAN.md 的阶段表和任务表按新结构更新；删除过时的 `docs/handoff.md` 内容。

## 五、重构期间的协作规则（PLAN.md 第四节已同步）

- 每个提交要么是**纯搬移**（固定种子的输出逐字节相同，提交说明写 "move only"），要么是**有意改变行为**（提交说明写清改了什么、为什么，必要时重新生成黄金文件）。两种不要混在一个提交里。
- 规范行为一律是默认值。不再为旧行为加开关。
- MQSim 是本体：轮数模型的公式改动，或新的计时方式，要附一组 MQSim 对照。
- 其余规则不变：
  - 提交说明用英文；
  - 仓库文件不写 AI 模型名称；
  - 1,024 plane 的 MQSim 一次只跑一个；
  - 规范规则注明章节号；
  - 假设的数值标明来源。

## 六、R0 基线记录（云端）

- 提交：`ce0cc35`，内容为 `bd3b91c` 加上 `tests/test_validation.py::test_tensor_specs_pipeline_and_meta` 的修正。K1 让 `meta.json` 省略未设置的可选字段，K5 又加了 `density`，这个测试的期望值一直没跟上。
- 全量测试：在 `ce0cc35` 上跑（之后的提交只改文档，测试代码相同），**508 项通过，3 项 xfail**，用时 15 min 10 s，[日志](../../evidence/pytest/pre-refactor.log)。3 项 xfail 与以前相同。本地 R0 跑出的数字应与此一致。

## 七、审查记录

第一版方案（`dda83aa`）交给一个只读子代理，对照代码逐条核对。R1 的 8 个问题全部属实。主要修订如下：

| # | 级别 | 问题 | 修订 |
|---|---|---|---|
| 1 | 阻断 | R2 把多请求放置默认改为 `packed`，会让所有 MQSim 运行和导入 trace 的运行报错（`cli.py:655-657`） | 推迟到 R5 第 6 步；`phase_a` 总是显式传 `--alloc` |
| 2 | 阻断 | 把参考引擎整体移走会弄坏 numpy 引擎（`analysis.py:378-381` 会调用 `protocol_rounds`） | 只移走 `analyze`、`analyze_group` |
| 3 | 阻断 | 地址轨迹缺子批、需求列、零行、存活规则和驻留集合，R3 的逐字节验收做不到 | 第 3.1 节重写：`subbatch` 列、`layers.npz`、`D` 操作、`resident`；验收文件清单更正 |
| 4 | 阻断 | `T-HBFW2` 不改 TSU 依赖的话，编程会先于擦除发出；擦除完成后的处理和 `simulator.py` 的守卫都会报错；验收的期望值没写到达时刻 | 补上改动位置，读写先后做成 `HBF_WRITE_ORDER` 开关，用写明到达时刻的手算用例验收 |
| 5 | 阻断 | 窗口太短，刷新和请求更替都没发生；窗口没有时间基准；缺"请求流 → LPA 级时间线"的模块 | R6 改为 R6a 时间线前端、R6b 缩小参数核对、R6c Python 全尺寸外推 |
| 6 | 重要 | 删掉扩展后核心有具体依赖（寿命时钟、`prefill_tok_s`、`parse_overlap`、`write_table`、K5/K6/K7 嵌在核心文件里）；划分表漏了格式，还写了不存在的 int8 | 第 3.2 节加"要拆开的依赖"表和挂接点；划分表按 `formats.PRESETS` 更正 |
| 7 | 重要 | R2 要删的选项有的仍在使用（`--count`、`--unit`、`--timing`）；HBF 命令模型不能放进 `Config` 的默认值；校准缓存要升版本；旧脚本要么改写要么冻结 | R2 重写：删除清单、保留清单、预设、版本号、脚本处理 |
| 8 | 重要 | R1 的顺序（第 2 条依赖第 7 条）、测试波及、对比基准、引用了不在仓库里的计划、走不到的缺陷 | R1 重新编号并补全 |
| 9 | 重要 | `T-HBFW1/3/4` 漏掉的修改位置 | 写进 R5 的表 |
| 10 | 重要 | R3 的路径定位、合并生成器与 R4 的矛盾、R3.6 没有验收 | R3.1 先收拢路径；生成器合并移到 R4；R3.6 补验收 |
| 11–15 | 建议 | 注册表的异常处理、测试用小拓扑、地址映射测试的对照对象、流式处理、目录名 | 均已采纳 |

审查没有核实的：规范原文（§5.1.2.1、§11.2.2，以及按 §4.6 推导的 8 通道几何）、gprof 各部分的占比和提速数字。gprof 和提速是云端实测的，见第二节和附录 A；规范原文请在 R1 第 3、7 条动手前对照 PDF 再看一遍。

## 附录 A：`T-PERF.patch`

`Waiting_user_requests` 里每个请求只在取进设备时加入一次（`Host_Interface_NVMe.cpp` 的读请求分支和写请求分支各一处，上游第 78、86 行），完成时删除一次。所以"从队头找到第一个就删"和 `std::list::remove` 删的是同一个元素，剩余顺序不变，模拟结果完全相同。

- 云端实测：同样的调用方式下输出逐字节相同，`ocp` 核对点的墙钟从 17.2 s 降到 6.0 s。
- 完成顺序大致是先进先出，所以通常很快就能找到。最坏情况下，每次完成仍是 O(n)。

```diff
diff --git a/src/ssd/Host_Interface_NVMe.cpp b/src/ssd/Host_Interface_NVMe.cpp
index 0406e6b..cdba641 100644
--- a/src/ssd/Host_Interface_NVMe.cpp
+++ b/src/ssd/Host_Interface_NVMe.cpp
@@ -1,3 +1,4 @@
+#include <algorithm>
 #include <stdexcept>
 #include "../sim/Engine.h"
 #include "Host_Interface_NVMe.h"
@@ -98,7 +99,8 @@ inline void Input_Stream_Manager_NVMe::Handle_arrived_write_data(User_Request *r
 inline void Input_Stream_Manager_NVMe::Handle_serviced_request(User_Request *request)
 {
 	stream_id_type stream_id = request->Stream_id;
-	((Input_Stream_NVMe *)input_streams[request->Stream_id])->Waiting_user_requests.remove(request);
+	{ std::list<User_Request*>& w = ((Input_Stream_NVMe *)input_streams[request->Stream_id])->Waiting_user_requests;
+	  std::list<User_Request*>::iterator it = std::find(w.begin(), w.end(), request); if (it != w.end()) w.erase(it); }
 	((Input_Stream_NVMe *)input_streams[stream_id])->On_the_fly_requests--;
 
 	DEBUG("** Host Interface: Request #" << request->ID << " from stream #" << request->Stream_id << " is finished")
```

补丁文件要以换行结尾。加入 `series` 后，`build()` 会因为补丁哈希变化自动重新编译。

## 附录 B：剖析 MQSim 的方法

```bash
B=$(ls -d .build/*/ | head -1); cp -r "$B" /tmp/mqprof && cd /tmp/mqprof && make clean
make -j4 CC_FLAGS="-std=c++11 -O3 -g -pg" LD="g++ -pg"
cd RUN_DIR && /tmp/mqprof/MQSim -i device.xml -w workload.xml < /dev/null && gprof -b -p /tmp/mqprof/MQSim gmon.out | head -30
```

`RUN_DIR` 是一次 `hbf.cli` 运行的输出目录（含 `device.xml`、`workload.xml`、`trace.txt`）。直接这样调用时没有 `hbf.cli` 设置的 `HBF_*` 环境变量，结果会和经过 `hbf.cli` 的运行略有不同；比较改动前后时，两次都用同一种调用方式。R3 之后如果要继续提速，先重新剖析，再决定是否去掉页级映射表、NVMe 队列模型、DRAM 数据缓存这些用不到的部分。
