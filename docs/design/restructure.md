# 模拟器整理与重构方案

> 执行者：本地 agent。审查：云端 agent（每个里程碑完成后）。
> 起点：分支 `claude/hbf-kv-terms`，提交 `bd3b91c` 之后的这次文档提交。
> 本文件取代 PLAN.md 里所有尚未开始的功能任务。R0–R7 完成前，不再增加新的研究功能。

## 一、已经定下的决定

以下是和用户逐条确认过的边界。执行中遇到冲突时，以这一节为准；拿不准的先问用户，不要自行扩展。

| 问题 | 决定 |
|---|---|
| 最终产出 | 结论优先，其次把代码整理到别人能用：能跑自己的 trace、能加自己的方法 |
| 模拟器本体 | **MQSim-HBF**。Python 轮数模型保留，作为快速通道，用于大范围扫描和把寿命外推到天、年。凡是进 REPORT 的 Python 结果，都要能用 MQSim 抽查 |
| 前后端分工 | 前端（Python）把 trace 经 KV 格式和布局变成**地址轨迹**，这是前后端之间唯一的接口。MQSim 和轮数模型读同一份地址轨迹 |
| 写入规则 | 短期用少量 MQSim 补丁实现：前端决定写什么、写到哪、什么时候刷新，MQSim 负责执行、计时、计数（R5）。把完整的 HBF 写入规则做进 MQSim 的 FTL 是长期目标，不在本方案内 |
| 寿命 | MQSim 跑稳态窗口，输出每块的读、写、擦计数，Python 外推到天、年。Python 事件模拟保留，用 MQSim 窗口的结果核对它（R6） |
| 旧代码和旧结果 | 打 tag 冻结，**不需要兼容**。旧 REPORT 的数字靠 tag 复现。新默认就是 OCP HBF 规范 v0.7.0 的行为 |
| GPU 计算和 TPOT | 作为内置扩展保留，默认关闭 |
| 第 3 阶段的方法 | MLA、kvspec 分组、loki、sparq 等作为内置扩展保留，默认关闭。短期取数据要用，所以不删；README 写明删掉 `hbf/ext/` 不影响模拟器工作 |
| 数据 | 请求级用公共数据集；token 级等有 GPU 后用自己截取的 trace。两种都要能跑 |
| MQSim 修改 | 短期尽量少改。只加 `T-PERF`（提速，不改变结果）和 R5 的写入补丁 |

另一份重构计划（"三轮独立审查终稿"）的处理：第一阶段的 bug 修复并入本方案的 R1；它的 `legacy` 档、三态开关、旧名别名、历史 `meta.json` 重放等兼容设计都不采用，原因是和"不需要兼容"的决定冲突。拆分 `endurance.py` 的思路可以参考，但按本方案的分包放置。

## 二、现状

- **代码规模**：`hbf/*.py` 共 10,927 行。最大的三个文件是 `endurance.py`（3,120 行）、`batch.py`（1,303 行）和 `kvspec.py`（1,098 行）。测试共 5,610 行，分在 18 个文件里。MQSim 有 9 个补丁。
- **选项**：`hbf.cli` 有 74 个命令行选项，`hbf.endurance` 另有 74 个。
- **默认值不是规范行为**：
  - 拓扑默认 `64x16`；
  - HBF 命令模型（`--read-plane-ns 0 --coalesce-ns 1 --cache-read`）要手动打开；
  - 放置默认 `layer_major`，而我们自己的结论是它在 12 个点中合法 0 个。
  
  原因是 PLAN 第四节的"新功能一律默认关闭"规则。
- **同一件事有两套实现**：
  - 读引擎：`analysis.py` 里的 numpy 引擎和纯 Python 参考引擎（`--engine`）；
  - 访问生成：`access.generate` 和 `kvspec.generate_grouped`。
- **依赖缠绕**：`endurance.py` 直接导入 `cli`（用 `phase_a` 跑校准）、`system`、`model`、`kvspec`。核心逻辑和扩展混在一起。
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
  device/                 HBF 器件，不依赖 numpy 以外的东西
    config.py             ← config.py：Config、拓扑、器件参数；默认 ocp
    address.py            静态地址映射（§4.6）：LPA ↔ (channel, die, plane, block, page)
                          收拢现在散在 pages.place、timing.block_ids、batch 里的同一个公式
  frontend/
    trace/
      tokens.py           token 级 trace：读写 access.csv（← access.read_access、write_access）
      synthetic.py        合成模式：dense、uniform_tokens、sink_recent（← access.generate 的通用部分）
      requests.py         请求级 trace：请求流（← workload.py）和数据集导入（R7）
    kv/
      tensors.py          ← access.Shape、TensorSpec、default_tensors、parse_tensors、tensors_meta
      formats.py          核心格式：bf16、int、kivi（← formats.py 的核心部分）
      layouts.py          ← access.Layout 和各布局
      placement.py        多请求放置、写前沿、写序检查（← batch.py）
    addrtrace.py          地址轨迹：数据结构、读写文件、校验（新，前后端唯一接口）
  backends/
    mqsim/                本体
      build.py run.py parse.py   ← simulator.py
      schedule.py         地址轨迹 → MQSim 批和请求（← controller.py）
      （C++ 源码和补丁留在 hbf/mqsim、hbf/patches 原位置）
    rounds/               快速通道
      pages.py analysis.py timing.py   ← 同名文件；只保留 numpy 引擎
  lifetime/
    counts.py             每块计数 → 天、年（新，R6）
    analytic.py clock.py sim/ report.py cli.py   ← endurance.py 拆分
    interference.py       ← interference.py：解析公式，R5 之后作为 MQSim 结果的快速近似
  models.py               模型形状预设：layers、kv_heads、head_dim、q_heads（← model.py 的形状部分）
  ext/                    扩展，默认关闭，可整体删除
    system/               ← system.py、model.py 的计算部分、gpus.json：GPU roofline、TPOT
    kv_terms/             ← kvspec.py、formats.py 的 mla/mla_fp8/fused/indexer、loki/sparq/channel_select、
                             K5 的选择真实性、K7 的分组放置
  registry.py             格式、布局、合成模式、指标的注册表；启动时按需导入 hbf.ext
  run.py                  一次运行：读配置 → 前端 → 后端 → 指标
  cli.py                  子命令：run、sweep、lifetime、check（MQSim 抽查）
  sweep.py
tests/
  reference/              纯 Python 参考引擎，只作为测试里的对照
```

### 3.1 地址轨迹（前后端唯一接口）

前端的所有输出都汇成地址轨迹，两个后端只读它。

| 列 | 类型 | 说明 |
|---|---|---|
| `t_ns` | int | 到达时间；同一 (step, layer) 的请求同时到达，与现在 `controller.schedule` 的批一致 |
| `op` | `R`/`W` | 擦除不单列：按 §5.4.1，写块的第 0 页时器件自动擦除 |
| `lpa` | int | 逻辑页地址，物理位置由 `device.address` 唯一决定 |
| `useful_bytes`、`burst_bytes`、`burst_runs` | int | 现有 `controller.Request` 的三个字段（64 B 读、突发） |
| `step`、`layer`、`request` | int | 归属，用于按层、按步、按请求汇总 |
| `cls` | 枚举 | `kv`、`prefill`、`flush`、`refresh`、`relocate`，用于把写入和擦除归因到来源 |

- 文件格式：`addr.npz`（按列存）加 `addr.json`（器件配置的摘要哈希、布局元数据、生成它的配置）；调试时可导出 CSV。
- 校验在 `addrtrace.validate()`：LPA 越界、块内顺序写（复用 `check_write_order`）、读未写页（§5.4.3，错误 0x7）。默认开启，失败就报错。
- 轮数后端按 (step, layer) 分组，直接从 LPA 算位置，不再从 access 和 layout 重新推导。MQSim 后端经 `schedule.py` 转成 `trace.txt`。
- token 级 trace（access.csv）和请求级 trace（requests.csv）是前端的输入，不是这个接口。

### 3.2 扩展机制

- 核心模块从不在顶层导入 `hbf.ext`。`registry.load_extensions()` 在 `run.py` 启动时逐个导入 `hbf.ext.*`，导入失败（目录被删）时静默跳过。
- 扩展通过注册表加入格式预设、布局、合成模式、kv-spec、额外指标列和 TPOT 后处理。
- 配置里用到某个未注册的名字时，报错要说明它属于哪个扩展。
- 测试 `tests/test_core_without_ext.py`：把 `hbf/` 复制到临时目录并删掉 `ext/`，在其中跑核心冒烟测试（一个 MQSim 运行、一个轮数运行、一个寿命外推）。

功能归属的初始划分如下。判断规则：第 1–5 项的基本结论（不涉及具体压缩方法）需要它，就算核心。

| 核心 | 扩展 |
|---|---|
| bf16、int4/int8、kivi 格式；量化参数张量 | mla、mla_fp8、DSA indexer、fused/pow2 |
| layer_contiguous、striped、staggered、vllm_interleaved 布局 | kvspec 分组（hybrid、ssm、cla、yoco、duo、pyramid）、`from_model` |
| dense、uniform_tokens、sink_recent 合成模式 | density、select-static、reuse、query-groups、full-tensors（K5）、loki、sparq、channel_select（K6） |
| packed、dedicated、vllm_pool、hbm_decode 放置和写序检查 | K7 的分组放置、按类别分写前沿、前缀别名 |
| 寿命：计数外推、解析模型、事件模拟 | K4 的 token 流接入（随 kvspec 进扩展） |
| 模型形状预设 | GPU roofline、TPOT、MoE 线性层时间 |

这张表在 R4 开工前由用户确认一次。

### 3.3 配置与命令行

- 一次运行由一个 TOML 或 JSON 配置文件描述，分为 `[device]`、`[trace]`、`[kv]`、`[placement]`、`[backend]`、`[lifetime]`、`[ext.*]` 几节。
- 命令行只保留常用的覆盖项，例如 `python -m hbf run CONFIG --set kv.format=kivi2`。
- 每次运行把解析后的完整配置写进输出目录的 `config.json`，`meta.json` 只放结果的元数据。
- 默认值全部取规范行为。器件参数规范没有给出的（tR、tPROG、tBERS、块大小、P/E 上限等），在 `device/config.py` 里注明"假设"和出处（`docs/research/params.md`）。

## 四、里程碑

**每个里程碑的通用要求**：
- 开工前读本节对应部分；完成后跑全量测试，日志放进 `evidence/pytest/`。
- 提交说明写清"纯搬移"还是"有意改变行为"，交给云端 agent 审查后再进入下一个。
- 一次只跑一个 1,024 plane 的 MQSim 进程。

### R0 基线

1. 拉取本次提交，跑 `python -m pytest -q`，确认通过数与本文件第六节一致。
2. 打 tag `baseline-pre-refactor` 并推送，旧 REPORT 的数字从此只靠这个 tag 复现。
3. 从这个 tag 切出 `refactor/modular` 分支，重构都在这个分支上做。也可以继续用 `claude/hbf-kv-terms`，由用户决定。

### R1 正确性修复（在旧结构上做，每条一个提交）

| # | 问题 | 改法 | 验收 |
|---|---|---|---|
| 1 | MQSim 完成请求时 O(n²) 查找 | 加 `hbf/patches/T-PERF.patch`（全文见附录 A），追加到 `series` 末尾；`tests/test_plane.py::test_patch_series_lists_every_patch` 的补丁数 9 → 10 | MQSim 测试全部通过（云端在 `6bc6e23` 上验证过：106 项通过、3 项 xfail，唯一失败是补丁计数）；`ocp` 核对点的墙钟约降到原来的 1/3 |
| 2 | UCIe 档位 1 允许 16 个主机通道（§5.1.2.1：档位 1 最多 8 个模块） | `analysis.py` 抽出 `check_protocol(cfg, protocol)`，档位 1 要求 `cfg.host_channels <= 8`，在所有入口调用，包括 numpy 的闭式路径；新增 8 主机通道的 OCP 预设，按 §4.6 推导其 bank 数和核心通道带宽 | 档位 1 配 16 通道报错；用新预设重跑 `evidence/grids/item1-layout-protocol.json` 的档位 1 行，订正 REPORT 第 2 阶段 A 的"稠密读慢 33%"（另一份计划估算约慢 2.7 倍，以重跑结果为准） |
| 3 | `prefill_batches` 只检查 `base` 是单元数的倍数 | 改为块第 0 页对齐：`layout.base % (cfg.units * cfg.pages_per_block) == 0` | 不对齐的 base 报错的测试 |
| 4 | `unwritten_reads` 不知道块被擦除 | 按 (unit, block) 记录已写页；写到某块第 0 页时，把该块重置为只有第 0 页已写 | 先写满一块、再写它的第 0 页、再读第 1 页，应计为读未写页 |
| 5 | `endurance._analytic_rows` 调 `analytic()` 时没有传 `align`（`bd3b91c` 上仍然如此） | 补上 `align=config["align"]` | `--analytic --align none` 与默认结果不同的测试 |
| 6 | T-PLANE 的 independent 规则让写也能按不同页号组成多 plane 命令（§11.2.2 要求同页号）；首个事务命中时没有更新基准 `pageID`/`blockID` | 只有读允许不同页号；修正基准更新 | 两个 plane 写前沿不同时，不合成多 plane 编程命令（构造方法见另一份计划 1.3 节） |
| 7 | `config.load()` 在拓扑覆盖 `aggregate_gb_s` 时报 `TypeError`（关键字参数重复） | 先合并字典，再构造 `Config` | 带 `aggregate_gb_s` 的拓扑预设能加载 |
| 8 | REPORT 第 2 阶段 B 把 `gaps` 写成"只占用 bank 空闲时间" | 改为"按 die 阻塞；优先放进读间隙，放不下的拉长 TPOT" | — |

### R2 新默认 = 规范行为（删除兼容）

1. 修改默认值：
   - 拓扑 `ocp`；
   - HBF 命令模型（`read_plane_ns=0`、`coalesce_ns=1`、`cache_read=True`）；
   - 静态放置；
   - 写序检查默认开启；
   - 多请求放置默认 `packed`；
   - MQSim 运行在多 plane 拓扑上自动 prefill，R5 之后由静态写补丁取代。
2. 删除只为原版 MQSim 或第 0、1 阶段存在的选项：
   - `--engine`：只留 numpy 引擎，参考引擎移到 `tests/reference/`，`tests/test_engine.py` 改为对照它；
   - `--timing`：总是计算；
   - `--placement mqsim`；
   - ONFI 多 plane 命令相关的 `--dbsy-ns` 和 `onfi=` 路径（T-ONFI 补丁保留，只用于第 1 阶段的验证测试）；
   - 合成模式里第 0 阶段的 `--unit`、`--count`、`--hot-fraction`、`--hot-units`（如果没有实验在用）。
   
   删除前用 `grep` 确认 `evidence/` 的网格和脚本没有用到；用到的改写或一并删除。
3. `64x16`、`16x64`、`256x4` 这些"chip 当 plane"的拓扑不再作为命令行预设，移到测试夹具里，供第 1 阶段的 MQSim 验证测试（T3–T10）继续使用。
4. 删除 PLAN 第四节"新功能一律默认关闭"的规则，换成第五节的新规则。
5. 用 `HBF_REGEN_GOLDEN=1` 按新默认重新生成黄金文件，提交说明写明原因。

验收：
- 不带任何选项运行，得到合规的 `ocp` 结果；
- 全量测试通过；
- 选 12 个点（第 2 阶段 A 的 MQSim 核对点和各项主结论各取几个），新代码不加选项的结果，与 tag 上加上相应显式选项的结果一致。

### R3 分包与地址轨迹（纯搬移为主）

按下面的顺序做，每一步一个提交，每一步测试全绿：

1. **建目录、搬文件、改导入**：不保留旧路径的转发模块。`evidence/scripts` 和 `tests` 的导入一起改。
2. **`device/address.py`**：把静态映射公式收拢成一处，`pages.place`、`timing.block_ids`、`batch` 都改为调用它。加一个随机测试：同一 LPA 在三处算出的位置相同。
3. **地址轨迹**：`frontend/addrtrace.py` 实现第 3.1 节的接口。
   - `controller.schedule` 改为"地址轨迹 → MQSim 批"；
   - 轮数后端改为读地址轨迹；
   - `cli.pipeline` 拆成"前端生成地址轨迹"和"后端消费地址轨迹"两步。
   - 验收：固定种子下，搬移前后的 `layers.csv`、`steps.csv`、`analysis_*.csv` 逐字节相同（只允许新增 `config.json`）。
4. **合并两套访问生成**：`access.generate` 和 `kvspec.generate_grouped` 合并（phase3-decisions 第 8 条原定在 K5 之后做）。单个分组时与原布局逐页相同的测试已有，保留。
5. **拆 `endurance.py`**：放进 `lifetime/`。
   - 先纯搬移：固定种子的事件模拟和解析模型 CSV 逐行相同，`test_m5_hook_per_quantum` 的 monkeypatch 目标跟着改。
   - 再切断它对 `cli` 的依赖：校准改为直接调用前端和轮数后端，不再经过命令行。
   - 每个文件尽量不超过 600 行，不能有循环导入（加一个逐个导入所有模块的测试）。
6. **配置文件和子命令**：第 3.3 节。`sweep.py` 的网格改为对配置键做笛卡儿积，网格文件同步改写。
7. **打包**：`pyproject.toml` 改用 `packages.find`；用 wheel 解包确认 `configs/`、`patches/`、`mqsim/` 都在。

### R4 扩展分离

1. 用户确认第 3.2 节的划分表。
2. 实现 `registry.py`，把扩展搬进 `hbf/ext/`。核心代码里与扩展有关的分支改成注册表查找。
3. 加 `tests/test_core_without_ext.py`。
4. README 新增"扩展"一节：列出每个扩展的作用、配置项，并写明删除 `hbf/ext/` 不影响模拟器。

验收：
- 删掉 `ext/` 后核心测试通过；
- 带扩展时，第 3 阶段的测试（`test_kvspec`、`test_selection`、`test_formats` 的 K 部分）全部通过，结果不变。

### R5 MQSim 的 HBF 写入补丁

原则：前端负责策略（写什么、写到哪、何时刷新），MQSim 负责机制（执行命令、计时、计数）。四个补丁都用 XML 参数打开，测试能单独开关它们。

| 补丁 | 内容 | 主要改动位置 | 验收 |
|---|---|---|---|
| `T-HBFW1` 静态写与原地重写 | 读和写都按静态公式从 LPA 算出 PPA（与 `device/address.py` 相同），不再用写前沿和空闲块池；块擦除后可以从第 0 页重新顺序写 | `Address_Mapping_Unit_Page_Level::translate_lpa_to_ppa`、`allocate_page_in_plane_for_user_write`、`online_create_entry_for_reads`；`Flash_Block_Manager::Allocate_block_and_page_in_plane_for_user_write` | 不 prefill 时，I2 输出的物理位置与 Python 静态映射逐页相同（随机合法写序列）；同一块写满、擦除、再写一轮成功 |
| `T-HBFW2` 页 0 自动擦除 | 写到已写过的块的第 0 页时，先对该块发一个擦除事务（复用 `NVM_Transaction_Flash_ER`），再编程 | 写事务进入 TSU 之前，在映射单元或 TSU 中插入 | 手算时序：同 die 的读排在擦除和编程之后，完成时间为 tBERS + tPROG + 读本身的时间；`Issued_Flash_Erase_CMD` 等于预期次数 |
| `T-HBFW3` 关掉 GC 和磨损均衡 | `GC_and_WL_Unit_Page_Level::Check_gc_required` 不触发；静态磨损均衡关闭 | GC 单元 | 写满整个器件也不出现 GC 事务；现有 GC 预算守卫在补丁开启时不再需要 |
| `T-HBFW4` 每块计数 | 每块的读次数、编程次数、擦除次数，运行结束时导出 CSV | `Flash_Block_Manager_Base::Read_transaction_serviced`、`Program_transaction_serviced`、`Block_Pool_Slot_Type::Erase` | 与 Python 从同一份地址轨迹数出的计数逐块相同 |

前端配套：
- prefill、decode flush、刷新搬移的写都进地址轨迹，并标上 `cls`；
- 读干扰刷新和保持期刷新的触发由前端从 trace 算出（每块读次数、经过的时间），插入相应的读、写；
- 写序由 `addrtrace.validate()` 把关。

写入干扰：
- 补丁完成后直接在 MQSim 里测：同一份带写的地址轨迹，对比只读时每层、每步的时间。
- 选 6–10 个点，与 `lifetime/interference.py` 的 `fifo`、`gaps` 模式对比，差异写进 REPORT。
- 规范没有编程、擦除挂起，`T-SUSP` 只作为非规范对照。

### R6 寿命：MQSim 窗口加外推

1. `lifetime/counts.py`：读入 MQSim 窗口的每块计数（`T-HBFW4`）和窗口时长，扣掉预热段，按速率外推每天的 P/E 次数、刷新次数，以及到 P/E 上限的天数。
2. 窗口长度：先用 10、30、100 步比较外推结果，取变化小于 2% 的最短窗口。`T-PERF` 之后，8B、32K、B = 8 稠密读每步约 50 s。
3. 用同一份工作负载跑 Python 事件模拟，对比每天的 P/E 次数和刷新与 KV 写入之比。从 long32k 和 agentic 两个工作负载开始（已有解析值，见 REPORT 第 2 阶段 B（一））。
4. 差异超过 5% 的要查清原因再写进 REPORT。

### R7 trace 输入

1. **请求级公共数据集**：在 `frontend/trace/requests.py` 加导入器，字段统一成 `arrival_s, prompt_tokens, output_tokens[, prefix_id]`。
   - 先做一个：Azure LLM inference trace、BurstGPT 或 Mooncake trace，由用户选；Mooncake 带前缀哈希，可以看前缀共享。
   - 数据集不进仓库，只提交下载脚本、校验和以及字段说明（含许可）。
2. **token 级 trace**：沿用现有 `access.csv` 格式作为外部输入格式，写清字段说明。截取脚本（T-C1，HF transformers 的 hook）等用户有 GPU 后再做，放在 `frontends/`。
3. 验收：一个公共数据集端到端跑通。请求 → 地址轨迹 → MQSim 窗口和轮数估算 → 寿命外推，并和合成工作负载的结果对比。

### R8 文档

- README 按新结构重写：安装、快速开始（默认即规范行为）、配置文件、地址轨迹格式、扩展。
- REPORT 开头加一段说明：`baseline-pre-refactor` 之前的数字来自旧默认，复现方式是检出该 tag。
- PLAN.md 的阶段表和任务表按新结构更新；删除过时的 `docs/handoff.md` 内容。

## 五、重构期间的协作规则（替换 PLAN 第四节的"不改变默认行为"）

- 每个提交要么是**纯搬移**（固定种子的输出逐字节相同，提交说明写 "move only"），要么是**有意改变行为**（提交说明写清改了什么、为什么，必要时重新生成黄金文件）。两种不要混在一个提交里。
- 规范行为一律是默认值。不再为旧行为加开关。
- MQSim 是本体：轮数模型的公式改动，或新的计时方式，要附一组 MQSim 对照。
- 其余规则不变：提交说明用英文，仓库文件不写 AI 模型名称，1,024 plane 的 MQSim 一次只跑一个，规范规则注明章节号，假设的数值标明来源。

## 六、R0 基线记录（云端）

- 提交：本次文档提交，内容为 `bd3b91c` 加上 `tests/test_validation.py::test_tensor_specs_pipeline_and_meta` 的修正。K1 让 `meta.json` 省略未设置的可选字段，K5 又加了 `density`，这个测试的期望值一直没跟上。
- 全量测试（云端，`ce0cc35`，即本方案提交的前一个提交，测试代码相同）：**508 项通过，3 项 xfail**，用时 15 min 10 s，[日志](../../evidence/pytest/pre-refactor.log)。3 项 xfail 与以前相同。本地 R0 跑出的数字应与此一致。

## 附录 A：`T-PERF.patch`

`Waiting_user_requests` 里每个请求只在取进设备时加入一次（`Host_Interface_NVMe.cpp` 的读请求分支和写请求分支各一处，上游第 78、86 行），完成时删除一次。所以"从队头找到第一个就删"和 `std::list::remove` 删的是同一个元素，剩余顺序不变，模拟结果完全相同。云端实测：同样的调用方式下输出逐字节相同，`ocp` 核对点的墙钟从 17.2 s 降到 6.0 s。

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

`RUN_DIR` 是一次 `hbf.cli` 运行的输出目录（含 `device.xml`、`workload.xml`、`trace.txt`）。R3 之后如果要继续提速，先重新剖析，再决定是否去掉页级映射表、NVMe 队列模型、DRAM 数据缓存这些用不到的部分。
