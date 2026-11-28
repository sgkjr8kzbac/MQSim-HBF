# 本地测试与实验清单

为了节省云端额度，测试和实验放在本地跑：代码修改在云端分支上完成，你拉到本地运行，把汇总结果提交回分支。分析结果、写报告时，我只需要读这些汇总 CSV。

## 准备

```bash
git fetch origin claude/hbf-kv-study
```

```bash
git checkout claude/hbf-kv-study
```

```bash
git pull origin claude/hbf-kv-study
```

```bash
python3 -m pip install numpy pytest
```

MQSim 相关任务（`tests`、`mqsim`）另外需要 g++、make、git 和 GNU time（`/usr/bin/time`），首次运行会自动编译。

## 一键运行

```bash
evidence/scripts/run_local.sh tests
```

任务名可以连写，比如 `run_local.sh placement interference`，`all` 表示依次跑全部任务。原始运行目录放在 `runs/local/` 下，不需要提交；汇总 CSV 会复制到 `evidence/` 下。环境变量 `JOBS` 控制分析扫描的并行进程数（默认 3），`PLACEMENT_JOBS` 控制分配方式实验的进程数（默认 2）。

| 任务 | 内容 | 时间（参考） | 内存 | 结果文件 | 需要检查什么 |
|---|---|---|---|---|---|
| `tests` | 完整测试 | 约 15 分钟 | MQSim 峰值约 4.2 GB | `evidence/pytest/local-<时间>.log` | 最后一行应为"N passed, 3 xfailed"（3 项 xfail 是已知的 T5/T8 偏差），没有 failed |
| `phase2a` | 第 2 阶段 A 的四组扫描（1,140 个点） | 3 进程约 15 分钟 | 每进程 < 1 GB | `evidence/phase2a/*.csv` | 与仓库中已有的 CSV 一致（`git diff --stat evidence/phase2a`，应无数值变化） |
| `mqsim` | 15 个 MQSim 核对点 | 约 5 分钟 | 4.2 GB，严格单进程 | `evidence/phase2a/mqsim-confirm.csv` | `error` 列都在 −5.5% 到 0.5% 之间 |
| `placement` | E5.1/E5.2：各分配方式的写顺序合法性与读并行度（96 个点） | 2 进程约 15 分钟 | 最大的点约 5 GB/进程 | `evidence/phase2b/placement.csv` | `status` 都是 ok；`layer_major`、`vllm_pool` 在有生成 token 时 `write_legal` 为 False，`dedicated`、`packed` 为 True |
| `interference` | E5.6：写入、刷新、擦除对 TPOT 的影响（960 个组合） | 几分钟 | < 1 GB | `evidence/phase2b/interference.csv` | 没有 inf（inf 表示写入量超出 HBF 的空闲时间，结果也有意义，记下即可） |
| `endurance` | E3.1：寿命解析曲线（8B、70B × 4 种负载 × 3 个读干扰阈值 × 4 种块大小，P/E 上限为列）；E4.1：6 种格式（8B，long32k 与 agentic） | 1–3 分钟 | 小 | `evidence/phase2b/endurance-analytic.csv`、`endurance-formats.csv` | `days_to_limit_*` 列有数值；报告 `runs/local/endurance-bf16/report.md` 的假设列表完整 |

## 跑完以后

```bash
git add evidence
```

```bash
git commit -m "Local runs: <任务名>"
```

```bash
git push origin claude/hbf-kv-study
```

然后告诉我跑了哪些任务。如果有失败，把对应的 `runs/local/<任务>.log` 或测试日志的最后几十行贴给我。

## 说明

- 各个汇总表的列名与含义见 README"运行目录"一节，以及 `hbf/sweep.py` 中 `summarize_run` 的说明。
- 分配方式实验（`placement`）是分析模式的快照，不跑 MQSim。写顺序检查依据 OCP 规范 §5.4.1：块内从第 0 页起按页顺序写；写第 0 页会自动擦除整块。
- 写入干扰实验（`interference`）中的写入量是设定的情景，用来给出影响范围。真实负载下的写入量由寿命模型给出。
- 结果和代码变更出现矛盾时以重新运行为准。修改代码后，我会在本文件里注明哪些任务需要重跑。

## 需要重跑的任务

| 日期 | 原因 | 需要重跑 |
|---|---|---|
| 第 2 阶段 B（一）开始 | 新增分配方式、写入干扰、寿命解析模型；M2 审查修复 | `tests`、`placement`、`interference`、`endurance`（已完成，提交 `fb9631d`） |
| T-B7、T-B2 之后 | `dedicated`、`packed` 默认错开各请求的起点（`--align stagger`），读并行度会变；扫描脚本新增 `"tool": "endurance"` | `placement`、`tests` |
| 事件模拟（T-B1d）合并后 | 新增 `python3 -m hbf lifetime` 的事件模拟（不加 `--analytic`，需要 `--hours`） | `tests`；另跑下面两条验证命令 |

## 事件模拟的验证（T-B1d）

```bash
python3 -m hbf lifetime --workload long32k --concurrency 8 --alloc packed --hours 24 --warmup-h 1 --rd-threshold 1e5,1e6 --output runs/sim-long32k
```

```bash
python3 -m hbf lifetime --workload agentic --concurrency 8 --alloc packed --hours 48 --warmup-h 2 --rd-threshold 1e5,1e6 --output runs/sim-agentic
```

每条约 1–2 分钟。在 `endurance_summary.csv` 里检查：
- `avg_pec_per_day`、`refresh_to_kv` 与同一行的 `analytic_avg_pec_per_day`、`analytic_refresh_to_kv` 对比：long32k 在 X = 1e6 时应相差几个百分点以内；agentic 的刷新比例应接近，调用速率会偏低，因为 TPOT 随批增大。
- `stalls` 为 0，`free_segments_min` 高于低水位。
- `endurance_timeseries.csv` 中 `avg_pec` 在预热后线性增长。

把两个运行目录里的 `endurance_summary.csv` 复制到 `evidence/phase2b/sim-long32k.csv`、`sim-agentic.csv` 后提交。
