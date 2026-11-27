# 实测数据摘要

`t9.csv` 汇总六个稠密实验。`t10.csv` 和 `t10-split-summary.json` 来自独立的连续性能测试。`validation/` 保留微基准和原版行为探针的主要数值，`pytest/` 保留测试结束状态。`versions.json` 记录代表性测试运行实际使用的源码和二进制版本。

完整记录保存在当前工作区的 `runs/checks/`、`runs/final-t9-*/`、`runs/performance-final/` 和 `runs/diagnostic-*/`。克隆仓库后可用 README 中的命令在本地重建这些记录。

第二轮审查修改：`t8-topologies.csv` 是三种拓扑 × 三档的 T8 时延比与通道项/单元项之比，`t9-unsplit.csv` 是单层一次提交与 4×8K 拆分的对照，`pytest/review-round2.log` 是对应的测试日志。

第 1 阶段：`stage1-analysis-sweep.csv` 是机制分析的示例扫描（Llama-3.1-8B 形状、32K、均匀随机选页、种子 11，1,024 与 256 个 plane，BF16 与 2 bit，两种布局，独立与同偏移联动），由 `hbf/analysis.py` 生成，不经过 MQSim。

实验排查：`selection-granularity.csv` 比较按页抽样（`uniform_pages`）与按 16-token 块选择（`uniform_tokens`），`ocp`、32 层、32K，BF16、2 bit 与 KIVI 格式，比例 0.1%–5%，每格 5 个种子（11–15）的平均、最小和最大轮数，由 `hbf/analysis.py` 生成；`mqsim-vs-rounds.csv` 是 `ocp` 上 MQSim 每层时延与分析轮数的对照（2 层、32K、`--fraction 0.05`、T-CMD + T-CACHE）。`protocol-credit-scan.csv` 已按修正后的协议模型重算。

`eventlab/`：HBF-EventLab 对照探针的输入、配置、重跑脚本和重跑结果摘要，见其中的 README。`scripts/`：`protocol_credit_scan.py` 和 `selection_granularity.py` 生成同名的 CSV（`python3 evidence/scripts/<脚本> <输出.csv>`）。

第 2 阶段 A：`time-calibration.csv` 用当前代码重算第 8 轮保存的 79 个 MQSim 层（`ocp` 和 `Cx1 --planes 16/64`，独立与联动、开关 cache read、合并窗口 1–3,000 ns、384–3,072 GB/s）的读时间估算，与 MQSim 实测逐层对照；`hbf_command_model` 列标出是否为校准所用的 HBF 命令模型（`--read-plane-ns 0`、有合并窗口、cache read）。由 `scripts/time_calibration.py RUN_DIR...` 生成；原始运行目录在当时会话的临时目录中，未入库。

`phase2a/`：第 2 阶段 A 的四个扫描汇总（`item1-device.csv`、`item1-system.csv`、`item1-layout-protocol.csv`、`item2-compression.csv`，网格见 `grids/`，表格由 `scripts/phase2a_tables.py` 生成），以及 15 个 MQSim 核对点逐层与估算的对照 `mqsim-confirm.csv`（`scripts/mqsim_confirm.sh` 运行，`scripts/time_calibration.py` 对照）。
