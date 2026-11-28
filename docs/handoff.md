# 交接说明（给本地继续开发的 agent）

先读 [docs/design/restructure.md](design/restructure.md)，它是当前唯一的工作计划。PLAN.md 第三节里尚未完成的功能任务都已暂停。

## 当前状态

| 分支 | 内容 |
|---|---|
| `refactor/modular` | 重构的工作分支：R0–R6 已完成（`6a977f2`）。**在这个分支上继续** |
| `claude/hbf-kv-terms` | 云端写方案的分支，等于 `refactor/modular` 加上方案文档的更新。在 `refactor/modular` 上执行 `git merge --ff-only origin/claude/hbf-kv-terms` 取回 |
| `claude/hbf-kv-study` | 第 2 阶段的旧分支，只作历史参考 |

## 接下来

顺序：**R7 和 R9 并行，R8 最后**。

- **审查**：R4–R6 交给云端审查。
- **R7**：请求级公共数据集。开工前先问用户用哪个数据集。合成的工作负载和合成模式全部保留。
- **R9**：HBFSim 作为第二后端的评估。只读地址轨迹，与 R7 互不影响。做完第 5、6 步后停下来，向用户汇报，由用户决定 HBFSim 的角色。
- **R8**：文档，在 R7、R9 之后做。README 要包括后端一节（含 HBFSim）和请求数据集的获取方式。

每个里程碑单独提交并推送。决定的边界见方案第一节，拿不准的先问用户。
