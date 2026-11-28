# 交接说明（给本地继续开发的 agent）

云端额度用完前的状态。先读 [PLAN.md](../PLAN.md)，再读本文件。

## 分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `claude/hbf-kv-study` | 第 2 阶段：读路径（A）、放置与写入干扰、寿命模型（B）。REPORT 已写到"第 2 阶段 B（一）" | 事件模拟（T-B1d）大部分完成并已合并（`01c1029`）；未做的 vllm_pool 和 die 分区提速见 `docs/design/phase2b-sim-plan.md` 的 Next steps |
| `claude/hbf-kv-terms` | 第 3 阶段：按 KV 大小公式逐项压缩。从 `claude/hbf-kv-study` 的 7c75e05 切出 | T-K1、T-K2 已完成；已合并 `claude/hbf-kv-study`（含事件模拟），在这个分支上继续开发即可 |

## 第 3 阶段已完成

- **T-K1 页格式**（`hbf/access.py` TensorSpec 的 `attn_dims`、`logical_dims`；`hbf/formats.py` 的 `mla`、`mla_fp8`、DSA 索引器、int/kivi 的 `fused=1[,pow2=1]`；`hbf/pages.py` 的求和）。测试：`tests/test_formats.py` 末尾的 T-K1 部分。
- **T-K2 KV 分组**（新文件 `hbf/kvspec.py`：`KVGroup`、`KVSpec`、`parse_kvspec`、`GroupedLayout`、`generate_grouped`；CLI `--kv-spec`；预设 hybrid、ssm、cla、yoco、duo、pyramid）。测试：`tests/test_kvspec.py`，包括"单个分组与原布局逐页相同"的对照。
- 两项都保持了黄金文件、引擎对照和系统模型测试不变。**两项都还没有经过子代理审查**（计划把它与第 2 阶段 B 的审查 T-B6 一起做）。

## 接下来按顺序做

1. **事件模拟的剩余部分**：按 `docs/design/phase2b-sim-plan.md` 的"Next steps for the local agent"做（先修 die 分区高并发时 `_reclaim`/`_victim` 每个量子空转的问题，再补 vllm_pool 和测试 16）。扫描网格驱动模拟时必须给 `"hours"`。
2. **寿命模型适配新格式**（K1 遗留）：`hbf/endurance.py` 的 `Geometry.demand` 仍按 `spec.dims` 计算元素数，改成 `spec.attended_dims`（注意力元素）和 `spec.logical`（HBF 元素、反量化）；`kv_bytes_per_token` 一列改为只计 HBF 字节（KIVI 现在多算了 HBM 残差）。改之前 MLA 和 fused 格式不能用于 `hbf.endurance`。
3. **审查**（T-B6 + K1 + K2）：按严重程度列问题、附复现方法，再逐条修。
4. **T-K3 模型预设**（`hbf/model.py`、`hbf/system.py`、`hbf/configs/models.json`）：MLA（DeepSeek-V2-Lite）、混合注意力（Gemma-3-27B、GPT-OSS-20B/120B）、SSM 混合（Qwen3-Next），MoE 只计线性层时间；参数对照官方 `config.json`，出处写进 `docs/research/params.md`；`kvspec.from_model(model, fmt)` 由模型的 `layer_pattern` 生成分组。验收值见设计文档 §3 K3。
5. **T-K5 选择的真实性**（`hbf/access.py` 的 generate、`hbf/pages.py`）：存储密度、静态选择、相邻步复用、GQA 并集。验收值见设计 §3 K5。
6. **T-K4、T-K7**（事件模拟合并后）：token 流（写入、死亡、重写）接入寿命模型；分组放置、按类别分写前沿、前缀别名。先修设计 §6 第 7 条（窗口下界为负时写入量算成 0）。
7. **T-K6** 部分通道读。

设计与验收值：[docs/design/phase3-kv-terms.md](design/phase3-kv-terms.md)；定案与 K1 的实现说明：[docs/design/phase3-decisions.md](design/phase3-decisions.md)。

## 本地需要跑的

- `evidence/scripts/run_local.sh tests placement`：T-B7 改变了 `dedicated`、`packed` 的读并行度，放置实验要重跑。
- 第 3 阶段的实验矩阵（设计 §4，X1–X6）还没有网格文件；K1、K2 已能支持 X2（页粒度）和 X3（按头、按层）。可以先写 `evidence/grids/phase3-granularity.json`、`phase3-groups.json`，用 `python -m hbf.sweep` 在本地跑。

## 协作规则（沿用 PLAN.md 第四节）

- 提交说明用英文；仓库文件不写 AI 模型名称。
- 改动默认行为前先确认黄金文件不变：`python3 -m pytest -q tests/test_golden.py tests/test_engine.py`。
- 每个里程碑完成后先审查，再开始下一个。
