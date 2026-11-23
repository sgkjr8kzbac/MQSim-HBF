# HBF 模拟链路 · 第 0 阶段

把 KV token 区间转换为 4 KiB 页请求，经固定版本的 MQSim 模拟，再还原为子批、层和 decode 步时延。使用原版 CWDP FTL 与一 chip 对应一 HBF plane 的拓扑。实测、偏差和源码位置见 [REPORT.md](REPORT.md)。

## 运行

Linux 上需要 Python 3.10+、g++、make、git、tar 和 GNU time。核心包只使用 Python 标准库。仓库内有未经修改的 MQSim 源码归档，构建无需网络。

```bash
python3 -m hbf.cli --topology 64x16 --mode uniform_pages --seed 234 --fraction 0.1
python3 -m hbf.cli --topology 256x4 --mode dense --layers 32 --heads 8 --context 32768
```

若需安装命令行工具，运行 `python3 -m pip install .`，再用 `hbf-stage0` 运行相同参数。测试可用以下命令。

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q -m 'not scale' --evidence-dir runs/validation
python3 -m pytest -q -m scale --evidence-dir runs/scale
```

每次用一个新目录。省略 `--output` 时，命令在 `runs/` 下生成目录。首次运行会在当前目录的 `.build/` 编译。全量稠密矩阵共提交约 629 万个读请求，通常需要数分钟，单进程设备模拟峰值约 575 MiB。

## 输入和输出

`access.csv` 的表头为 `step,layer,kv_head,kind,op,token_start,token_end,bits`。层和头从 0 开始，prefill 用 step −1，token 区间左闭右开，位宽为 16。用 `--access-csv PATH` 导入已有记录，并通过 `--layers`、`--heads`、`--head-dim`、`--context` 指定形状。默认上下文及预留范围相同。读区间重叠时，有用字节按并集计。写仅支持 prefill 的满页 token 写入。

可选布局为 `layer_contiguous`、`vllm_interleaved`。生成模式包括 `dense`、`uniform_pages`、`same_unit`、`distinct_units`、`same_channel_distinct_units`、`hot_unit` 和 `write_then_read`。定向模式受地址空间约束，形状应覆盖所需页面。生成参数见 `--help`。

运行目录保存输入、形状、种子、参数和版本，MQSim 轨迹、设备和 workload XML、原始 XML 和 stdout、逐请求完成日志，以及子批、层、步指标 CSV 和简报。`--physical` 另外保存物理事件 CSV。层、步的时延是子批服务时间之和，结果表同时保留拆分和未拆分估算。

## 补丁开关

I1 和 I2 的环境变量 `HBF_I1`、`HBF_I2` 指定 CSV 路径。I3 和 I4 由 `HBF_I3`、`HBF_I4` 的存在启用。主链路设置 I1、I3、I4，按需设置 I2。T6 和 ONFI 对照用 Python 接口设置 `simulate(..., suspension=True)` 或 `onfi=True`，分别对应 `HBF_T_SUSP`、`HBF_T_ONFI`。T-ONFI 可通过 XML 覆盖 `ONFI_t_CS`、`ONFI_t_WC`、`ONFI_t_WB`、`ONFI_t_RR`。关闭环境开关时应取消设置变量。
