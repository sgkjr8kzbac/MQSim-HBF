# HBF-EventLab 对照探针

`tests/test_analysis.py` 中协议模型的几项对照数字来自 HBF-EventLab（sjtu-epcc，Apache-2.0，arXiv 2611.19861）。HBF-EventLab 不是本仓库的依赖，这里只保存重跑所需的输入和结果摘要。

- `ocp.ini`：几何与时序。16 个主机通道 × 每通道 4 个 die × 每 die 16 个 bank，4 KiB 交织，感测 4.096 µs，UCIe 档位 3。
- `workloads/`：输入文件。先整页（`--burst`）预写，2 ms 时发出读。

| 文件 | 内容 | 对应测试 |
|---|---|---|
| `pages128.txt`、`pages128-trigger.txt` | lpa 0–127，每主机通道 8 页在 8 个 bank；后者先发一条 64 B 读触发感测 | `test_credit_rounds_match_hbf_eventlab_probe` |
| `bank8x4.txt`、`bank8x4-trigger.txt` | 主机通道 0 的 8 个 bank 各 4 页 | `test_credit_rounds_against_hbf_eventlab_four_pages_per_bank` |
| `banks8.txt` | 主机通道 0 的 8 个 bank 各 1 页 | `test_items_larger_than_mocs_start_their_sense_once`（MOCS 1） |
| `wanted7.txt`、`wanted13.txt` | 7 或 13 个 bank 各读一页的前 2,560 B（40 条 64 B 命令） | 同上（变长项部分放行） |
| `bank0x8.txt`、`bank2x4.txt` | bank 0 上 8 页；bank 0、1 各 4 页 | 复审时的补充探针 |

- `run.sh`：重跑全部探针并输出摘要。统计 JSON 每个可达 30 MB，所以只写到给定的输出目录，不进仓库。

```bash
evidence/eventlab/run.sh /path/to/hbf_eventlab_cli /tmp/eventlab-out > summary.csv
```

- `summarize.py`：从统计 JSON 计算读时间、感测次数、感测波次和 cache 命中数。感测波次的计法：某次感测与本波第一次感测相隔不到半个 tR，就算入同一波。
- `summary.csv`：本仓库保存的那次重跑结果。
