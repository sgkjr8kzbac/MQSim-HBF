# HBF 模拟链路第 0 阶段验证报告

## 结论

逻辑访问、页布局、开环轨迹、MQSim 和逐请求结果还原已贯通。单页读校准为 **4,471 ns**。同一 chip 上不同 die 的读仍按 **4,470 ns** 的器件周期串行。稠密基线每层读 32,768 页，三种拓扑中 `layer_contiguous` 的实测服务时间为 **738.448、288.080、273.088 µs/层**。同样的读量使用 `vllm_interleaved` 后，每层只使用 32 个单元，服务时间升到 **4.579–6.609 ms/层**。

T5 中将 PCIe 加倍，每个 8,192 页子批缩短 **64 ns（0.089%）**。吞吐与估算一致、队列深度对照通过，严格相等预期记为 xfail。T9 的 256×4 连续布局实测时间比估算器 1 高 **57.2%**，物理轨迹显示同通道感测就绪后的传输排队。公式和基线参数都保留任务定义。

## 交付与可复现性

| 路径 | 内容 |
|---|---|
| `hbf/access.py` | 逻辑访问 CSV、形状校验、两种布局、七种生成模式 |
| `hbf/config.py`、`hbf/configs/` | Python 参数、三套设备和 workload XML、互校验、CWDP 与地址上界 |
| `hbf/controller.py` | 区间求并集、页请求去重、子批调度、两个估算值与负载指标 |
| `hbf/simulator.py`、`hbf/cli.py` | 固定版本构建、运行、按到达时间和 LBA 对回完成日志、聚合和简报 |
| `hbf/vendor/`、`hbf/patches/` | 上游未修改的源码归档、SHA-256、I1–I4、T-SUSP、T-ONFI |
| `tests/`、`evidence/`、`runs/` | pytest、随 Git 保存的摘要、当前工作区的完整运行记录 |

固定上游 commit 为 `51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16`，tree 为 `c5cbd1a1e75c61f2020273ff415ebf40f9446a1c`。构建时在缓存目录解包再应用补丁。g++ 13.3.0、C++11、Python 3.12、pytest 9.2.0。本机用原版设备 XML 和自带 `tpcc-small.trace` 场景验证了未打补丁的 MQSim，生成并完成 6,999 个请求，用时 2.44 s。每次正式运行的 `version.json` 记录当时的仓库 commit、二进制及补丁散列和开关状态。可复现命令见 [README.md](README.md)。上游来源见 [MQSim 固定版本](https://github.com/CMU-SAFARI/MQSim/commit/51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16)。

## 设备与工作负载配置

设备参数采用任务给定的 SPLASH 单堆栈基线。带宽沿用题设的 16×64、64×16、256×4，三组标称均为 **1,024 GB/s**。带宽及吞吐以十进制 GB/s 计，容量的 GiB 以二进制计。 [SPLASH 论文](https://arxiv.org/abs/2609.23816)

| 参数 | 最终取值 | 理由 |
|---|---|---|
| `Flash_Channel_Count × Chip_No_Per_Channel` | 16×64、64×16、256×4 | 扫描每通道 chip 数，共 1,024 个单元 |
| `Die_No_Per_Chip`、`Plane_No_Per_Die` | 1、1 | 对应 HBF plane，T0 另用 2 die 验证 |
| `Flash_Channel_Width`、`Channel_Transfer_Rate` | 64、16、4 B，rate 各为 1000 | 1 B 对应 1 GB/s，4 KiB 均可被两倍通道宽度整除 |
| `Page_Capacity`、扇区、`Page_No_Per_Block`、`Block_No_Per_Plane` | 4,096 B、512 B、256、64 | 页请求为 8 扇区，设备 64 GiB 物理容量 |
| `Flash_Technology` | SLC | 只使用 LSB 时序 |
| `Page_Read_Latency_*` | 4,096 ns | 感测时延 |
| `Page_Program_Latency_*`、`Block_Erase_Latency` | 100,000、1,000,000 ns | 写入和擦除基线 |
| `PCIe_Lane_Bandwidth × PCIe_Lane_Count` | 256 GB/s × 16 | 为 NAND 留出主机传输余量 |
| `IO_Queue_Depth`、`Queue_Fetch_Size` | 65,535、65,535 | uint16 上界，支持大规模开环发射 |
| `Plane_Allocation_Scheme`、调度器 | CWDP、PRIORITY_OUT_OF_ORDER | 固定上游 FTL 与调度基线 |
| `Ideal_Mapping_Table`、`Enabled_Preconditioning` | true、false | 映射命中，首次读由原版 FTL 分配页 |
| `Overprovisioning_Ratio`、`GC_Exec_Threshold` | 0.07、0.05 | 保留空闲块，触发下限为 10 块 |
| 静态和动态磨损均衡、`Preemptible_GC_Enabled` | false | 保持读基线可重复 |
| `CMD_Suspension_Support` | NONE | T6 独立使用 PROGRAM |
| `Suspend_Program_Time`、`Preferred_suspend_write_time_for_read` | 100,000 ns、100,000 ns | 保留样例值，T6 为 5,000 ns、0 ns |
| flow、设备缓存、重放、执行比例 | 单 flow、TURNED_OFF、1、100% | 覆盖全部 ID，页请求直达 FTL |
| 子批上限、开环间隔 | 8,192 页、100 ms | 抑制大批运行时间，逐批检查完成时刻 |
| 默认补丁 | I1、I3、I4；I2 按需 | 记录完成时间、修复 ID 解析、批处理，主结果关闭 T 类补丁 |

单 flow 的可用逻辑容量为 `1024 × floor(64 × 256 × 8 × 0.93) = 124,821,504` 扇区，低于 2 TiB。控制器只允许完整页区域内的 LPA。上游 NVMe 拆包在 `Host_Interface_NVMe.cpp` 的 `Request_Fetch_Unit_NVMe::Process_pcie_read_message` 把高位移 31 位，主机打包在 `IO_Flow_Base::NVMe_read_sqe` 所在文件把地址右移 32 位。2 TiB 上限使扇区地址高位字为零。

## T0 校准

| 物理阶段或用例 | 实测 |
|---|---:|
| 发命令到命令完成 | 290 ns |
| 命令完成到 R/B 就绪 | 4,116 ns＝4,096 ns 感测＋20 ns |
| 64 GB/s 通道页传输 | 64 ns |
| 单页请求主机侧残差 | 1 ns |
| 单页读 | 4,471 ns |
| 同 chip 4 页 | 17,881 ns，命令间隔 4,470 ns |
| 同 chip 两 die 各一页 | 8,941 ns，命令间隔 4,470 ns，交错读命令 0 |
| 同通道 4 chip 各一页 | 5,341 ns，命令间隔均为 290 ns |
| 不同通道 4 chip 各一页 | 4,474 ns，4 个命令同时发出 |

宽度 64 GB/s 的参考传输时间与单页结果已写入参数 JSON。三组拓扑的估算器使用 64、256、1,024 ns 的数据传输时间，由参考值和通道宽度得到。单页 1 ns 主机残差作为估算器 1 的常数。并发时主机侧残差可累积，例如四通道同时返回时尾延迟多 3 ns。

## 预审事实逐条复核

1. **并行单元是 chip。** T0 两 die 都被寻址，命令相隔一个完整的页周期，`Issued_Flash_Interleaved_Read_CMD=0`。两个调度器的 `service_read_transaction` 检查 chip 状态，`NVM_PHY_ONFI_NVDDR2::Execute_simulator_event` 在最后一次读传输完成后将 chip 置为 IDLE。
2. **ONFI 启动和就绪延迟。** `ONFI_Channel_NVDDR2::ONFI_Channel_NVDDR2` 用 `20 + 6×25 + 100 + 20` ns 计算命令时间。R/B 的 20 ns 默认值来自 `Flash_Chip.h` 的构造参数。原版 XML 无对应字段。独立 T-ONFI 对照把命令改为 90 ns，单页读随之变为 4,271 ns。
3. **同 chip 的感测与传输串行。** T0、T3 的后继命令与上一读页数据完成在同一时刻，T3 的斜率为 4.470 µs/页。相关路径为 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip`、`transfer_read_data_from_chip` 和 `Execute_simulator_event`。
4. **传输率取整。** `SSD_Device::SSD_Device` 中 tRC 为 `2 × floor(1000/rate)` ns，`ONFI_Channel_NVDDR2.h` 的 `NVDDR2DataOutTransferTime` 每两个通道宽度计一次。宽度 64 B 时，rate 500、501、750、1000、1001 实测为 128、64、64、64、0 ns，rate 1001 正常退出。
5. **program 挂起。** 两个 `service_read_transaction` 都复现原版 fallthrough。首次只补 `break`，挂起计数仍为 0。进一步核对两个派生构造函数和 `TSU_Base::TSU_Base`，发现三项时间阈值和两个布尔开关传参错序。修正后可以发挂起命令，但原版 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip` 在普通单 die program 感测期没有更新 chip 级挂起账本，读完成也无法驱动写恢复。T-SUSP 开关内同时修复转参及单 die 挂起／恢复账本，并由 T6 检查读和写都完成。调度基线保留原有 chip 门控。
6. **静默错误与 GC。**
   - `Device_Parameter_Set.h` 两个队列字段为 uint16。单独设为 65,536 的两个原版 XML 探针均正常退出且读命令为 0。T5 把 fetch 降到 512，吞吐从 465.904 降到 401.450 GB/s。
   - `IO_Flow_Parameter_Set::XML_deserialize` 原版用 1,000 字节数组和 `strncpy`。320 个 ID 时，指定通道 319 的 LPA 实测落到 42，开启 I3 后落到 319。256 个紧凑 ID 的串长为 913 字节，该拓扑的最高编号也已用 T1 实测。
   - `IO_Flow_Trace_Based::Start_simulation` 在非五字段行停止预扫描，三行输入的中间行畸形时只完成第一条。源码检查时间戳非降序。设置 MICROSECOND 的探针提交时间仍为 100、123 ns。
   - `Address_Mapping_Unit_Page_Level::online_create_entry_for_reads` 调用 `Flash_Block_Manager::Allocate_block_and_page_in_plane_for_user_write`。两条首次读依次占用物理页 0、1，重读第一个 LPA 仍访问页 0。块写满时推进写前沿并检查 GC。
   - `GC_and_WL_Unit_Base::GC_and_WL_Unit_Base` 与 `GC_and_WL_Unit_Page_Level::Check_gc_required` 确认为空闲池低于 `max(floor(GC_Exec_Threshold×块数), 10)` 时检查 GC，候选块还要满足安全和可回收条件。`(64−10)×256=13,824` 是粗略预算，预留写前沿会占用块，实际临界页数未用压满实验核实。OP 由 `Logical_Address_Partitioning_Unit::Allocate_logical_address_for_flows` 用于逻辑容量计算。T7 实测验证所选工作负载未执行 GC。
   - 保留 `main` 中原版 `cin.get()` 行为，基线样例通过 EOF 退出。批量链路使用 I4 和 EOF。
7. **大批的墙钟时间超线性。** 本机从 8,192 页 0.314 s 升至 32,768 页 3.871 s。64K 一次提交 16.794 s，拆成八批 1.516 s。绝对值与预审机器不同，趋势一致。设备物理页数为 16,777,216，峰值 RSS 除以这一容量约 35 B/物理页。
8. **其余行为。** 三处 CWDP 已对照 `Address_Mapping_Unit_Page_Level::allocate_plane_for_preconditioning`、`allocate_plane_for_user_write`、`online_create_entry_for_reads`，并用 T1 直接核对首次读位置。 `IO_Flow_Trace_Based::Execute_simulator_event` 按轨迹时间发射，T6 读在写完成前进入队列。 `Data_Cache_Manager_Flash_Advanced::process_new_user_request` 在 TURNED_OFF 下调用 FTL 地址翻译。理想映射的映射页读数在 T7 为 0。 `SSD_Device::SSD_Device` 的 SLC 分支使用 LSB，改动 CSB/MSB 的探针保持 4,116 ns 感测加就绪延迟。 `SSD_Defs.h` 定义扇区为 512 B。 `PCIe_Link::estimate_transfer_time` 的 TLP 负载占比为 128/(128+28)≈82.1%，将主机链接降至 16 GB/s 后，端到端有效率实测 80.6%，包括请求控制流量和设备启动时间。

## T0–T10 验收

| 项 | 预期 | 实测 | 判定 |
|---|---|---|---|
| T0 | 拆解单页时延，复现 chip、die、通道并行特性 | 290＋4,096＋20＋64＋1 ns。两 die 串行，交错读为 0 | 通过 |
| T1 | 三种拓扑中高编号与指定物理位置一致 | 包含 (15,63)、(63,15)、(255,3)，全部精确命中 | 通过 |
| T2 | 单页匹配校准 | 4,471 ns，估算 4,471 ns | 通过 |
| T3 | 同单元读按完整页周期线性递增 | N=1、2、4、8、16、32、64，精确符合 `4470×N+1` ns | 通过 |
| T4 | 横跨通道近似平坦，同通道随 chip 数出现阶梯 | 256×4 扫描 N=1 到 1,024。1／256／257／512／1,024 页为 5.431／5.686／6.455／6.710／8.758 µs。同通道由 4.471 升至 26.005 µs，平均额外 341.81 ns/chip，参考 354 ns | 通过 |
| T5 | 均匀子批吞吐与估算一致，PCIe 加倍保持相同 | 6×8,192 页，465.904 GB/s，时延比 1.0305。fetch 32,768 逐批相同，fetch 512 降速。PCIe 加倍后 466.319 GB/s | 吞吐通过，严格相等 xfail |
| T6 | program 期间同单元不同 LPA 的读挂起对照 | 两调度器相同。关／开读为 99.897／9.566 µs，挂起命令 0／1，写正常完成于 100.426／109.991 µs | 通过 |
| T7 | 擦除、GC 入队、映射页读均为 0，读命令数等于页数 | T3、T4、T5 的所有运行均满足 | 通过 |
| T8 | 三档随机稀疏，时延比 0.8–1.25，热点量级合理 | 1%／10%／50% 为 336／3,280／16,384 页，时延比分别 1.112／1.147／1.089–1.143，最忙单元 3／10／24 页，理论量级 3／9／25 | 通过 |
| T9 | 跑完六组 32 层稠密矩阵，验证映射冲突与两种估算 | 每组 1,048,576 页，GC 为 0。连续布局实测接近预审的 16×64 和 256×4 值，估算器在 256×4 有明显低估 | 通过，估算偏差保留 |
| T10 | 测五档规模的资源和单位页耗时，给出子批建议 | 见性能表，64K 附加拆分实验支持 8K 选择 | 通过 |

T4 在前 256 页每个通道只处理 1 页，255 ns 的尾部增长来自主机侧返回序列化。当 N 跨越 256 的倍数时，每通道增加一轮命令与数据竞争，造成阶梯。T0 直接测得同通道的命令启动错开 290 ns。单通道多 chip 的阶段变化反映命令和数据共享通道的调度。

T8 按每个 (head, K/V) 无放回抽样，向上取整页数。用每单元 32 个候选页的二项近似估算最忙单元，取 `1024×P(X≥k)≥1` 的最大 k。同头内的无放回相关性使该统计只作为量级对照。50% 拆成两个 8,192 页子批，时延比按各子批核验，最大负载按完整层统计。

## T9 稠密矩阵

32 层、8 个 KV 头、head_dim=128、BF16，上下文 32,768。每页 16 个 token，每层 128 MiB，每步 4 GiB。每层四个 8,192 页子批，表中的层时延为四个子批 makespan 之和，步时延为 32 层之和。两种估算都逐子批计算后求和。每组 32 层的最小、平均、最大在本轮相同。

| 拓扑 | 布局 | 最忙单元页 | MQSim µs/层 | MQSim 估算 µs/层 | 理想下界 µs/层 | MQSim/估算 | 估算−理想 µs/层 | MQSim ms/步 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16×64 | layer_contiguous | 32 | 738.448 | 724.996 | 131.072 | 1.019 | 593.924 | 23.630 |
| 64×16 | layer_contiguous | 32 | 288.080 | 279.556 | 131.072 | 1.030 | 148.484 | 9.219 |
| 256×4 | layer_contiguous | 32 | 273.088 | 173.764 | 131.072 | 1.572 | 42.692 | 8.739 |
| 16×64 | vllm_interleaved | 1,024 | 4,578.760 | 4,577.284 | 4,194.304 | 1.000 | 382.980 | 146.520 |
| 64×16 | vllm_interleaved | 1,024 | 4,776.136 | 4,773.892 | 4,194.304 | 1.000 | 579.588 | 152.836 |
| 256×4 | vllm_interleaved | 1,024 | 6,608.960 | 5,560.324 | 4,194.304 | 1.189 | 1,366.020 | 211.487 |

估算器 1 取最忙单元页数乘串行页读周期与最忙通道页数乘每页通道占用的较大值，再加 1 ns。理想下界取最忙单元感测时间与总字节除聚合带宽的较大值。连续布局均匀铺满 1,024 个单元，理想步时延为 4.194 ms。交错布局每层的 16 个 (head, kind) 偏移按 512 LPA 步长推进，模 1,024 只能遍历两组偏移，于是每层 32 个活跃单元各承担 1,024 页，理想步时延升至 134.218 ms。

256×4 的通道页传输为 1,024 ns。第一个连续布局子批中，通道 0 的 32 页有 24 页在 R/B 就绪后排队，平均等待 1,101 ns，最大 2,202 ns。交错布局的通道 0 则有 512 页，其中 256 页等待，平均 367 ns。`NVM_PHY_ONFI_NVDDR2::handle_ready_signal_from_chip` 把通道忙时就绪的读放入等待队列。这段等待会推迟 chip 进入 IDLE，随后才能启动同 chip 的下一条命令。估算器 1 只取两个独立占用时间的最大值，无法表达这一相互制约。

子批间隔只用于实验隔离。对照 T10 的 64K 开环发射和 8×8K 拆分，同组地址的服务时间分别为 561.492 和 576.160 µs，拆分高出约 2.61%，因为每批单独经历启动和排空。T9 的整批公式也保存在 `layers.csv`，本矩阵的公式拆分差别为 3 ns/层，来自额外三次主机常数。T9 的未拆分 MQSim makespan 未测。

## T10 性能

64×16，均匀读，关闭 I2。以下墙钟时间为一次 MQSim 子进程的启动、模拟和原始输出总时间。单批每百万页耗时按本次页数线性折算，包含一次固定启动成本。

| 单次提交页数 | 墙钟 s | 峰值 RSS KiB | 折算 s/百万页 |
|---:|---:|---:|---:|
| 1,024 | 0.264 | 568,756 | 257.87 |
| 4,096 | 0.214 | 570,772 | 52.18 |
| 8,192 | 0.314 | 573,304 | 38.36 |
| 16,384 | 0.965 | 578,568 | 58.91 |
| 32,768 | 3.871 | 588,920 | 118.12 |
| 65,536 同时到达 | 16.794 | 609,936 | 256.25 |
| 65,536 分 8 批 | 1.516 | 573,684 | 23.13 |

建议默认每子批 8,192 页。一次进程可承载多个子批，复用设备初始化。64K 同时到达探针受 NVMe 队列上限约束，有少量请求在主机等待入队。墙钟时间对机器、负载分布和批次顺序敏感，特别是大量页堆在少数 chip 的交错布局。

## 局限与下一阶段接口

- MQSim 主结果沿用 SSD 的 chip 门控、固定 ONFI 命令和 PCIe/NVMe 协议。当前模型无法表达 NAND cache read 或单 plane 感测和传输重叠。理想估算给出相应的吞吐下界模型，T-ONFI 用于独立研究四项命令时序。
- 轨迹按时间戳开环发射，层、步的服务时间通过事后相加重建。GPU 计算、层间动态依赖和真实模型决策反馈仍待接入。
- 真实模型可逐步导出 `Access` 决策表及形状 meta。以同一接口实现 KIVI、StreamingLLM、Quest 适配器。新增位宽、量化参数开销与块级选择语义时，布局应明确每个张量占用的页与有效字节。
- decode 追加写适合先在 HBM 缓冲，攒够完整页后提交给 HBF。对未满页保留 token 范围和有效位图，待下一阶段定义尾页 flush、写可见性、覆盖及读后写顺序。当前完整页 prefill 写可以用作控制对照。

## 结果文件

本次验证合计 **35 项通过，1 项 xfail**。所有六组稠密实验在最终补丁版关闭 T 类开关后复跑，层时延与首次测量逐组一致。结果表与关键运行版本见 [T9 结果](evidence/t9.csv)、[T10 结果](evidence/t10.csv) 和 [验证证据索引](evidence/README.md)。
