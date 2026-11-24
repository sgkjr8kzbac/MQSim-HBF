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

| 拓扑 | 设备 XML | Workload XML |
|---|---|---|
| 16×64 | [16x64.xml](hbf/configs/16x64.xml) | [16x64-workload.xml](hbf/configs/16x64-workload.xml) |
| 64×16 | [64x16.xml](hbf/configs/64x16.xml) | [64x16-workload.xml](hbf/configs/64x16-workload.xml) |
| 256×4 | [256x4.xml](hbf/configs/256x4.xml) | [256x4-workload.xml](hbf/configs/256x4-workload.xml) |

设备 XML 以 MQSim 样例为模板，按下表设置 HBF 基线及其余沿用的参数。中间件从参数 JSON 读取拓扑、几何、时序和控制器基线，运行时生成并保存实际使用的设备和 workload XML。T0、T5、T6 等用例的覆盖值可在各自的运行目录查阅。 [设备模板](hbf/configs/template.xml) · [参数 JSON](hbf/configs/parameters.json) · [配置生成与校验](hbf/config.py)

| 参数 | 最终取值 | 理由 |
|---|---|---|
| `Seed` | 321 | 沿用样例并固定 MQSim 随机状态。逻辑访问生成器的种子单独保存在每次运行的 meta 中 |
| `Memory_Type`、`HostInterface_Type`、`Flash_Comm_Protocol` | FLASH、NVME、NVDDR2 | 固定 NAND 器件、PCIe/NVMe 主机路径和 ONFI NVDDR2 通道模型 |
| `Flash_Channel_Count × Chip_No_Per_Channel` | 16×64、64×16、256×4 | 扫描每通道 chip 数，共 1,024 个单元 |
| `Die_No_Per_Chip`、`Plane_No_Per_Die` | 1、1 | 对应 HBF plane，T0 另用 2 die 验证 |
| `Flash_Channel_Width`、`Channel_Transfer_Rate` | 64、16、4 B，rate 各为 1000 | 1 B 对应 1 GB/s，4 KiB 均可被两倍通道宽度整除 |
| `Page_Capacity`、扇区、`Page_No_Per_Block`、`Block_No_Per_Plane` | 4,096 B、512 B、256、64 | 页请求为 8 扇区，设备 64 GiB 物理容量 |
| `Flash_Technology` | SLC | 只使用 LSB 时序 |
| `Page_Read_Latency_*` | 4,096 ns | 感测时延 |
| `Page_Program_Latency_*`、`Block_Erase_Latency` | 100,000、1,000,000 ns | 写入和擦除基线 |
| `Page_Metadat_Capacity`、`Block_PE_Cycles_Limit` | 448 B、10,000 次 | 沿用样例的页元数据和耐久性参数，本阶段只扫描布局和时序 |
| `PCIe_Lane_Bandwidth × PCIe_Lane_Count` | 256 GB/s × 16 | 为 NAND 留出主机传输余量 |
| `SATA_Processing_Delay` | 400,000 ns | 保留样例设置，基线使用 NVMe |
| `Enable_ResponseTime_Logging`、`ResponseTime_Logging_Period_Length` | false、1,000,000 ns | 关闭周期响应日志，保留样例周期，逐请求结果由 I1 提供 |
| `IO_Queue_Depth`、`Queue_Fetch_Size` | 65,535、65,535 | uint16 上界，支持大规模开环发射 |
| `Address_Mapping`、`Plane_Allocation_Scheme`、`Transaction_Scheduling_Policy` | PAGE_LEVEL、CWDP、PRIORITY_OUT_OF_ORDER | 使用上游页级 FTL，固定 CWDP 分配和调度基线 |
| `Ideal_Mapping_Table`、`Enabled_Preconditioning` | true、false | 映射命中，首次读由原版 FTL 分配页 |
| `CMT_Capacity`、`CMT_Sharing_Mode` | 2,097,152 B、SHARED | 沿用样例，映射访问由理想映射设置控制 |
| `Caching_Mechanism`、`Data_Cache_Sharing_Mode`、`Data_Cache_Capacity` | ADVANCED、SHARED、268,435,456 B | 保留样例的缓存管理器配置，基线 flow 关闭数据缓存 |
| `Data_Cache_DRAM_Row_Size`、`Data_Cache_DRAM_Data_Rate`、`Data_Cache_DRAM_Data_Busrt_Size`、`Data_Cache_DRAM_tRCD`、`Data_Cache_DRAM_tCL`、`Data_Cache_DRAM_tRP` | 8,192、100、1、13、13、13（XML 原值） | 沿用样例的缓存 DRAM 参数，基线 flow 绕过缓存 |
| `Overprovisioning_Ratio`、`GC_Exec_Threshold` | 0.07、0.05 | 保留空闲块，触发下限为 10 块 |
| `GC_Block_Selection_Policy`、`GC_Hard_Threshold`、`Use_Copyback_for_GC` | RGA、0.005、false | 沿用样例的 GC 候选选择和阈值。T7 核对基线的 GC 队列保持为空 |
| `Static_Wearleveling_Enabled`、`Dynamic_Wearleveling_Enabled`、`Preemptible_GC_Enabled` | false、false、false | 关闭磨损均衡和可抢占 GC，固定读基线 |
| `Static_Wearleveling_Threshold` | 100 | 保留样例阈值，静态磨损均衡已关闭 |
| `CMD_Suspension_Support` | NONE | T6 独立使用 PROGRAM |
| `Suspend_Program_Time`、`Preferred_suspend_write_time_for_read` | 100,000 ns、100,000 ns | 保留样例值，T6 为 5,000 ns、0 ns |
| `Suspend_Erase_Time`、`Preferred_suspend_erase_time_for_read`、`Preferred_suspend_erase_time_for_write` | 各 700,000 ns | 保留样例的擦除挂起参数，T6 仅对照写入挂起 |
| flow、`Priority_Class`、`Channel_IDs`、`Chip_IDs`、`Die_IDs`、`Plane_IDs` | 单 flow、HIGH。通道和 chip 各从 0 覆盖至该拓扑上限，die 和 plane 均为 0 | 一个优先级的请求流覆盖全部并行单元 |
| `Device_Level_Data_Caching_Mode`、`Initial_Occupancy_Percentage` | TURNED_OFF、0% | 页请求直达 FTL，从空设备开始 |
| `File_Path`、`Time_Unit`、`Relay_Count`、`Percentage_To_Be_Executed` | trace.txt、NANOSECOND、1、100% | 每次运行提交完整的纳秒轨迹一次 |
| 子批上限、开环间隔 | 8,192 页、100 ms | 抑制大批运行时间，逐批检查完成时刻 |
| 默认补丁 | I1、I3、I4；I2 按需 | 记录完成时间、修复 ID 解析、批处理，主结果关闭 T 类补丁 |

单 flow 的可用逻辑容量为 `1024 × floor(64 × 256 × 8 × (1.0 − 0.07)) = 124,821,504` 扇区，低于 2 TiB。页容量、扇区大小、每块页数及每 plane 块数须为正整数。容量计算与 `src/utils/Logical_Address_Partitioning_Unit.cpp` 中的 `Logical_Address_Partitioning_Unit::Allocate_logical_address_for_flows` 一致，先以二进制双精度计算每 plane 扇区数并截断，再求和。控制器只允许完整页区域内的 LPA。上游 NVMe 拆包在 `src/ssd/Host_Interface_NVMe.cpp` 的 `Request_Fetch_Unit_NVMe::Process_pcie_read_message` 把高位移 31 位，主机打包在 `src/host/IO_Flow_Base.cpp` 的 `IO_Flow_Base::NVMe_read_sqe` 把地址右移 32 位。2 TiB 上限使扇区地址高位字为零。

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

以下 `src/…` 路径以固定版本的 MQSim 源码根目录为起点。

1. **并行单元是 chip。** T0 两 die 都被寻址，命令相隔一个完整的页周期，`Issued_Flash_Interleaved_Read_CMD=0`。`src/ssd/TSU_Priority_OutOfOrder.cpp` 中的 `TSU_Priority_OutOfOrder::service_read_transaction` 和 `src/ssd/TSU_OutofOrder.cpp` 中的 `TSU_OutOfOrder::service_read_transaction` 均检查 chip 状态。`src/ssd/NVM_PHY_ONFI_NVDDR2.cpp` 中的 `NVM_PHY_ONFI_NVDDR2::Execute_simulator_event` 在最后一次读传输完成后将 chip 置为 IDLE。
2. **ONFI 启动和就绪延迟。** `src/ssd/ONFI_Channel_NVDDR2.cpp` 中的 `ONFI_Channel_NVDDR2::ONFI_Channel_NVDDR2` 按头文件 `src/ssd/ONFI_Channel_NVDDR2.h` 的默认参数计算命令时间 `20 + 6×25 + 100 + 20` ns。`src/nvm_chip/flash_memory/Flash_Chip.h` 中的 `Flash_Chip::Flash_Chip` 声明将 R/B 延迟默认设为 20 ns。原版 XML 无对应字段。T-ONFI 扩展了 `src/exec/Device_Parameter_Set.cpp` 中的 `Device_Parameter_Set::XML_deserialize`，由 `src/exec/SSD_Device.cpp` 中的 `SSD_Device::SSD_Device` 将命令时序传入 ONFI 通道，估算器也用同一组参数推导 `tCS + 6×tWC + tWB + tRR`。四项非对称覆盖值为 3、9、20、7 ns 时，命令实测 84 ns，单页实测和估算均为 4,265 ns。未提供覆盖值时使用固定的 290 ns，单页实测和估算均为 4,471 ns。非默认 `cmd_ns` 需要显式 ONFI 时序配置。
3. **同 chip 的感测与传输串行。** T0、T3 的后继命令与上一读页数据完成在同一时刻，T3 的斜率为 4.470 µs/页。相关逻辑在 `src/ssd/NVM_PHY_ONFI_NVDDR2.cpp` 的 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip`、`transfer_read_data_from_chip` 和 `Execute_simulator_event` 中。
4. **传输率取整。** `src/exec/SSD_Device.cpp` 中的 `SSD_Device::SSD_Device` 将 tRC 设为 `2 × floor(1000/rate)` ns，`src/ssd/ONFI_Channel_NVDDR2.h` 的 `NVDDR2DataOutTransferTime` 宏每两个通道宽度计一次。宽度 64 B 时，rate 500、501、750、1000、1001 实测为 128、64、64、64、0 ns，rate 1001 正常退出。
5. **program 挂起。** 第 1 项列出的两个 `service_read_transaction` 都复现原版 fallthrough。首次只补 `break`，挂起计数仍为 0。对照同文件中的 `TSU_Priority_OutOfOrder::TSU_Priority_OutOfOrder`、`TSU_OutOfOrder::TSU_OutOfOrder` 和 `src/ssd/TSU_Base.cpp` 中的 `TSU_Base::TSU_Base`，发现三项时间阈值和两个布尔开关传参错序。修正后可以发挂起命令。原版 `src/ssd/NVM_PHY_ONFI_NVDDR2.cpp` 的 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip` 在普通单 die program 感测期未更新 chip 级挂起账本，读完成也无法驱动写恢复。T-SUSP 开关内同时修复转参及单 die 挂起／恢复账本，并由 T6 检查读和写都完成。调度基线保留原有 chip 门控。
6. **静默错误与 GC。**
   - `src/exec/Device_Parameter_Set.h` 将两个队列字段声明为 uint16，由 `src/exec/Device_Parameter_Set.cpp` 中的 `Device_Parameter_Set::XML_deserialize` 赋值。单独设为 65,536 的两个原版 XML 探针均正常退出且读命令为 0。T5 把 fetch 降到 512，吞吐从 465.904 降到 401.450 GB/s。
   - `src/exec/IO_Flow_Parameter_Set.cpp` 中的 `IO_Flow_Parameter_Set::XML_deserialize` 原版用 1,000 字节数组和 `strncpy`。320 个 ID 时，指定通道 319 的 LPA 实测落到 42，开启 I3 后落到 319。256 个紧凑 ID 的串长为 913 字节，该拓扑的最高编号也已用 T1 实测。
   - `src/host/IO_Flow_Trace_Based.cpp` 中的 `IO_Flow_Trace_Based::Start_simulation` 检查时间戳非降序，在非五字段行停止预扫描。三行输入的中间行畸形时只完成第一条。设置 MICROSECOND 的探针提交时间仍为 100、123 ns。
   - `src/ssd/Address_Mapping_Unit_Page_Level.cpp` 中的 `Address_Mapping_Unit_Page_Level::online_create_entry_for_reads` 调用 `src/ssd/Flash_Block_Manager.cpp` 中的 `Flash_Block_Manager::Allocate_block_and_page_in_plane_for_user_write`。两条首次读依次占用物理页 0、1，重读第一个 LPA 仍访问页 0。块写满时推进写前沿并检查 GC。
   - `src/ssd/GC_and_WL_Unit_Base.cpp` 中的 `GC_and_WL_Unit_Base::GC_and_WL_Unit_Base` 计算阈值，`src/exec/SSD_Device.cpp` 中的 `SSD_Device::SSD_Device` 传入每 plane 最多 10 个在途 GC 请求，构成阈值下限。`src/ssd/GC_and_WL_Unit_Page_Level.cpp` 中的 `GC_and_WL_Unit_Page_Level::Check_gc_required` 在空闲池低于 `max(floor(GC_Exec_Threshold×块数), 10)` 时检查 GC，候选块还要满足安全和可回收条件。`(64−10)×256=13,824` 是粗略预算，预留写前沿会占用块，实际临界页数未用压满实验核实。OP 由 `src/utils/Logical_Address_Partitioning_Unit.cpp` 中的 `Logical_Address_Partitioning_Unit::Allocate_logical_address_for_flows` 用于逻辑容量计算。T7 实测验证所选工作负载未执行 GC。
   - `src/main.cpp` 的 `main` 保留原版 `cin.get()`，基线样例通过 EOF 退出。批量链路使用 I4 和 EOF。
7. **大批的墙钟时间超线性。** 本机从 8,192 页 0.314 s 升至 32,768 页 3.871 s。64K 一次提交 16.794 s，拆成八批 1.516 s。绝对值与预审机器不同，趋势一致。设备物理页数为 16,777,216，峰值 RSS 除以这一容量约 35 B/物理页。
8. **其余行为。**
   - `src/ssd/Address_Mapping_Unit_Page_Level.cpp` 中的 `Address_Mapping_Unit_Page_Level::allocate_plane_for_preconditioning`、`allocate_plane_for_user_write`、`online_create_entry_for_reads` 使用相同的 CWDP 公式。T1 直接核对首次读位置。同文件中 `AddressMappingDomain::Mapping_entry_accessible` 在理想映射下返回可访问，T7 实测映射页读数为 0。
   - `src/host/IO_Flow_Trace_Based.cpp` 中的 `IO_Flow_Trace_Based::Execute_simulator_event` 按轨迹时间发射，T6 的读在写完成前进入队列。`src/ssd/Data_Cache_Manager_Flash_Advanced.cpp` 中的 `Data_Cache_Manager_Flash_Advanced::process_new_user_request` 在 TURNED_OFF 下调用 FTL 地址翻译。
   - `src/exec/SSD_Device.cpp` 中的 `SSD_Device::SSD_Device` 在 SLC 分支使用 LSB。改动 CSB/MSB 的探针保持 4,116 ns 感测加就绪延迟。`src/ssd/SSD_Defs.h` 的 `SECTOR_SIZE_IN_BYTE` 宏定义扇区为 512 B。
   - `src/host/PCIe_Link.h` 中的 `PCIe_Link::estimate_transfer_time` 按 GB/s 计算链路时延，满载 TLP 的载荷占比为 128/(128+28)≈82.1%。将主机链路降至 16 GB/s 后，端到端有效率实测 80.6%，包括请求控制流量和设备启动时间。

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

**T5 的 64 ns 差异来自 MQSim 对 PCIe 返回时延的整数截断及串行排队。** 基线为 16 lane × 256 GB/s，加倍为 16 lane × 512 GB/s。`src/ssd/Host_Interface_NVMe.cpp` 中的 `Request_Fetch_Unit_NVMe::Send_read_data` 提交 4 KiB 读数据。`src/host/PCIe_Link.h` 中的 `PCIe_Link::estimate_transfer_time` 将这笔数据计为 32 个载荷 128 B、额外开销 28 B 的包，共 4,992 B，分摊到 16 条 lane 为 312 B/lane。两档的耗时为 1.21875 ns 和 0.609375 ns，转成整数 `sim_time_type` 后分别为 **1 ns 和 0 ns**。`src/host/PCIe_Link.cpp` 中的 `PCIe_Link::Deliver` 和 `PCIe_Link::Execute_simulator_event` 逐个发送排队的返回消息。

按到达时间和 LBA 对照 I1 的完成日志，两档的入队时间逐请求一致。每批最后一组 64 页在加倍档同一时刻完成，在基线档逐个完成，差值从 1 ns 累积到 64 ns。六批的 makespan 均由 72,020 ns 降为 71,956 ns。吞吐与 fetch 对照仍通过，PCIe 加倍后严格相等的断言保留为 **xfail**。 [T5 汇总](evidence/validation/t5-summary.json) · [回归日志](evidence/pytest/recheck-fixes.log)

T8 按每个 (head, K/V) 无放回抽样，向上取整页数。pytest 从调度后的请求还原各张量的实际页集合，逐组核对三档各有 21、205、1,024 页，并检查整层读页数和 NAND 读计数。生成器测试验证 100% 抽样包含末尾的部分页，以及同一种子重跑得到相同页集合、改变种子可改变所选页。用每单元 32 个候选页的二项近似估算最忙单元，取 `1024×P(X≥k)≥1` 的最大 k。同头内的无放回相关性使该统计只作为量级对照。50% 拆成两个 8,192 页子批，时延比按各子批核验，最大负载按完整层统计。

## T9 稠密矩阵

32 层、8 个 KV 头、head_dim=128、BF16，上下文 32,768。每页 16 个 token，每层 128 MiB，每步 4 GiB。每层四个 8,192 页子批，表中的层时延为四个子批 makespan 之和，步时延为 32 层之和。两种估算都逐子批计算后求和。每组 32 层的最小、平均、最大在本轮相同。

回归测试按拓扑成对执行两种布局。各层实测时延允许相对表中基线偏差 5%，两种布局的平均层时延比允许相对基线偏差 10%。基线时延比按 16×64、64×16、256×4 顺序为 6.201、16.579、24.201。测试按数值核对子批和层的两种估算，涵盖层级未拆分估算，并核对步实测时延与两种估算均等于相应的逐层结果之和。

| 拓扑 | 布局 | 最忙单元页 | MQSim µs/层 | MQSim 估算 µs/层 | 理想下界 µs/层 | MQSim/估算 | 估算−理想 µs/层 | MQSim ms/步 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16×64 | layer_contiguous | 32 | 738.448 | 724.996 | 131.072 | 1.019 | 593.924 | 23.630 |
| 64×16 | layer_contiguous | 32 | 288.080 | 279.556 | 131.072 | 1.030 | 148.484 | 9.219 |
| 256×4 | layer_contiguous | 32 | 273.088 | 173.764 | 131.072 | 1.572 | 42.692 | 8.739 |
| 16×64 | vllm_interleaved | 1,024 | 4,578.760 | 4,577.284 | 4,194.304 | 1.000 | 382.980 | 146.520 |
| 64×16 | vllm_interleaved | 1,024 | 4,776.136 | 4,773.892 | 4,194.304 | 1.000 | 579.588 | 152.836 |
| 256×4 | vllm_interleaved | 1,024 | 6,608.960 | 5,560.324 | 4,194.304 | 1.189 | 1,366.020 | 211.487 |

估算器 1 取最忙单元页数乘串行页读周期与最忙通道页数乘每页通道占用的较大值，再加 1 ns。理想下界取最忙单元感测时间与总字节除聚合带宽的较大值。连续布局均匀铺满 1,024 个单元，理想步时延为 4.194 ms。交错布局每层的 16 个 (head, kind) 偏移按 512 LPA 步长推进，模 1,024 只能遍历两组偏移，于是每层 32 个活跃单元各承担 1,024 页，理想步时延升至 134.218 ms。

256×4 的通道页传输为 1,024 ns。第一个连续布局子批中，通道 0 的 32 页有 24 页在 R/B 就绪后排队，平均等待 1,101 ns，最大 2,202 ns。交错布局的通道 0 则有 512 页，其中 256 页等待，平均 367 ns。`src/ssd/NVM_PHY_ONFI_NVDDR2.cpp` 中的 `NVM_PHY_ONFI_NVDDR2::handle_ready_signal_from_chip` 把通道忙时就绪的读放入等待队列。这段等待会推迟 chip 进入 IDLE，随后才能启动同 chip 的下一条命令。估算器 1 只取两个独立占用时间的最大值，无法表达这一相互制约。

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

常规入口默认且最多每子批 8,192 页，可用 `--batch-pages` 选择更小的上限。一次进程可承载多个子批，复用设备初始化。T10 用底层接口提交显式批次。64K 同时到达探针受 NVMe 队列上限约束，有少量请求在主机等待入队。墙钟时间对机器、负载分布和批次顺序敏感，特别是大量页堆在少数 chip 的交错布局。

## 局限与下一阶段接口

- MQSim 主结果沿用 SSD 的 chip 门控、固定 ONFI 命令和 PCIe/NVMe 协议。当前模型无法表达 NAND cache read 或单 plane 感测和传输重叠。理想估算给出相应的吞吐下界模型，T-ONFI 用于独立研究四项命令时序。
- 轨迹按时间戳开环发射，层、步的服务时间通过事后相加重建。GPU 计算、层间动态依赖和真实模型决策反馈仍待接入。
- 真实模型可逐步导出 `Access` 决策表及形状 meta。以同一接口实现 KIVI、StreamingLLM、Quest 适配器。新增位宽、量化参数开销与块级选择语义时，布局应明确每个张量占用的页与有效字节。
- decode 追加写适合先在 HBM 缓冲，攒够完整页后提交给 HBF。对未满页保留 token 范围和有效位图，待下一阶段定义尾页 flush、写可见性、覆盖及读后写顺序。当前完整页 prefill 写可以用作控制对照。

## 结果文件

复审修复后，以代码提交 `afa5a2a` 跑完全部 pytest，共 **43 项通过，1 项 xfail**。非规模测试通过 34 项，T10 通过 6 项，T9 的三个拓扑用例各跑完两种布局。六组 T9 的层时延、步时延及两种估算均与表中数据一致。T5 的基线和 PCIe 加倍逐批复测为 72,020 ns 和 71,956 ns，严格相等预期保留为 xfail。 [本轮复测日志](evidence/pytest/recheck-fixes.log)

容量回归使用 16×1、每 plane 125 块的配置。最后一个完整页 LPA 476,157 以 LBA 3,809,256 正常完成，只发出一次闪存读。下一页在写入轨迹前被拒绝。每 plane 125.5 块等小数配置也在生成 XML 前被拒绝。常规 CLI 的 16,384 页设置被拒绝。

CLI 和主线程中的 API 调用会在模拟期间暂时处理 SIGTERM，清理 GNU time 与 MQSim 进程组后恢复原有信号处理方式，再递交 SIGTERM。回归测试使用含空格路径的解释器和模拟器替身，分别验证内部超时与外部 SIGTERM。另用外层 GNU `timeout` 复现原调用方式，退出码为 124，替身停止写入。工作线程中的外部信号处理由宿主进程负责。

T10 性能表为首次实测。本轮复测中，8,192 页用时 0.365 s，32,768 页用时 4.526 s，同组地址从 64K 同时到达改为 8×8K 后，墙钟时间由 19.421 s 降至 1.719 s。两种方式的模拟服务时延分别为 561.492 µs 和 576.160 µs。

复测原始记录位于 `runs/recheck-validation/`、`runs/recheck-t9-*/` 和 `runs/recheck-t10/`。历史结果及关键版本见 [上一轮复测日志](evidence/pytest/review-fixes.log)、[T9 结果](evidence/t9.csv)、[T10 结果](evidence/t10.csv) 和 [验证证据索引](evidence/README.md)。
