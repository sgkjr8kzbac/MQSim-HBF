# HBF 模拟链路第 0 阶段验证报告

## 结论

逻辑访问、页布局、开环轨迹、MQSim 和逐请求结果还原已贯通。单页读校准为 **4,471 ns**。同一 chip 上不同 die 的读仍按 **4,470 ns** 的器件周期串行。稠密基线每层读 32,768 页，三种拓扑中 `layer_contiguous` 的实测服务时间为 **738.448、288.080、273.088 µs/层**。同样的读量使用 `vllm_interleaved` 后，每层只使用 32 个单元，服务时间升到 **4.579–6.609 ms/层**。

T5 中将 PCIe 加倍，每个 8,192 页子批缩短 **64 ns（0.089%）**。吞吐与估算一致、队列深度对照通过，严格相等预期记为 xfail。T9 的 256×4 连续布局实测时间比估算器 1 高 **57.2%**，物理轨迹显示同通道感测就绪后的传输排队。T8 扩展到三种拓扑后，16×64 的 1% 与 256×4 的 50% 两组时延比为 **1.259** 和 **1.31**，超出 0.8–1.25，记为不通过（xfail），原因同为通道与单元占用相互耦合。公式和基线参数都保留任务定义。估算器 1 的两项分别是单 chip 和单通道必须串行占用的时间，物理上是 MQSim 的下界。T3、T4、T5、T8、T9 的所有子批、层和步都满足 `估算器 1 ≤ MQSim`，现在由测试检查，属于经验验证。`理想下界 ≤ 估算器 1` 由公式恒成立，只作守卫。

## 交付与可复现性

| 路径 | 内容 |
|---|---|
| `hbf/access.py` | 逻辑访问 CSV、形状校验、两种布局、七种生成模式 |
| `hbf/config.py`、`hbf/configs/` | Python 参数、三套设备和 workload XML、互校验、CWDP 与地址上界 |
| `hbf/controller.py` | 区间求并集、页请求去重、子批调度、两个估算值与负载指标 |
| `hbf/simulator.py`、`hbf/cli.py` | 固定版本构建、运行、按到达时间和 LBA 对回完成日志、聚合和简报 |
| `hbf/mqsim/`、`hbf/mqsim.sha256`、`hbf/patches/` | 上游原版源码、逐文件 SHA-256、I1–I4、T-SUSP、T-ONFI |
| `tests/`、`evidence/`、`runs/` | pytest、随 Git 保存的摘要、当前工作区的完整运行记录 |

固定上游 commit 为 `51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16`，tree 为 `c5cbd1a1e75c61f2020273ff415ebf40f9446a1c`。源码可直接在仓库中浏览，构建时逐文件校验并复制到缓存目录，再应用补丁。g++ 13.3.0、C++11、Python 3.12、pytest 9.2.0。本机用原版设备 XML 和自带 `tpcc-small.trace` 场景验证了未打补丁的 MQSim，生成并完成 6,999 个请求，用时 2.44 s。每次正式运行的 `version.json` 记录当时的仓库 commit、二进制及补丁散列和开关状态。 [MQSim 源码](hbf/mqsim/src) · [MQSim 固定版本](https://github.com/CMU-SAFARI/MQSim/commit/51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16) · [复现命令](README.md)

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
| `Page_Capacity`、扇区、`Page_No_Per_Block`、`Block_No_Per_Plane` | 4,096 B、512 B、256、64（第 1 阶段起默认 512） | 页请求为 8 扇区，设备 64 GiB 物理容量（512 块时 512 GiB） |
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
5. **program 挂起。** 第 1 项列出的两个 `service_read_transaction` 都复现原版 fallthrough。首次只补 `break`，挂起计数仍为 0。对照同文件中的 `TSU_Priority_OutOfOrder::TSU_Priority_OutOfOrder`、`TSU_OutOfOrder::TSU_OutOfOrder` 和 `src/ssd/TSU_Base.cpp` 中的 `TSU_Base::TSU_Base`，发现三项时间阈值和两个布尔开关传参错序：`TSU_Base` 的形参依次为两个布尔开关和三个时间阈值，两个子类却按三个时间阈值和两个布尔开关传入。于是 `programSuspensionEnabled` 收到 700,000 ns（转为 true），`writeReasonableSuspensionTimeForRead` 收到 `Preferred_suspend_erase_time_for_write` 的 700,000 ns。100 µs 的 program 剩余时间总小于该阈值，所以只补 `break` 时 `service_read_transaction` 仍然返回 false，不会挂起。修正后可以发挂起命令。`case ERASING` 末尾同样没有 `break`，落入 `default: return false`。T-SUSP 只修复写挂起，擦除挂起即使开启 T-SUSP 也仍然无效，本阶段不涉及擦除。T-SUSP 打开而 `CMD_Suspension_Support=NONE` 时，T0、T5、`write_then_read` 三个场景的统计输出与原版逐位一致。原版 `src/ssd/NVM_PHY_ONFI_NVDDR2.cpp` 的 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip` 在普通单 die program 感测期未更新 chip 级挂起账本，读完成也无法驱动写恢复。T-SUSP 开关内同时修复转参及单 die 挂起／恢复账本，并由 T6 检查读和写都完成。调度基线保留原有 chip 门控。
6. **静默错误与 GC。**
   - `src/exec/Device_Parameter_Set.h` 将两个队列字段声明为 uint16，由 `src/exec/Device_Parameter_Set.cpp` 中的 `Device_Parameter_Set::XML_deserialize` 赋值。单独设为 65,536 的两个原版 XML 探针均正常退出且读命令为 0。T5 把 fetch 降到 512，吞吐从 465.904 降到 401.450 GB/s。
   - `src/exec/IO_Flow_Parameter_Set.cpp` 中的 `IO_Flow_Parameter_Set::XML_deserialize` 原版用 1,000 字节数组和 `strncpy`。320 个 ID 时，指定通道 319 的 LPA 实测落到 42，开启 I3 后落到 319。256 个紧凑 ID 的串长为 913 字节，该拓扑的最高编号也已用 T1 实测。
   - `src/host/IO_Flow_Trace_Based.cpp` 中的 `IO_Flow_Trace_Based::Start_simulation` 检查时间戳非降序，在非五字段行停止预扫描。三行输入的中间行畸形时只完成第一条。设置 MICROSECOND 的探针提交时间仍为 100、123 ns。
   - `src/ssd/Address_Mapping_Unit_Page_Level.cpp` 中的 `Address_Mapping_Unit_Page_Level::online_create_entry_for_reads` 调用 `src/ssd/Flash_Block_Manager.cpp` 中的 `Flash_Block_Manager::Allocate_block_and_page_in_plane_for_user_write`。两条首次读依次占用物理页 0、1，重读第一个 LPA 仍访问页 0。块写满时推进写前沿并检查 GC。
   - `src/ssd/GC_and_WL_Unit_Base.cpp` 中的 `GC_and_WL_Unit_Base::GC_and_WL_Unit_Base` 计算阈值，`src/exec/SSD_Device.cpp` 中的 `SSD_Device::SSD_Device` 传入每 plane 最多 10 个在途 GC 请求，构成阈值下限。`src/ssd/GC_and_WL_Unit_Page_Level.cpp` 中的 `GC_and_WL_Unit_Page_Level::Check_gc_required` 在空闲池低于 `max(floor(GC_Exec_Threshold×块数), 10)` 时检查 GC，候选块还要满足安全和可回收条件。`src/ssd/Flash_Block_Manager_Base.cpp` 中的 `Flash_Block_Manager_Base::Flash_Block_Manager_Base` 在启动时为每个 flow 取走 data、translation、GC 三个写前沿块，所以单 flow 时空闲池从 B−3 块起算。data 块写满时立即换取新块，空闲池在分配到第 `(B − 3 − 阈值块数 + 1) × 每块页数` 页时低于阈值。64 块时这一页是第 13,312 页。单 plane 压满实验的结果如下：
   - 先写 6,000 个 LPA、全部重写一遍、再写新 LPA。总分配 13,311 页时没有擦除；13,312 页时发出 1 次擦除，与上述时间线一致。重写留下了失效页，GC 可以回收。
   - 只做首次读，所有页都有效。空闲池低于阈值后，`Check_gc_required` 因为没有失效页而直接返回，14,080 页（55 块）仍能正常完成。到 14,336、14,337、14,400 页时，MQSim 持续占用 CPU，不发擦除，也不再输出，只能由外层超时终止。每 plane 12 块、首次读 2,000 页同样卡住。死循环的具体触发条件未核实。

   中间件因此在写轨迹前按请求顺序累计每个 plane 的新分配页数（每次写，以及本次运行中首次出现的 LPA 的读）。超过 `(B − 3 − 阈值块数 + 1) × 每块页数 − 1` 时拒绝运行，64 块时为 13,311 页。这个上限对两种情况都成立。运行结束后，中间件还断言擦除命令数和 GC 入队数均为 0，不再只由测试检查。`test_gc_budget_guard` 和 `test_gc_budget_with_rewrites` 分别在只读和重写两种负载下核对上限：恰好等于上限时可以运行，多一页即被拒绝。OP 由 `src/utils/Logical_Address_Partitioning_Unit.cpp` 中的 `Logical_Address_Partitioning_Unit::Allocate_logical_address_for_flows` 用于逻辑容量计算。T7 实测验证所选工作负载未执行 GC。
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
| T8 | 三档随机稀疏，时延比 0.8–1.25，热点量级合理 | 三种拓扑 × 三档共 9 组。1%／10%／50% 为 336／3,280／16,384 页。整层最忙单元为 3／10／24 页，二项量级为 3／9／25。7 组时延比在 1.048–1.147。16×64@1% 为 1.259，256×4@50% 为 1.306／1.310 | **部分不通过**：2 组越界，记为严格 xfail，见下文分析 |
| T9 | 跑完六组 32 层稠密矩阵，验证映射冲突与两种估算 | 每组 1,048,576 页，GC 为 0。所有子批、层、步满足理想下界 ≤ 估算器 1 ≤ MQSim。单层不拆分提交时 16×64、256×4 连续布局为 728.356、272.320 µs，与预审 728、272 µs 一致。估算器在 256×4 有明显低估 | 通过，估算偏差保留 |
| T10 | 测五档规模的资源和单位页耗时，给出子批建议 | 见性能表，64K 附加拆分实验支持 8K 选择 | 通过 |

T4 在前 256 页每个通道只处理 1 页，255 ns 的尾部增长来自主机侧返回序列化。当 N 跨越 256 的倍数时，每通道增加一轮命令与数据竞争，造成阶梯。T0 直接测得同通道的命令启动错开 290 ns。单通道多 chip 的阶段变化反映命令和数据共享通道的调度。

**T5 的 64 ns 差异来自 MQSim 对 PCIe 返回时延的整数截断及串行排队。** 基线为 16 lane × 256 GB/s，加倍为 16 lane × 512 GB/s。`src/ssd/Host_Interface_NVMe.cpp` 中的 `Request_Fetch_Unit_NVMe::Send_read_data` 提交 4 KiB 读数据。`src/host/PCIe_Link.h` 中的 `PCIe_Link::estimate_transfer_time` 将这笔数据计为 32 个载荷 128 B、额外开销 28 B 的包，共 4,992 B，分摊到 16 条 lane 为 312 B/lane。两档的耗时为 1.21875 ns 和 0.609375 ns，转成整数 `sim_time_type` 后分别为 **1 ns 和 0 ns**。`src/host/PCIe_Link.cpp` 中的 `PCIe_Link::Deliver` 和 `PCIe_Link::Execute_simulator_event` 逐个发送排队的返回消息。

按到达时间和 LBA 对照 I1 的完成日志，两档的入队时间逐请求一致。每批最后一组 64 页在加倍档同一时刻完成，在基线档逐个完成，差值从 1 ns 累积到 64 ns。六批的 makespan 均由 72,020 ns 降为 71,956 ns。吞吐与 fetch 对照仍通过，PCIe 加倍后严格相等的断言保留为 **xfail**。 [T5 汇总](evidence/validation/t5-summary.json) · [回归日志](evidence/pytest/recheck-fixes.log)

T8 按每个 (head, K/V) 无放回抽样，向上取整页数。pytest 从调度后的请求还原各张量的实际页集合，逐组核对三档各有 21、205、1,024 页，并检查整层读页数和 NAND 读计数。生成器测试验证 100% 抽样包含末尾的部分页，以及同一种子重跑得到相同页集合、改变种子可改变所选页。用每单元 32 个候选页的二项近似估算最忙单元，取 `1024×P(X≥k)≥1` 的最大 k。同头内的无放回相关性使该统计只作为量级对照。50% 拆成两个 8,192 页子批，时延比按各子批核验，最大负载按完整层统计。

T8 原先只在默认的 64×16 上运行。现在对三种拓扑各跑三档，并记录每个子批的通道项与单元项之比 `N_c·(t_cmd+t_xfer) / (n_u·(t_cmd+tR+t_RB+t_xfer))`。

| 拓扑 | 比例 | n_u | N_c | 单元项 ns | 通道项 ns | 通道项/单元项 | MQSim ns | MQSim/估算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16×64 | 1% | 3 | 31 | 13,410 | 10,974 | 0.82 | 16,891 | **1.259** |
| 16×64 | 10% | 10 | 228 | 44,700 | 80,712 | 1.81 | 91,453 | 1.133 |
| 16×64 | 50% | 14 | 533／532 | 62,580 | 188,682／188,328 | 3.0 | 200,131／197,299 | 1.061／1.048 |
| 64×16 | 1% | 3 | 10 | 13,986 | 5,460 | 0.39 | 15,557 | 1.112 |
| 64×16 | 10% | 10 | 64 | 46,620 | 34,944 | 0.75 | 53,481 | 1.147 |
| 64×16 | 50% | 14 | 143／151 | 65,268 | 78,078／82,446 | 1.20／1.26 | 89,223／89,812 | 1.143／1.089 |
| 256×4 | 1% | 3 | 7 | 16,290 | 9,198 | 0.57 | 18,339 | 1.126 |
| 256×4 | 10% | 10 | 22 | 54,300 | 28,908 | 0.53 | 57,777 | 1.064 |
| 256×4 | 50% | 14 | 42／41 | 76,020 | 55,188／53,874 | 0.73／0.71 | 99,263／99,573 | **1.306／1.310** |

九组中七组的偏差在 5–15%。超出区间的两组，通道项与单元项之比为 0.82 和 0.71–0.73，两项接近。最忙单元的读页感测完成后，若所在通道正被同通道其他 chip 的命令和数据占用，数据要排队等待，chip 在传完之前不回到 IDLE，不能开始下一页。于是单元的串行周期被通道等待拉长，实测落在两项的较大值与两项之和之间。例如 256×4@50% 实测 99,263 ns，介于 76,020 与 131,208 ns 之间。T9 的 256×4 连续布局（每子批单元项 43,440 ns、通道项 42,048 ns，比值 0.97）是同一机制，偏差 57%。按约束不修改估算器公式和阈值，两组越界这样记录：测试先完成其余全部断言，再断言越界组合恰好是这两组，最后对这两组调用 `pytest.xfail`。其他错误不会被记成 xfail；若以后这两组回到区间内，或有别的组越界，测试都会失败。下一阶段若需要估算器在这一区域可用，可以增加按通道做列表调度的耦合估算，作为第三列对照，而不替换估算器 1。

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

子批间隔只用于实验隔离。对照 T10 的 64K 开环发射和 8×8K 拆分，同组地址的服务时间分别为 561.492 和 576.160 µs，拆分高出约 2.61%，因为每批单独经历启动和排空。T9 的整批公式也保存在 `layers.csv`，本矩阵的公式拆分差别为 3 ns/层，来自额外三次主机常数。

`test_t9_unsplit_layer` 把第 0 层的 32,768 页一次性提交，与拆成 4×8K 的结果对照：

| 拓扑 | 布局 | 不拆分 µs/层 | 4×8K µs/层 | 拆分/不拆分 | 墙钟 s（不拆分） |
|---|---|---:|---:|---:|---:|
| 16×64 | layer_contiguous | 728.356 | 738.448 | 1.0139 | 21.8 |
| 64×16 | layer_contiguous | 281.940 | 288.080 | 1.0218 | 12.0 |
| 256×4 | layer_contiguous | 272.320 | 273.088 | 1.0028 | 15.9 |
| 16×64 | vllm_interleaved | 4,577.650 | 4,578.760 | 1.0002 | 18.0 |
| 64×16 | vllm_interleaved | 4,774.450 | 4,776.136 | 1.0004 | 17.3 |
| 256×4 | vllm_interleaved | 6,608.912 | 6,608.960 | 1.0000 | 16.0 |

拆分最多使层时延增加 2.2%，来自每个子批单独经历的流水线启动和排空。交错布局的时延由 32 个单元上的长串行队列主导，启动和排空占比可以忽略。预审参考值 728、272 µs 对应不拆分的提交方式，测试用 5% 容差核对这两个外部参考值。表中 T9 主矩阵仍按 8K 子批报告，数值偏保守。

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

## 第 1 阶段：模拟器改进

本节记录第 0 阶段之后对模拟器本身的三项改进。三项都在分支 `claude/gallant-maxwell-x9sgi0` 上分别提交，可以单独评估是否合入。

### 容量：每 plane 512 块

第 0 阶段每 plane 只配 64 块，逻辑容量约 1,560 万页，Llama-3.1-8B 在 32K 上下文、batch 16 时就会超出。这个上限来自配置，不是 MQSim 写死的。实测 MQSim 内存约为每物理页 31 B：

| 每 plane 块数 | 逻辑容量 | MQSim 峰值 RSS（1,024 plane） |
|---:|---:|---:|
| 64 | 1,560 万页（约 64 GiB 物理） | 0.57 GB |
| 256 | 6,240 万页 | 2.1 GB |
| 512 | 1.25 亿页（512 GiB 物理，与 OCP 规范示例 cube 相同） | 4.2 GB |

默认值改为 512 块，GC 预算相应变为每 plane 124,159 页。两个 GC 边界测试仍固定使用 64 块，以保持运行时间不变。

### 机制分析模式

新模块 `hbf/analysis.py` 不运行 MQSim，只根据 CWDP 映射把每页放到 (die, plane, 偏移)，逐个 (step, layer) 计算：

- 读轮数 R：独立 plane 时为最忙 plane 的页数；同偏移联动时为每组需要的不同偏移个数的最大值；
- 最少轮数 R* = ⌈N/P⌉、plane 利用率 N/(R·P)、均衡损失 R/R*；
- 感测页数与感测放大：联动时，组内每个 plane 每轮都要感测，即使它没有要读的页；
- 64 B 突发传输字节：OCP 规范 §4.1 允许读突发为 64 B 到 4 KiB，因此传输可以小于整页，感测始终是整页。

页在 plane 内的偏移有两种算法（`--placement`）。默认 `static` 按 OCP 规范：地址决定 bank、die、块和页（§4.6、图 4；§5.4.1 块内连续写、禁止跳地址；§11.4 无页级 FTL），偏移为 `lpa // 单元数`，并假定数据已在 HBF 中。`mqsim` 按 MQSim 的实际分配：每个 plane 按轨迹顺序，在每次写入、以及未写过的 LPA 首次被读时，取写前沿的下一页；加 `--prefill`（按地址顺序写满整个布局区间，含填充页）后两者相同，不加时稀疏读的页按首次读取的顺序紧凑排列，这在规范中不允许（读未写页返回错误 0x7）。一层超过子批上限被拆开时，`batch_rounds` 给出各子批轮数之和，这是与 MQSim 对照的口径；`rounds` 把整层当作一次突发。独立 plane 的轮数与已有指标 `max_unit_pages` 逐层相等，有测试核对。

**与 MQSim 逐层核对**：2 die × 8 plane，`vllm_interleaved`、`layer_contiguous` 和 `striped_contiguous`，含层数、头数不整除 plane 数的形状，independent 与 lockstep 两种规则，加或不加 T-CMD 合并窗口，共 16 组，全部先整体 prefill。物理时间线上每个写入页的 (块, 页) 都等于规范的静态偏移 `lpa // 单元数`；每层每 die 实际发出的读命令数落在 [静态分析轮数, 轮数 + 1] 之内，开合并窗口时正好相等。多 plane 配置下，读未写过的页会在运行前被拒绝（chip 当 plane 时偏移不影响时序，只计数）。

这一部分经历过两次修正：初版直接取 `lpa // 单元数`，但当时的 prefill 逐层写入、不写条带布局的填充页，MQSim 的位置因此与之不符（第二轮审查）；随后一度把分析改为跟随 MQSim 的分配，规范一致性审查（见下文 S1–S4）指出规范要求的是静态位置，于是改为按地址顺序写满整个区间，使 MQSim 与规范一致，分析默认回到静态位置。

**示例扫描**：Llama-3.1-8B 形状（32 层、8 个 KV 头、32K），先整体 prefill，再每层每头每张量均匀随机选页，下表为 32 层合计的读轮数（括号内为均衡损失 R/R*），数据见 [evidence/stage1-analysis-sweep.csv](evidence/stage1-analysis-sweep.csv)：

| plane 数 | 格式 | 布局 | 选页比例 | 读页 | 独立 plane | 联动 16 | 联动 32 | 联动 64 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1,024 | BF16 | layer_contiguous | 1% | 10,752 | 101 (3.2) | 332 | 461 | 664 |
| 1,024 | BF16 | layer_contiguous | 5% | 52,736 | 215 (3.4) | 789 | 971 | 1,024 |
| 1,024 | BF16 | layer_contiguous | 20% | 209,920 | 473 (2.1) | 1,024 | 1,024 | 1,024 |
| 1,024 | 2 bit | layer_contiguous | 1% | 1,536 | 58 (1.8) | 95 | 106 | 123 |
| 1,024 | BF16 | vllm_interleaved | 1% | 10,752 | 488 (15.3) | 488 | 488 | 672 |
| 256 | BF16 | layer_contiguous | 1% | 10,752 | 175 (2.7) | 841 | — | — |

几点观察（均匀随机选页，真实方法的选择有局部性，结论需用真实 trace 复核）：

1. BF16、`layer_contiguous` 的稀疏读，即使 plane 独立寻址，利用率也只有约 10–43%，原因是每层页数少于或接近 plane 数，最忙 plane 决定轮数（球入箱问题）。
2. 2 bit 格式读的页数是 BF16 的 1/7，轮数只少 1.7 倍：压缩和稀疏选择的收益在感测粒度上叠加不起来。
3. BF16 时同偏移联动使稀疏读的轮数增加 3–7 倍，1% 选页时感测放大达到 15–48；选页比例高到每组的每个偏移都要读时，联动和独立的差距才缩小。
4. 布局影响仍然最大：`vllm_interleaved` 每层只用 32 个 plane，独立 plane 下轮数也是 `layer_contiguous` 的 4.8 倍。

### T-PLANE：真实 die/plane 层级

第 0 阶段用"一个 MQSim chip 当一个 plane"规避了 MQSim multi-plane 的问题。源码核对发现原版 multi-plane 有三处限制：

| 问题 | 位置 | 后果 |
|---|---|---|
| 命令时间表只填了 1–4 个 plane | `ONFI_Channel_NVDDR2.h` 中 `ReadCommandTime[5]` 等 | 超过 4 个 plane 时越界读取 |
| plane 掩码为 32 位 `unsigned int` | `TSU_Base::issue_command_to_chip` | 超过 32 个 plane 时溢出 |
| 合并条件固定为页号相同 | 同上 | 只能表达同偏移规则，不能表达规范的独立寻址 |

补丁 `T-PLANE.patch` 把三张命令时间表按原版的逐 plane 增量扩展到 64 个 plane，掩码改为 64 位，并由 `HBF_T_PLANE` 选择合并规则：`independent`（任意块、页）或 `lockstep`（块号和页号都相同）；不设时保持原版。另把 t_DBSY 暴露到 XML（`ONFI_t_DBSY`，原版 500 ns）。1–4 个 plane 的表项不变；32 个 plane 以内掩码结果不变；规则在单 plane 配置下不起作用，均有测试核对。

**时序核对**：单 die 读 n 页（n = 4、8、64，t_DBSY 取 500、0、100 ns），MQSim 时延与公式

host + [cmd(1) + tR + R/B + x] + [cmd(n−1) + tR + R/B + (n−1)·x]，cmd(n) = t_CS + n·6·t_WC + (n−1)·t_DBSY + t_WB + t_RR

逐 ns 相等。第一页单独下发，是因为它到达时 die 空闲、队列里只有它一个。n = 4 时为 10,369 ns，与第 0 阶段探针一致。

**与机制分析对照**：16 die × 64 plane，8 个头、8K 上下文、均匀选 5% 的页，先整体 prefill：

| 规则 | 分析：最忙 die 的轮数 | 分析：16 个 die 的轮数之和 | MQSim 读命令数（16 个 die 合计） | MQSim 读时延 |
|---|---:|---:|---:|---:|
| chip 当 plane（对照） | 4 | — | 416（每页一条） | 21.8 µs |
| independent | 4 | 38 | 54 | 42.9 µs |
| lockstep | 8 | 122 | 133 | 61.7 µs |
| 不设（原版规则） | 8 | 122 | 133 | 61.7 µs |

16 个 die 合计的读命令数落在 [各 die 所需轮数之和, 该和 + die 数] 之内，多出的部分来自上面说的首页单独下发，有测试核对。prefill 不可省：MQSim 在首次读到一个 LPA 时按到达顺序分配物理页，不写入就读，稀疏读的页会被紧凑地分配在相同偏移上，lockstep 和 independent 会给出相同结果；按规范这种读本应返回错误 0x7，所以多 plane 配置下现在直接拒绝。

### T-CMD：HBF 命令模型

第二轮审查指出，真实层级下 MQSim 的时延基本随页数而不是随轮数变化：64 个 plane 的 ONFI 读命令要 20 + 64·150 + 63·500 + 120 = 41.2 µs，是 tR 的 10 倍；第一页到达空闲 die 时单独下发，使稀疏 decode 每层多出一条命令。补丁 `T-CMD.patch` 提供两个开关：

- `HBF_T_READ_PLANE_NS=δ`：n plane 读命令 = 单 plane 命令 + (n−1)·δ（`ONFI_Channel_NVDDR2` 构造函数）；
- `HBF_T_COALESCE_NS=W`：`TSU_Priority_OutOfOrder::Schedule` 收到新事务后先放进暂存表，不进调度队列，并登记 now + W 的事件（窗口内再到的事务只加入暂存表）；事件到时才把暂存的事务放入队列并统一下发。初版只推迟了 `Schedule` 自己的下发，通道或 chip 空闲信号仍会提前取走队列里的新事务，第三轮审查发现后改为暂存。

单 die 读 n 页，时延等于 W + cmd(n) + tR + R/B + n·x + host，n = 4、8、64 时逐 ns 相等，只发 1 条命令。同一组 16 die × 64 plane、5% 选页的对照：

| 设置 | independent 命令数 / 时延 | lockstep 命令数 / 时延 | 时延比 | 轮数比 |
|---|---:|---:|---:|---:|
| ONFI 原始命令 | 54 / 42.9 µs | 133 / 61.7 µs | 1.44 | 2 |
| 合并窗口 1 µs | 38 / 40.1 µs | 122 / 58.9 µs | 1.47 | 2 |
| 合并 + δ = 10 ns | 38 / 21.0 µs | 122 / 39.1 µs | 1.86 | 2 |
| 合并 + δ = 10 ns + 每 die 256 GB/s | 38 / 19.4 µs | 122 / 37.2 µs | 1.91 | 2 |

加合并窗口后，命令数正好等于各 die 轮数之和（38、122），逐层逐 die 的精确相等也写成了测试。δ 取多少没有公开数据，应作为敏感性参数扫描。剩下的差距来自数据逐页传出、不能与下一轮感测重叠，即缺少 cache read。

补丁改动了 T-SUSP 也改过的文件，必须在其后应用，所以新增 `hbf/patches/series` 规定顺序，构建时按它应用，并检查它列出了全部补丁。

### 条带布局与 sink_recent 负载

新增布局 `striped_contiguous`：按 `layer_contiguous` 的顺序排列区域，但每个 (层, 张量, 头) 区域补齐到指定的 plane 相位再开始，相位为 t·P/T + h·P/(T·H)，即默认 K/V 时头之间错开 P/(2H)、V 与 K 相隔 P/2（SPLASH 的做法）。每个区域最多补 P−1 个空页，空页不会被写入或读取。新增生成模式 `sink_recent`：每层每头每张量读前 `count` 个 token 加最后 `fraction` 比例的 token（StreamingLLM 式）。

32K 上下文下 `layer_contiguous` 每个头 2,048 块，正好是 1,024 的倍数，各头各张量的同一块落在同一 plane。机制分析（Llama-3.1-8B 形状、32 层、sink 4 个 token、最近 1/64）：

| 布局 | 每层读页 | 32 层轮数：独立 plane（利用率） | 联动 16 | 联动 64 | 最少轮数 |
|---|---:|---:|---:|---:|---:|
| layer_contiguous | 528 | 512（0.032） | 512 | 1,024 | 32 |
| striped_contiguous | 528 | 32（0.516） | 192 | 576 | 32 |
| vllm_interleaved | 528 | 544（0.030） | 544 | 544 | 32 |

MQSim 真实层级（16 die × 64 plane，T-CMD δ = 10 ns、窗口 1 µs，T-CACHE，prefill 后读 4 层）：

| 布局 | plane 规则 | 4 层轮数 | 每层时延 |
|---|---|---:|---:|
| layer_contiguous | independent | 64 | 72.0 µs |
| striped_contiguous | independent | 4 | 8.9 µs |
| layer_contiguous | lockstep | 128 | 142.3 µs |
| striped_contiguous | lockstep | 72 | 80.7 µs |

条带化在 plane 独立寻址时效果最好（轮数降到下限）；在 lockstep 下也有效，但幅度小得多：每层 32 → 18 轮，因为联动要求同一轮读同一偏移，条带化把各头的页分散到不同 plane，却不能让它们的偏移对齐。此前版本写的是"lockstep 下轮数不变（128 轮）"，那是 prefill 没有写填充页、MQSim 压紧偏移造成的假象（规范一致性审查 S3），已按规范的静态位置重测。

### 规范示例拓扑 `ocp`

按规范图 4（p23）的例子新增命名拓扑 `ocp`：每个主机通道 2 条 core die 通道 × 每通道 2 个 die × 每 die 16 个 bank，16 个主机通道共 1,024 个 bank；MQSim 中为 32 通道 × 2 chip × 16 plane，plane 规则 independent。`Config.address_map = "ocp"` 按规范排列地址：4 KiB 单元在主机通道间交织（图 3，n = 6）；通道内低 4 位为 bank、再 2 位为 die（§4.6）；die 0、2 在 core die 通道 0，die 1、3 在通道 1（图 4）。`Config.device_lpa` 把这个顺序换算成 MQSim CWDP 下的 LPA，`host_lpa` 是其逆；测试逐个核对图 4 中的位置、双射性，以及 MQSim 物理时间线上每个写入页的 (通道, chip, plane, 偏移)。

机制分析中，`ocp` 每个 die 的 16 个 bank 与 16×64 中"同一通道、相邻 16 个 chip"恰好是同一组单元，所以独立 plane 和 16 联动的轮数两者完全相同（Llama-3.1-8B 形状、32K：5% 均匀选页 215 / 789 轮，sink_recent 512 / 512 轮，条带布局 32 / 192 轮）。两者的差别在 MQSim 的总线结构：`ocp` 是 32 条通道各挂 2 个 die，16×64 是 16 条通道各挂 64 个 chip。

### T-CACHE：cache read

原版 MQSim 的读要等全部页传出后 chip 才回到空闲，感测和传输不能重叠。补丁 `T-CACHE.patch`（`HBF_T_CACHE`）对用户读实现双缓冲：

- 感测完成时（`handle_ready_signal_from_chip`），这批页的命令和事务移入一个独立的 cache 记账项，die 的记账项立即释放；
- TSU 判断能否发读时（`service_read_transaction`），若该 chip 没有正在感测的 die、没有正在传的命令、每个 die 等待传出的批次都不超过 1 个，就视为空闲；
- 每传完一页（以及感测刚完成、通道空闲时），先做一次"cache 优先"调度：这一轮只允许向上述 cache 就绪的 chip 发读命令，然后继续传排队的数据。写、擦除和其他 chip 的命令仍按上游顺序排在数据传出之后。初版让任何命令都插到数据传出之前，读写混合时读反而慢 5.5%，第三轮审查发现后收窄；
- 某批数据全部传出后删除对应的 cache 记账项。

时序逐 ns 核对（1 die，T-CMD δ = 10 ns、窗口 500 ns；关闭/打开 T-CACHE）：

| 场景 | 关闭 | 打开 | 手算 |
|---|---:|---:|---|
| 1 plane 读 4 页 | 17,881 | 17,689 | 每页只藏住 64 ns 传输 |
| 64 plane × 4 轮，64 GB/s（感测受限） | 37,029 | 24,741 | 500 + 920 + 4·4116 + 3·920 + 64·64 + 1 |
| 64 plane × 3 轮，16 GB/s（传输受限） | 64,761 | 56,529 | 第三轮命令在 22,840 ns 才发出，即第一批传完、cache 空出之后 |

物理时间线上的 `ready` 事件在 cache 路径中也会记录。另有 6 组随机读写混合测试（多 chip、多 die、批次重叠，开关 T-CACHE 各跑一遍），检查全部请求完成、同一 die 的感测不重叠、数据传出不早于感测结束。

cache read 不改变轮数：16 die × 64 plane、lockstep、10% 选页时，开关前后读命令数相同且等于分析轮数，时延下降。目前只覆盖用户读；映射读和 GC 读仍走原路径，并要求不开挂起（与 T-SUSP 的交互未验证）。

不开 T-CACHE 时，真实层级下的时延反而高于"chip 当 plane"，原因是上游 MQSim 没有 cache read：一轮 multi-plane 读要等全部页依次传完，die 才能开始下一轮，每轮约为 cmd(n) + tR + n·x，命令和传输都不能与感测重叠；而 chip 当 plane 时各 plane 的命令和传输可以与其他 plane 的感测交错。上面的 T-CACHE 补丁补上了这一点。

## 规范一致性审查（OCP HBF v0.7.0）

第 1 阶段之后，由两个子代理先后对照 OCP HBF 规范 v0.7.0 审查并互相核实，关键数字另行复算。页码取自规范 PDF 页脚。结论：独立 plane 的结果都有效；页在 plane 内的偏移不符合规范，影响所有 lockstep 和条带布局的结论；另有两项配置能力缺失和若干文档措辞问题。

| # | 问题 | 规范依据 | 影响 | 处理 | 状态 |
|---|---|---|---|---|---|
| S1 | 页在 plane 内的偏移由 MQSim 按写入或首次读的顺序分配；规范是地址决定位置（无页级 FTL） | §4.6 p22–23 地址位序与图 4；§5.4.1 p58 块内连续写、禁止跳地址、写 page 0 自动擦除；§5.7 p62–63 由地址算 page-0；§11.4 p117 无 GC、只做块级重映射 | lockstep 轮数：4 层 × 32K、5% 选页、32 联动时，不加 prefill 43 轮，按规范 122 轮；独立 plane 不受影响 | 分析默认用静态偏移；prefill 按地址顺序写满；多 plane 运行要求页已写入 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S2 | 读未写过的页在规范中返回错误，MQSim 却分配新页并成功返回 | 表 12 p60 "Read from ErasePage"，错误码 0x7；§11.5.1.7 p120 | 同 S1 | 统计并拒绝（多 plane 时）读未写页 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S3 | 条带布局的填充地址没有写入，MQSim 把后续页的偏移压紧 | §5.4.1 第 4 条 "No skipping or jumping of 4KiB local address" | 条带布局 lockstep-64 每层按规范 18 轮，报告写成 32 轮；"条带化在 lockstep 下无效"的结论不成立 | prefill 连同填充地址一起写 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S4 | README 的 lockstep 示例没加 `--prefill` | 同 S1 | 示例结果低估约 2.8 倍 | 改示例 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S5 | 规范示例层级没有现成配置：每主机通道 2 条 core die 通道 × 每通道 2 die × 每 die 16 bank，共 1,024 bank；地址先 bank 后 die | 图 4 p23（"for example"）；表 3 p16 只给 cube 汇总，信息不全 | 现有 T-PLANE 用 SPLASH 的 16 die × 64 plane，分组不同 | 新增规范示例拓扑和地址顺序 | 已改：新增 `--topology ocp`（32 通道 × 2 die × 16 bank，规范地址顺序） |
| S6 | 规范三档带宽 0.384/1.536/3.072 TB/s 无法配置：16 通道下每通道 24/96/192 GB/s 不是 2 的幂，配置检查拒绝 | 表 2、表 4 p16 | 无法做带宽档位扫描 | 放宽检查，传输时间与 MQSim 的整数算法一致 | 待改 |
| S7 | 文档措辞：缓冲个数写成"要求"；"16x16 对应规范示例"；"16 die × 64 plane 和 HBF 一致"；"标称 1 TB/s"；512 GiB 是物理容量而规范表 3 是用户容量 | §5.3.1 第 7 条 p57；图 4；表 3、表 4 | 仅文档 | 改措辞并注明出处 | 待改 |
| S8 | 调度按 die 门控（T-PLANE、T-CACHE），规范按 bank 排序感测、每 bank 两个缓冲 | §5.3.1 第 5b–8 条 p57 | 多 plane 模式下时延偏保守，轮数不变 | 不改：chip 当 plane 模式即按 bank 独立调度；在文档中说明 | 不改 |
| S9 | 读计数、刷新、坏块、ECC、读重试、数据保持未建模 | §4.4 p19；§11.5 p118（刷新是主机责任）；§4.1；§5.3.1 第 9 条；§9 p106 | 无故障时不改变轮数和利用率 | 列为范围外 | 不改 |

## 局限与下一阶段接口

- MQSim 主结果沿用 SSD 的 chip 门控、固定 ONFI 命令和 PCIe/NVMe 协议。上游没有 cache read，感测和传输不能重叠；T-CACHE 为用户读补上了双缓冲，映射读和 GC 读仍走上游路径。T-CMD 的合并窗口对所有新事务统一生效，所以孤立批次和重叠批次都会多出 W 的等待。理想估算给出相应的吞吐下界模型，T-ONFI 用于独立研究四项命令时序。
- I2 只在 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip` 记录命令下发。同文件 `perform_interleaved_cmd_data_transfer` 发出的 die 交错命令没有 issue 记录，后续的 cmd_done、ready、data 事件仍有记录。本阶段 D=P=1，不经过该路径。若以后用 D>1 的配置分析时间线，需要补记这一处。
- 运行需要 `/usr/bin/time`（GNU time）记录峰值 RSS。缺少时中间件在启动 MQSim 前报错，并提示安装 `time` 包。
- 轨迹按时间戳开环发射，层、步的服务时间通过事后相加重建。GPU 计算、层间动态依赖和真实模型决策反馈仍待接入。
- 真实模型可逐步导出 `Access` 决策表及形状 meta。访问记录的表头不变，`kind` 现在指张量名，`bits` 必须等于该张量的存储位宽。张量规格 `TensorSpec(name, bits, dims)` 由 `--tensors` 或 Python 接口声明，默认是 BF16 的 K 和 V，此时两种布局的 LPA 与原公式逐位相同，有测试逐页核对。各方法的表达方式：
  - KIVI：2 bit 或 4 bit 的 K、V，加量化参数张量，如 `K_scale`。
  - ThinK 静态通道裁剪：K 的 dims 小于 head_dim。
  - 低秩：latent 张量，dims 取秩。
  - Quest：页摘要是每块一行而不是每 token 一行，现在的 token 区间语义表达不了，需要单独的块索引维度，尚不支持。
  - StreamingLLM、SnapKV：沿用 BF16 的 K、V，只改变读取的 token 区间。
- 每页 token 数与有用字节按各张量的 `dims × bits` 计算。`layer_contiguous` 按 (层, 张量, 头) 依次预留区域，各张量的块数可以不同。`vllm_interleaved` 要求各张量每页 token 数相同。尚未支持的情况有两种：一是读一个张量的部分通道（动态通道选择），需要通道主序布局；二是同一张量在不同 token 段使用不同位宽，例如 KIVI 的全精度残差窗口，目前要声明成两个张量。
- decode 追加写适合先在 HBM 缓冲，攒够完整页后提交给 HBF。对未满页保留 token 范围和有效位图，待下一阶段定义尾页 flush、写可见性、覆盖及读后写顺序。当前完整页 prefill 写可以用作控制对照。

## 结果文件

**第 1 阶段（提交 `970c2b8`）**：全量 pytest 共 **116 项通过，3 项 xfail**，用时 14 min 19 s，[测试日志](evidence/pytest/stage1.log)。3 项 xfail 与第 0 阶段相同。新增测试 58 项（含参数化），其中 `tests/test_analysis.py` 17 项、`tests/test_plane.py` 41 项；两项 GC 边界测试改为显式使用 64 块。


**第二轮审查修改后（提交 `df8925e`）**：全量 pytest 共 **58 项通过，3 项 xfail**。3 项 xfail 是 T5 的 PCIe 严格相等，以及 T8 的 16×64@1% 和 256×4@50%。全量运行（含规模测试）在 `df8925e` 之前的代码上完成，结果为 57 项通过、3 项 xfail；`df8925e` 只改动了 GC 预算、T8 的 xfail 写法和文档，此后非规模测试重跑为 43 项通过、3 项 xfail，新增 1 项重写场景的 GC 测试。六组 T9 的层时延与上一轮逐位一致。本轮新增的证据：
- [测试日志](evidence/pytest/review-round2.log)
- [T8 三拓扑摘要](evidence/t8-topologies.csv)
- [T9 不拆分对照](evidence/t9-unsplit.csv)

以下为上一轮记录。

复审修复后，以代码提交 `afa5a2a` 跑完全部 pytest，共 **43 项通过，1 项 xfail**。非规模测试通过 34 项，T10 通过 6 项，T9 的三个拓扑用例各跑完两种布局。六组 T9 的层时延、步时延及两种估算均与表中数据一致。T5 的基线和 PCIe 加倍逐批复测为 72,020 ns 和 71,956 ns，严格相等预期保留为 xfail。 [本轮复测日志](evidence/pytest/recheck-fixes.log)

容量回归使用 16×1、每 plane 125 块的配置。最后一个完整页 LPA 476,157 以 LBA 3,809,256 正常完成，只发出一次闪存读。下一页在写入轨迹前被拒绝。每 plane 125.5 块等小数配置也在生成 XML 前被拒绝。常规 CLI 的 16,384 页设置被拒绝。

CLI 和主线程中的 API 调用会在模拟期间暂时处理 SIGTERM，清理 GNU time 与 MQSim 进程组后恢复原有信号处理方式，再递交 SIGTERM。回归测试使用含空格路径的解释器和模拟器替身，分别验证内部超时与外部 SIGTERM。另用外层 GNU `timeout` 复现原调用方式，退出码为 124，替身停止写入。工作线程中的外部信号处理由宿主进程负责。

T10 性能表为首次实测。本轮复测中，8,192 页用时 0.365 s，32,768 页用时 4.526 s，同组地址从 64K 同时到达改为 8×8K 后，墙钟时间由 19.421 s 降至 1.719 s。两种方式的模拟服务时延分别为 561.492 µs 和 576.160 µs。

复测原始记录位于 `runs/recheck-validation/`、`runs/recheck-t9-*/` 和 `runs/recheck-t10/`。历史结果及关键版本见 [上一轮复测日志](evidence/pytest/review-fixes.log)、[T9 结果](evidence/t9.csv)、[T10 结果](evidence/t10.csv) 和 [验证证据索引](evidence/README.md)。
