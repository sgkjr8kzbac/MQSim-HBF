# HBF 模拟链路验证与实验报告（第 0、1 阶段，第 2 阶段 A 与 B（一））

## 结论

逻辑访问、页布局、开环轨迹、MQSim 和逐请求结果还原已贯通。单页读校准为 **4,471 ns**。同一 chip 上不同 die 的读仍按 **4,470 ns** 的器件周期串行。稠密基线每层读 32,768 页，三种拓扑中 `layer_contiguous` 的实测服务时间为 **738.448、288.080、273.088 µs/层**。同样的读量使用 `vllm_interleaved` 后，每层只使用 32 个单元，服务时间升到 **4.579–6.609 ms/层**。

T5 中将 PCIe 加倍，每个 8,192 页子批缩短 **64 ns（0.089%）**。吞吐与估算一致、队列深度对照通过，严格相等预期记为 xfail。T9 的 256×4 连续布局实测时间比估算器 1 高 **57.2%**，物理轨迹显示同通道感测就绪后的传输排队。T8 扩展到三种拓扑后，16×64 的 1% 与 256×4 的 50% 两组时延比为 **1.259** 和 **1.31**，超出 0.8–1.25，记为不通过（xfail），原因同为通道与单元占用相互耦合。公式和基线参数都保留任务定义。估算器 1 的两项分别是单 chip 和单通道必须串行占用的时间，物理上是 MQSim 的下界。T3、T4、T5、T8、T9 的所有子批、层和步都满足 `估算器 1 ≤ MQSim`，现在由测试检查，属于经验验证。`理想下界 ≤ 估算器 1` 由公式恒成立，只作守卫。

## 交付与可复现性

| 路径 | 内容 |
|---|---|
| `hbf/access.py` | 逻辑访问 CSV、形状校验、四种布局、九种生成模式 |
| `hbf/analysis.py` | 机制分析（读轮数、联动组、核心通道下界）与主机协议约束 |
| `hbf/config.py`、`hbf/configs/` | Python 参数、三套设备和 workload XML、互校验、CWDP 与地址上界 |
| `hbf/controller.py` | 区间求并集、页请求去重、子批调度、两个估算值与负载指标 |
| `hbf/simulator.py`、`hbf/cli.py` | 固定版本构建、运行、按到达时间和 LBA 对回完成日志、聚合和简报 |
| `hbf/mqsim/`、`hbf/mqsim.sha256`、`hbf/patches/` | 上游原版源码、逐文件 SHA-256、I1–I4、T-SUSP、T-ONFI、T-PLANE、T-CMD、T-CACHE，应用顺序见 `patches/series` |
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

| 拓扑 | 布局 | 不拆分 µs/层 | 4×8K µs/层 | 拆分/不拆分 | 墙钟 s（不拆分，取自 [evidence/t9-unsplit.csv](evidence/t9-unsplit.csv)） |
|---|---|---:|---:|---:|---:|
| 16×64 | layer_contiguous | 728.356 | 738.448 | 1.0139 | 17.0 |
| 64×16 | layer_contiguous | 281.940 | 288.080 | 1.0218 | 17.5 |
| 256×4 | layer_contiguous | 272.320 | 273.088 | 1.0028 | 9.9 |
| 16×64 | vllm_interleaved | 4,577.650 | 4,578.760 | 1.0002 | 26.0 |
| 64×16 | vllm_interleaved | 4,774.450 | 4,776.136 | 1.0004 | 14.3 |
| 256×4 | vllm_interleaved | 6,608.912 | 6,608.960 | 1.0000 | 25.8 |

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

本节按时间顺序记录第 0 阶段之后的模拟器改进与实验：容量、机制分析、T-PLANE、T-CMD、条带布局、`ocp` 拓扑、带宽档位、协议约束、T-CACHE、实验排查和第 8 轮审查。它们都在分支 `claude/gallant-maxwell-x9sgi0` 上分别提交，可以单独评估是否合入。

### 容量：每 plane 512 块

第 0 阶段每 plane 只配 64 块，逻辑容量约 1,560 万页，Llama-3.1-8B 在 32K 上下文、batch 16 时就会超出。这个上限来自配置，不是 MQSim 写死的。实测 MQSim 内存约为每物理页 31 B：

| 每 plane 块数 | 逻辑容量 | MQSim 峰值 RSS（1,024 plane） |
|---:|---:|---:|
| 64 | 1,560 万页（约 64 GiB 物理） | 0.57 GB |
| 256 | 6,240 万页 | 2.1 GB |
| 512 | 1.25 亿页（物理 512 GiB，与规范表 3 的数字相同；表 3 是用户容量，模型的逻辑容量扣除 7% OP 后为 476 GiB） | 4.2 GB |

默认值改为 512 块，GC 预算相应变为每 plane 124,159 页。两个 GC 边界测试仍固定使用 64 块，以保持运行时间不变。

### 机制分析模式

新模块 `hbf/analysis.py` 不运行 MQSim，只根据 CWDP 映射把每页放到 (die, plane, 偏移)，逐个 (step, layer) 计算：

- 读轮数 R：独立 plane 时为最忙 plane 的页数；同偏移联动时为每组需要的不同偏移个数的最大值；
- 最少轮数 R* = ⌈N/P⌉、plane 利用率 N/(R·P)、均衡损失 R/R*；
- 感测页数与感测放大：联动时，组内每个 plane 每轮都要感测，即使它没有要读的页；
- 64 B 突发传输字节：OCP 规范 §4.1 允许主机读突发为 64 B 到 4 KiB（64 B 的倍数，不跨页），所以主机链路上的传输可以小于整页；但 base die 与 core die 之间最小按 4 KiB 读写（§4.6 "Device Logical Unit Size"），较长的突发读由产品可选支持（§5.3.1 第 1 条）。因此这个指标只代表主机侧字节，NAND 侧仍是整页传输，感测始终是整页。

页在 plane 内的偏移有两种算法（`--placement`）。默认 `static` 按 OCP 规范：地址决定 bank、die、块和页（§4.6、图 4；§5.4.1 块内连续写、禁止跳地址；§11.4 无页级 FTL），偏移为 `lpa // 单元数`，并假定数据已在 HBF 中。`mqsim` 按 MQSim 的实际分配：每个 plane 按轨迹顺序，在每次写入、以及未写过的 LPA 首次被读时，取写前沿的下一页；加 `--prefill`（按地址顺序写满整个布局区间，含填充页）后两者相同，不加时稀疏读的页按首次读取的顺序紧凑排列，这在规范中不允许（读未写页返回错误 0x7）。一层超过子批上限被拆开时，`batch_rounds` 给出各子批轮数之和，这是与 MQSim 对照的口径；`rounds` 把整层当作一次突发。独立 plane 的轮数与已有指标 `max_unit_pages` 逐层相等，有测试核对。

**与 MQSim 逐层核对**：2 die × 8 plane，`vllm_interleaved`、`layer_contiguous` 和 `striped_contiguous`，含层数、头数不整除 plane 数的形状，independent 与 lockstep 两种规则，加或不加 T-CMD 合并窗口，共 16 组，全部先整体 prefill。物理时间线上每个写入页的 (块, 页) 都等于规范的静态偏移 `lpa // 单元数`；每层每 die 实际发出的读命令数落在 [静态分析轮数, 轮数 + 1] 之内，开合并窗口时正好相等。多 plane 配置下，读未写过的页会在运行前被拒绝（chip 当 plane 时偏移不影响时序，只计数）。

这一部分经历过两次修正：初版直接取 `lpa // 单元数`，但当时的 prefill 逐层写入、不写条带布局的填充页，MQSim 的位置因此与之不符（第二轮审查）；随后一度把分析改为跟随 MQSim 的分配，规范一致性审查（见下文 S1–S4）指出规范要求的是静态位置，于是改为按地址顺序写满整个区间，使 MQSim 与规范一致，分析默认回到静态位置。

**示例扫描**：Llama-3.1-8B 形状（32 层、8 个 KV 头、32K），先整体 prefill，再每层每头每张量均匀随机选页（种子 11），下表为 32 层合计的读轮数（括号内为均衡损失 R/R*），数据见 [evidence/stage1-analysis-sweep.csv](evidence/stage1-analysis-sweep.csv)：

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
2. 2 bit 格式读的页数是 BF16 的 1/7，轮数只少 1.7 倍：压缩和稀疏选择的收益在感测粒度上叠加不起来。（这是按存储页抽样的结果；按固定 token 块选择时，2 bit 连页数都基本不减，见下文"实验排查"。）
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

新增布局 `striped_contiguous`：按 `layer_contiguous` 的顺序排列区域，每层从相位 0 开始，各 (张量, 头) 区域之间留 P/(T·H) 页的间隔。区域长度是 P 的倍数时（如 32K BF16），相位正好是 t·P/T + h·P/(T·H)，即默认 K/V 时头之间错开 P/(2H)、V 与 K 相隔 P/2（SPLASH 的做法）。初版直接把每个区域补齐到这个相位，短区域（2 bit、KIVI、短上下文）因此要补大量空页，第 8 轮审查后改为固定间隔，每层补的页少于 2P（见"第 8 轮审查"）。填充页不读，但规范要求块内连续写，`--prefill` 会把它们一起写入。新增生成模式 `sink_recent`：每层每头每张量读前 `count` 个 token 加最后 `fraction` 比例的 token（StreamingLLM 式）。

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

### 规范带宽档位

规范表 4（p16）的三档用户带宽为 0.384、1.536、3.072 TB/s（UCIe 8、16、32 GT/s）。原先的配置检查要求 4 KiB 能被"2 × 每通道宽度"整除，16 通道时每通道 24、96、192 GB/s 都被拒绝。MQSim 的传输时间是 `bytes / width / 2` 个 2 ns 周期，按整数截断；现在 `Config.transfer_ns` 采用同样的算法，并用 `transfer_error` 给出偏差：

| 档位 | 每通道（16 通道） | 传输时间 | 精确值 | 误差 |
|---|---:|---:|---:|---:|
| 0.384 TB/s | 24 GB/s | 170 ns | 170.67 ns | −0.4% |
| 1.536 TB/s | 96 GB/s | 42 ns | 42.67 ns | −1.6% |
| 3.072 TB/s | 192 GB/s | 20 ns | 21.33 ns | −6.25% |

三档的单页读时延与"命令 + tR + R/B + 传输 + 主机"逐 ns 相等，有测试核对。`ocp` 拓扑有 32 条通道，三档为每通道 12、48、96 GB/s，误差 −0.4%、−1.6%、−1.6%。整除时（如默认的 64 GB/s）公式与原先完全相同。规范的档位指主机侧链路，MQSim 的通道是 NAND 侧总线，两者不是同一个量；1,024 GB/s 的默认值来自 SPLASH。

### 协议约束：命令额度、64 B 读与主机链路

HBF-EventLab（sjtu-epcc，按 OCP 规范实现的事务级模拟器）中观察到一个此前没有建模的机制：读的并行度不仅受 bank 限制，还受主机协议限制。规范依据：

- §5.3.1 第 1 条（p56）："All Read requests are 64-byte granularity; however, the host shall send 64 read commands to retrieve the complete 4KiB data with different AXI IDs. Or if product support burst read then send burst read with burst length (0-63)."；§4.1（p15）：读突发为 64 B 的倍数、64 B 对齐、不跨 4 KiB 页，可从页内任意 64 B 处开始。
- MOCS（§5.9.2.1 p70、§5.9.2.3 p71）：base die 命令队列能同时容纳的命令数，取值 1 或 256–16,384 的 2 的幂。该寄存器描述 "the Base Die UCIe Channel"，所以额度按主机通道计。一个 UCIe 通道内的多个虚拟 AXI 接口是否分摊额度，规范没有规定。
- §5.3.1 第 5a、5b、7、8 条（p57）：cache 命中的读立即返回；未命中时，前一次该 bank 的读完成后才感测；每个 bank 有两个 cache 缓冲。
- 主机链路（§5.1.2.1–5.1.2.3，p29–35）：每个 UCIe 模块的系统读写吞吐为 48、94、188 GB/s（档位 1、2、3）；档位 1 最多 8 个模块、堆叠高度 8（合计 0.384 TB/s），档位 2、3 最多 16 个。规范没有说明是否按方向计，这里按读方向计。

分析模式新增可选的 `Protocol(mocs, read_burst, read_unit, host_order, ucie_grade, group_sense)`（CLI：`--mocs`、`--read-burst`、`--read-unit page|wanted`、`--host-order fcfs|trigger`、`--ucie-grade 1|2|3`、`--group-sense arrived|whole`）：

- 每页的读命令数：整页读为 64 条（支持突发时 1 条）；只读需要的 64 B 段时为段数（支持突发时为连续段数）。
- 主机发送顺序：`fcfs` 把一页的全部命令一起排队，已放行的命令在该页感测期间占着额度，放不下的随后作为 cache 命中返回（HBF-EventLab 默认主机的做法）；`trigger` 先为每页发一条 64 B 读触发感测，等数据进入 cache 后再发其余命令，它们作为 cache 命中立即返回，感测期间每页只占 1 个额度。两种都符合规范。
- 轮数按主机通道贪心调度：每轮每个 bank（或联动组）最多感测一个偏移，每个主机通道该轮占用的额度不超过 MOCS。一页只要有一条命令进入队列就开始感测；放不下的其余命令在感测完成后进入，作为 cache 命中返回，不再占感测轮（§5.3.1 第 5a、7 条；HBF-EventLab 核对见下表）。联动组由 `group_sense`（`--group-sense`）决定，这是规范和 HBF-EventLab 都没有给出的建模假设：默认 `arrived` 只感测命令已到的页，同一偏移其余的页要再感测一次；`whole` 在第一条命令到达时感测整组（megaplane），其余页都是 cache 命中，`trigger` 主机每个偏移只需一条命令。每项额度相同时，结果等于下界 max(最长 bank 队列, ⌈项数 / ⌈MOCS / 每项额度⌉⌉)，有随机测试核对，复审时另用约 2.9 万例穷举核对（一次性脚本，未入库）；额度不均匀时（`wanted` 读、联动组）是可达到的上界。
- 链路下界：每个主机通道一轮 tR 最多传 档位吞吐 × tR 字节的读数据，`link_rounds` 为最忙主机通道的数据量除以这个值；"全部约束下的轮数" 取两者的较大者。
- `cwdp` 拓扑一律按规范的 16 个主机通道计（64×16、256×4 的 MQSim 通道按模 16 归并），否则额度会被放大 4 倍、16 倍。

**与 HBF-EventLab 核对**（`ocp` 几何：16 主机通道 × 4 die × 16 bank，4 KiB 交织，感测 4.096 µs，`--burst` 写、64 B 读；感测波次由统计 JSON 的 `bank_operations` 计出。最初几组探针的档位为 2 或 3，输入、配置（档位 3）、重跑脚本和重跑结果见 [evidence/eventlab/](evidence/eventlab/)，重跑的波次与下表一致）：

| 负载 | 设置 | HBF-EventLab | 本模型 |
|---|---|---|---|
| 128 页（lpa 0–127，每主机通道 8 页在 8 个 bank） | MOCS 256，自然顺序 | 8.43 µs，2 波（5 + 3 页） | 2 轮 |
| 同上 | MOCS 1024 | 4.54 µs，1 波 | 1 轮 |
| 同上 | MOCS 256，先发触发命令 | 4.31 µs，1 波 | 1 轮（trigger） |
| 32 页在主机通道 0 的 8 个 bank（每 bank 4 页） | MOCS 256，自然顺序 | 28.90 µs，7 波（每波最多 5 页） | 8 轮 |
| 同上 | MOCS 256，先发触发命令；或 MOCS 1024 | 16.60 µs，4 波（bank 下界） | 4 轮 |
| 8 页在主机通道 0 的 8 个 bank（`burst4096` 写入） | MOCS 1，自然顺序 | 33.17 µs，8 次感测 + 504 次 cache 命中 | 8 轮 |
| 同上 | MOCS 256，自然顺序 | 8.31 µs，2 波 | 2 轮 |

`fcfs` 下模型偏保守一轮左右：它让额度在整轮内占用，而 HBF-EventLab 每条命令的响应送达就释放额度，一波能多放一页。

**扫描**（`ocp`，Llama-3.1-8B 形状、32 层、32K；32 层合计轮数，括号内为利用率；数据见 [evidence/protocol-credit-scan.csv](evidence/protocol-credit-scan.csv)）：

| 负载 | 布局 | 无协议约束 | fcfs，MOCS 256 | fcfs，MOCS 1024 | MOCS 256 + 读突发 | trigger，MOCS 256 |
|---|---|---:|---:|---:|---:|---:|
| BF16，5% 均匀选页 | layer_contiguous | 215（0.24） | 971（0.053） | 256（0.20） | 215 | 215 |
| BF16，sink_recent | layer_contiguous | 512（0.032） | 512 | 512 | 512 | 512 |
| BF16，sink_recent | striped_contiguous | 32（0.52） | 384（0.043） | 96（0.17） | 32 | 32 |
| 2 bit，5% 均匀选页 | layer_contiguous | 75（0.087） | 166（0.039） | 75 | 75 | 75 |
| BF16 稠密，1 层 | layer_contiguous | 32（1.0） | 512 | 128 | 32 | 32 |

链路档位（trigger，MOCS 256）：以上稀疏负载在三个档位下都不受链路限制；稠密一层在档位 1 下需要 44 轮（每主机通道每轮 64 页 × 4 KiB = 256 KiB，48 GB/s 要 5.46 µs，超过 tR），档位 2、3 仍为 32 轮。档位 1 的数字按 16 个主机通道逐通道计算；规范的档位 1 最多 8 个 UCIe 模块，16 通道的 `ocp` 几何在档位 1 下不是规范配置。

结论：

1. 命令额度是否成为瓶颈，**取决于主机的发送策略**，不是规范强加的限制。主机若把每页的 64 条命令一起排队（fcfs），MOCS 256 下稀疏读轮数增加约 4.5 倍、条带布局的优势几乎消失；主机若先发一条触发感测（trigger），或产品支持读突发，限制消失；只把 MOCS 提到 1024 时，均匀选页基本恢复（256 对 215），但页集中在每主机通道少数 bank 的负载仍受额度限制（条带 sink_recent 96 对 32，稠密一层 128 对 32）。条带布局在 fcfs 下的损失有一部分来自对齐：间隔为偶数时，一层各区域的同一块落在同一个主机通道上；`striped_staggered` 把间隔改为奇数，sink_recent 从 384 降到 288 轮（第 8 轮审查）。
2. 只读需要的 64 B 段只在 fcfs 下有作用（减少命令），均匀选页时帮助有限，sink_recent 的窗口边缘页受益（384 → 256 轮）。
3. 稠密读在档位 1 下受主机链路限制，档位 2、3 下仍是 bank 受限。

模型的简化：一轮按 tR 计，额度在轮末全部释放；cache 命中不计时间（MOCS 1 时每页 63 次命中要串行往返，HBF-EventLab 中约 0.8 ns 一次，若真实往返在 10 ns 量级，每轮约多 15%）；主机可在同一主机通道内任意重排请求；普通读和批量读窗口（§5.3.2）只影响每轮延迟，不改变轮数，没有建模；每 bank 两个 cache 缓冲在"每轮每 bank 感测一页、回传快于 tR"时不构成额外限制，档位 1 下的稠密读除外。

**审查记录**（协议约束上线后，由子代理对照规范并用 HBF-EventLab 复现审查）：

| # | 问题 | 处理 |
|---|---|---|
| P1 | 初版假定一页的 64 条命令在感测期间都占额度，并据此得出"MOCS 256 时额度先于 bank 成为瓶颈"；HBF-EventLab 实测，先发触发命令时没有额外轮次 | 新增 `host_order`（fcfs / trigger），结论改写为"取决于主机发送策略" |
| P2 | 需要额度超过 MOCS 的项被一次放行，出现"只受额度限制的轮数"大于"全部约束下的轮数" | 初版改为超额项分几轮放行，每段各算一轮感测；之后的实验发现这不对（见下一条），已改为第一条命令进入即开始感测、其余为 cache 命中，额度下界相应改写 |
| P7 | （实验中发现）分段放行让每段都重新等一次 tR：MOCS 1 时每页算 64 轮。HBF-EventLab 在 MOCS 1 下 8 页只用 8 次感测（33.2 µs），其余 504 条命令都是 cache 命中；MOCS 256 下 7 页 × 40 条命令一波完成（4.22 µs） | 改为一页只感测一次；`wanted` 读允许部分放行，sink_recent 的 fcfs 结果 288 → 256 轮；新增 HBF-EventLab 对照测试和"额度下界不超过轮数"的随机测试 |
| P8 | （复审）P7 初版把"一次感测"也用于联动组，但 HBF-EventLab 没有联动组，无从验证；`trigger` 下联动项的额度也与该语义不一致 | 新增 `group_sense`：默认 `arrived`（联动组恢复修改前的结果），`whole` 作为可选假设并改为每个偏移一条触发命令 |
| P3 | `cwdp` 拓扑把每个 MQSim 通道当作主机通道，64×16、256×4 的额度被放大 4、16 倍 | 一律按 16 个主机通道计 |
| P4 | 额度不均匀时贪心不保证最优（小规模穷举中 6% 的实例多 1–2 轮） | 文中注明为上界 |
| P5 | 未记录 HBF-EventLab 的配置 | 已补充 |
| P6 | 没有考虑主机链路带宽 | 新增 `ucie_grade` 链路下界 |

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

### 实验排查：MQSim 对照、选择粒度与发现的问题

在 `ocp` 拓扑上补跑了一轮实验，核对分析模式与 MQSim 是否一致，并找不合理的地方。

**MQSim 与分析轮数**（`ocp`，2 层、32K、`--fraction 0.05`（均匀选页与 sink_recent 的近期窗口都用它）、整体 prefill，T-CMD 每多一个 plane 0 ns、合并窗口 1 µs、T-CACHE；数据见 [evidence/mqsim-vs-rounds.csv](evidence/mqsim-vs-rounds.csv)）：

| 布局 | 负载 | 分析轮数（独立 / 联动 16） | MQSim independent（µs/层，÷ tR） | MQSim lockstep（µs/层，÷ tR） |
|---|---|---:|---:|---:|
| layer_contiguous | 5% 均匀选页 | 6–7 / 26 | 27.9–32.3（6.8–7.9） | 115.7–116.0（28.2–28.3） |
| layer_contiguous | sink_recent | 16 / 16 | 72.4（17.7） | 72.4（17.7） |
| striped_contiguous | 5% 均匀选页 | 6–7 / 25 | 27.9–32.4（6.8–7.9） | 111.7–111.8（27.3） |
| striped_contiguous | sink_recent | 2 / 7 | 13.2（3.2） | 32.9（8.0） |

MQSim 的每层时延约为（分析轮数 + 0.8 至 2.3 轮）× tR，多出的是命令、最后一轮的传出和 1,024 GB/s 下核心通道的传输。两种规则的排序和倍数与分析一致。稠密读时核心通道带宽开始起作用：1 层 8K 稠密读的分析下界为 8 轮（32.8 µs），MQSim 在 1,024 GB/s 下为 42.3 µs，3,072 GB/s 下为 37.6 µs。分析模式只计主机链路，不计核心通道带宽；`ocp` 预设沿用 SPLASH 的 1,024 GB/s，不是规范的三档之一（384、1,536、3,072），需要时用 `--aggregate-gb-s` 指定。

**选择粒度**：原有的 `uniform_pages` 按存储页抽样，且每个张量各自抽。这有两个后果：一是选择单位随压缩变大，2 bit 的一页有 128 个 token，BF16 只有 16 个；二是同一头的 K 和 V 读的是不同的 token。真实的稀疏方法（如 Quest）按固定的 token 块选择，同一头的 K、V 读同一批 token。新增 `uniform_tokens` 模式（`--select-tokens`，默认 16）：每个 (step, 层, KV 头) 选 `fraction` 的 token 块，所有张量读这些 token。Quest 按 query 头选择，GQA 下一个 KV 头要读同组各 query 头所选块的并集，相当于更高的比例，这里没有区分。对比如下（`ocp`、32 层、32K、8 个 KV 头，32 层合计，5 个种子的平均；最小、最大值见 [evidence/selection-granularity.csv](evidence/selection-granularity.csv)）：

| 格式 | 选择 | 比例 | 读页 | 独立 plane 轮数（layer_contiguous / striped / staggered） | 联动 16 轮数（同左） |
|---|---|---:|---:|---:|---:|
| BF16 | 按页 | 1% | 10,752 | 103.4 / 104.4 / 106.6 | 325.4 / 323 / 333.8 |
| BF16 | 按 16-token 块 | 1% | 10,752 | 153.6 / 97.8 / 104.6 | 407.6 / 298.4 / 309.8 |
| BF16 | 按页 | 5% | 52,736 | 215.2 / 214.4 / 217 | 779.6 / 795.2 / 790.2 |
| BF16 | 按 16-token 块 | 5% | 52,736 | 298.8 / 206.6 / 213 | 861.6 / 763 / 775.6 |
| 2 bit K、V | 按页 | 1% | 1,536 | 53.8 / 49 / 49 | 91.6 / 95 / 93.8 |
| 2 bit K、V | 按 16-token 块 | 0.1% | 1,530 | 74.4 / 43.6 / 47 | 121.6 / 83.8 / 85.2 |
| 2 bit K、V | 按 16-token 块 | 0.3% | 3,548 | 98.8 / 62.4 / 66 | 128 / 121.8 / 122.2 |
| 2 bit K、V | 按 16-token 块 | 1% | 10,382 | 128 / 85.4 / 91.2 | 128 / 159.8 / 159.8 |
| 2 bit K、V | 按 16-token 块 | 5% | 44,314 | 128 / 128 / 134 | 128 / 160 / 160 |
| KIVI（K 2 bit、V BF16） | 按页 | 1% | 6,144 | 82 / 83 / 84.4 | 218.4 / 219.8 / 217 |
| KIVI（K 2 bit、V BF16） | 按 16-token 块 | 1% | 10,567 | 100.4 / 100.2 / 99.6 | 257 / 264.2 / 262.6 |
| KIVI（K 2 bit、V BF16） | 按 16-token 块 | 5% | 48,525 | 188 / 193 / 196 | 494.8 / 519 / 521.6 |

按 16-token 块选择时，读放大约为 BF16 1.0、2 bit 7.7（1%）和 6.7（5%）、KIVI 1.75（1%）。

结论：

1. 按 token 块选择时，**2 bit 压缩几乎不减少读页数**（1% 时 10,382 对 10,752），每页只用到约 1/8。"2 bit 读的页是 BF16 的 1/7"只在按存储页抽样时成立，相当于选择粒度随压缩放大 8 倍。上文示例扫描的第 2 条结论据此修正：压缩与稀疏的收益在感测粒度上叠加不起来，按 token 选择时连页数都基本不减。
2. **BF16 下，K、V 同选让 `layer_contiguous` 变差，条带布局的作用变明显**：32K BF16 时，同一块的 K、V（以及各头的同一块）都在同一个 plane，一起读就排在一起；独立 plane 轮数从按页抽样的 103 升到 154（5% 时 215 → 299），条带布局把各头错开、V 放到半个器件之外，保持 98（207）。按页抽样时两种布局几乎没有差别，掩盖了这一点。这一结论**只对 BF16 成立**：KIVI 的 K、V 每页 token 数不同，`layer_contiguous` 本来就不对齐，条带布局与它持平（1% 时 100 对 100，联动时 264 对 257）；2 bit 独立 plane 在比例 ≤ 1% 时条带布局更好（1% 时 128 对 85），联动时更差（1% 时 128 对 160）。（第 8 轮审查前的条带规则会把短区域补齐到固定相位，2 bit 联动曾为 193，KIVI 也更差；数字已按新规则重算，见"第 8 轮审查"。）
3. 2 bit、`layer_contiguous` 时每层 4,096 页，每个 plane 正好 4 页：头 h 与头 h+4 的同一块 K、V。K、V 同选使每个被选的块在一个 plane 上占 2 页，若头 h 与头 h+4 恰好选中同一个 128-token 页，该 plane 就要读满 4 页。比例 ≥ 1% 时几乎每层都会发生，轮数饱和在 128（每层 4 轮）；比例更低时轮数下降（0.1% 为 74，0.3% 为 99）。

**发现并已修正的问题**：

| # | 问题 | 后果 | 处理 |
|---|---|---|---|
| X1 | 协议模型对放不下的项分段放行，每段都算一轮感测 | MOCS 1 时每页算 64 轮（HBF-EventLab 实测每页约 1 个 tR） | 见上文 P7、P8 |
| X2 | `uniform_pages` 的选择粒度随压缩变化，K、V 各自抽样 | 高估压缩收益；BF16 下低估条带布局的收益 | 新增 `uniform_tokens`；原模式保留，文档注明 |
| X3 | MQSim 结果表中的"MQSim 估算"按每页一条命令计算 | 多 plane 拓扑下 MQSim 与估算之比在 0.39–4.0 之间：最忙 plane 决定时延时（独立 plane、命令开销 0）接近 1，通道项决定时（每页记一条命令，而 MQSim 一条命令覆盖整个 die）或 lockstep 时偏差很大 | 多 plane 时不给出估算（显示"—"），用分析轮数代替；chip 当 plane 时不变 |
| X4 | 报告的机制分析表只有整层轮数 | 层被拆成子批时与 MQSim 不可比（8192 页按 100 页一批：整层 8 轮，子批合计 82 轮），而协议约束表按子批计 | 表中增加"子批轮数"列 |
| X5 | `same_unit`、`hot_unit` 等模式要的页超出布局时报"Invalid logical access" | 看不出原因 | 改为说明哪个单元的第几页超出布局 |

未修改、只记录的事项：分析模式不计核心通道带宽（见上）；`--topology 24x4` 等通道数不整除 1,024 GB/s 时报错，需要同时给出 `--aggregate-gb-s`。

### 第 8 轮审查：四个子代理的实验与审查

四个子代理分别负责 MQSim 实测、分析模式实验与代码审查、研究方向调研、文档与规范引用核对。下面是核实后的结论和处理。

**MQSim 实测**：在 `ocp`、`16x1 --planes 64`、`4x1 --planes 16` 和三种 chip 当 plane 的拓扑上，共跑了约 60 次 MQSim。覆盖多步、子批拆分、`uniform_tokens`、2 bit 与 KIVI、`write_then_read`、四档带宽、两种 plane 规则，以及 T-CMD、T-CACHE 的开关组合。

- 物理时间线上没有发现违例：同通道上命令与数据不重叠，同一 die 的感测不重叠，数据传出不早于就绪。
- 没有一层低于物理下界。
- 小规模时延与手算逐 ns 相等，例如 `4x1 --planes 16` 的 13 轮为 1 + 13 × 4,406 + 16 + 1 ns。

**发现与处理**：

| # | 来源 | 问题 | 处理 |
|---|---|---|---|
| R1 | MQSim | T-CACHE 的 cache 优先调度只允许向 cache 就绪的 chip 发读命令，完全空闲的 chip 要等排队的数据全部传完。chip 当 plane、每通道 64 个 chip 时，开 cache read 反而更慢：16x64 稠密 8K 时为 200,778 ns，不开为 184,612 ns；稀疏读也慢 7–9% | 空闲 chip 在这一轮也可以发读命令，同一用例降到 181,264 ns；写和擦除仍不插到排队的数据前面。其他配置逐 ns 不变；新增回归测试 |
| R2 | 分析 | 条带布局的相位不考虑区域长度：短区域被补齐到固定相位，2 bit 32K 时 span 是页数的 4.2 倍（32 头时 16 倍），联动轮数虚高；KIVI 的 K 区域挤在约一半 plane 上，稠密读为下界的 2 倍 | 改为每层从相位 0 开始、区域之间留固定间隔。32K BF16 的相位不变，每层补的页少于 2P，稠密读至多比下界多 2 轮。2 bit 与 KIVI 的数字已重算 |
| R3 | 分析 | 间隔为 64 或 512（16 的倍数）时，一层各区域的同一块都在同一个主机通道上，fcfs 额度下条带布局的收益被抵消 | 新增 `striped_staggered`（奇数间隔）：fcfs、MOCS 256 下 sink_recent 从 384 降到 288 轮（等于均衡下界）；均匀选页 971 对 986，不占优。不改默认布局 |
| R4 | 方向调研、分析 | 只给 `--ucie-grade`、`--read-unit` 等参数时，会隐式施加 MOCS 256、fcfs 的额度限制 | 不给 `--mocs` 时不限额度 |
| R5 | MQSim | 分析模式不计核心通道带宽：384 GB/s 稠密读时 MQSim 为 23.6 tR，分析只给 8 轮 | 新增"核心通道下界轮数"（`channel_rounds`，每通道页数 × 传输时间 ÷ tR）：ocp 稠密 8K 在四档下为 22、8、6、3 轮 |
| R6 | MQSim | 合并窗口取 1 µs 是纯开销：一个子批同时到达，1 ns 的窗口就能完全合并（命令数相同），每层却多出 0.24 tR | README 示例改为 `--coalesce-ns 1`；已有表格保留 1 µs 的口径 |
| R7 | 分析 | `ceil(blocks * fraction)` 受浮点误差影响，100 × 0.07 会选 8 块 | 先取整到 9 位小数再向上取整 |
| R8 | 分析 | 按单元生成的模式在条带布局上会取到填充页并报错 | 跳过填充页，取该单元的下一页 |
| R9 | 分析 | base 不是单元数的倍数时，prefill 后 MQSim 的位置与静态偏移不一致，ocp 的 `unit_lpa` 也会出错 | 两处都要求 base 对齐，否则报错 |
| R10 | 文档、MQSim | 报告抬头看不出多 plane 配置和开关；prefill 行写"读页 0"却有时延；`pip install` 的包缺 `patches/series`；`--layers 0` 时打印 traceback；输入错误会留下空的输出目录 | 都已修正 |
| R11 | 文档 | T9 不拆分表的墙钟时间与证据文件不符；协议结论 1 把"MOCS ≥ 1024"说成限制基本消失；档位 1 最多 8 个 UCIe 模块没有写；若干过时描述、规范小节号与种子 | 按证据文件和规范原文更正 |

**建模局限，未修改**：

- T-CACHE 要求 die 空闲后才接收下一条命令，所以开 cache 时每轮多付一次命令时间（约 310 ns，即 tR 的 7.6%）。这与 ONFI 的 cache read 一致；HBF 的 base die 有命令队列，可以在感测期间预先送达命令。
- 子批拆分后，每个子批各付一次感测加传输的尾巴（`batch_rounds` 已反映）。
- 通道数少于 16 的 cwdp 拓扑按通道数计主机通道（8x128 只有 8 份 MOCS 额度），这样联动组不会跨主机通道。
- 在 64x16、32x32 上，一个联动组就是一条 MQSim 通道，条带布局在 lockstep 下不起作用。"条带在 lockstep 下也有效"依赖拓扑。

## 第 2 阶段 A：带宽不足与压缩收益

这一阶段回答研究计划五个问题中的第 1 项（HBF 带宽不足的表现）、第 2 项（KV 压缩能把有效带宽提高多少），以及第 5 项中读路径上的冲突。写路径上的冲突和寿命（第 3、4 项）在阶段 B。

为此新增了四块能力，用法见 README：
- numpy 分析引擎；
- 读时间估计（`--timing`）；
- KV 存储格式（`--format`、张量窗口与层级）；
- 系统级评估（`--model`、`--gpu`、`--batch`、`--stacks`）与扫描脚本（`python -m hbf.sweep`）。

设计由两个子代理先讨论定案，一个负责方案，一个负责参数调研。实现分工：GPU 模型和扫描脚本由子代理完成，其余由主代理完成。审查与修正见本节末尾。

### 方法与设定

**读时间**
- `est_ns` 按核心通道逐轮累加命令、感测、R/B 和传输，并计入 cache read 的重叠。
- `bound_ns` 是理想下界。
- 估算用第 8 轮保存的 79 个 MQSim 层逐层核对过（[evidence/time-calibration.csv](evidence/time-calibration.csv)）：在 HBF 命令模型（`--read-plane-ns 0 --coalesce-ns 1 --cache-read`）、tR = 4,096 ns 下，误差不超过 4.7%，估算从不比 MQSim 高出 0.1% 以上。
- 没有合并窗口的那一组少算约 14%：MQSim 让第一页单独下发，多出一轮。
- 其余设定不在校准范围内，报告中会标为"未经验证"。

**系统级**
- GPU 用 B200 预设：BF16 稠密 2,250 TFLOP/s，HBM 8 TB/s，180 GB。
- 效率系数是假设：compute_eff 0.5，memory_eff 0.8。
- 线性层和 lm_head 按 roofline 计时。注意力计算量按被注意的 K、V 元素数计算。
- TPOT 按"每层 pre + max(HBF 读, 注意力) + post"累加（`serial`、`stream`）。
- HBM 基线：同一格式的 KV 全部从 HBM 读，并检查 HBM 放不放得下。

**批**
- `--batch B` 把一个 stack 上的 ceil(B/N) 个请求按层优先排成额外的 KV 头。各请求上下文相同，这是批的近似。
- `--stacks N` 把请求分到 N 个 stack 上。

**模型与配置**
- 模型取 Llama-3.1-8B 和 70B 的形状。
- 拓扑为 `ocp`，使用 HBF 命令模型，除另注外为条带布局。

数据见 [evidence/phase2a/](evidence/phase2a/)（四个扫描的汇总表）。完整表格由 `python3 evidence/scripts/phase2a_tables.py` 生成，网格在 [evidence/grids/](evidence/grids/)，四个扫描共 1,140 个点，3 进程并行约 15 分钟。

### 第 1 项：带宽不足的表现

单个 stack，B200，稠密读，tR 4,096 ns，核心通道合计 1,024 GB/s，每步时间（ms）：

| 模型 | 上下文 | B | HBF 读（估算） | 理想下界 | TPOT（HBF） | TPOT（KV 在 HBM） | HBF/HBM | GPU 等待占比 | HBM 放得下 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8B | 8K | 1 | 1.26 | 1.05 | 3.60 | 2.51 | 1.43 | 0.36 | 是 |
| 8B | 8K | 32 | 36.23 | 33.55 | 38.57 | 7.71 | 5.00 | 0.94 | 是 |
| 8B | 32K | 4 | 18.18 | 16.78 | 20.52 | 5.03 | 4.08 | 0.89 | 是 |
| 8B | 32K | 32 | 144.51 | 134.22 | 146.85 | 23.82 | 6.17 | 0.98 | 是 |
| 8B | 128K | 16 | 288.88 | 268.44 | 291.23 | 45.29 | 6.43 | 0.99 | 否 |
| 70B | 8K | 1 | 3.15 | 2.62 | 24.87 | 22.14 | 1.12 | 0.13 | 是 |
| 70B | 32K | 4 | 45.45 | 41.94 | 67.16 | 28.43 | 2.36 | 0.68 | 否 |
| 70B | 128K | 4 | 180.80 | 167.77 | 202.52 | 48.56 | 4.17 | 0.89 | 否 |

1. **单个 stack 读 KV 的时间 ≈ 读的字节数 ÷ 1 TB/s 乘一个开销系数，所有点都受 bank 感测限制。**
   - 开销系数是估算与理想下界之比，即命令和最后一轮传出的开销：KV 较大时约 1.08，8K、B=1 这类读页少的点到 1.20。
   - 核心通道从 1,024 GB/s 提到 3,072 GB/s，读时间只少 0.1 ms 左右。1,024 GB/s 恰好等于 1,024 个 bank × 4 KiB ÷ 4.096 µs，降到 384 GB/s 才会受核心通道限制（第 1 阶段"规范带宽档位"）。
   - tR 减半，读时间也大约减半。例如 8B、32K、B=1、核心通道 3,072 GB/s：4.55 → 2.46 ms（1.85 倍）。
   - 所以在 tR = 4.1 µs 的假设下，瓶颈在 NAND 感测，不在 UCIe 档位 3。
2. **TPOT 由 HBF 主导。**
   - 每步读的 KV 达到约 4 GiB（8B：8K、B=4，或 32K、B=1）时，GPU 有 2/3 以上的时间在等 HBF；1 GiB（8K、B=1）时为 36%。128K、B=16 时达到 99%。
   - 与 KV 放 HBM 相比，单个 stack 慢 1.1–6.4 倍。上限约为 HBM 有效带宽 6.4 TB/s 与单 stack 约 1 TB/s 之比；tR = 8.2 µs 时可到 12 倍。
   - 70B 的权重读取时间长，KV 较小时比值接近 1。
3. **HBF 的价值在容量。**表中"HBM 放得下 = 否"的点，KV 全放 HBM 已经不可行，例如 8B 的 128K、B=16，以及 70B 的 32K、B=4。
4. **每个 GPU 配 8 个 stack（SPLASH 的设定）时，HBF 与 HBM 基线大致相当。**

   | 模型 | 上下文 | B | HBF 读 | TPOT serial/stream | TPOT prefetch:1 | TPOT（KV 在 HBM） | HBM 放得下 |
   |---|---:|---:|---:|---:|---:|---:|:---:|
   | 8B | 32K | 32 | 18.18 | 20.52 | 24.38 | 23.82 | 是 |
   | 8B | 128K | 64 | 144.51 | 146.85 | 178.65 | 174.14 | 否 |
   | 70B | 32K | 64 | 90.56 | 112.28 | 130.20 | 129.09 | 否 |

   - 在 serial/stream 下，HBF 比 HBM 快 1–16%，但这部分来自效率不对称：HBM 按 0.8 折算，HBF 用的是带命令开销的估算。所以应理解为同一量级。
   - prefetch 反而比 serial 慢。预取的数据要先写进 HBM 再读出来，读回的流量和 KV 全放 HBM 一样多。只有注意力直接消费 HBF 送来的页（`stream`）时，HBF 的带宽才能叠加到 HBM 之上。这一点决定了 HBF 的收益能不能兑现，需要在 GPU 一侧确认可行性。
5. **布局与协议（8B、32K、B=4、单 stack，3 个种子的平均）。**

   | 读模式 | 布局 | 协议 | HBF 读 | bank 轮次利用率 | 瓶颈 |
   |---|---|---|---:|---:|---|
   | 稠密 | 条带 | 不限 | 18.18 | 1.000 | bank |
   | 稠密 | vllm_interleaved | 不限 | 288.77 | 0.062 | bank |
   | 稠密 | 条带 | MOCS 256，fcfs，档位 2 | 288.76 | 1.000 | 额度 |
   | 稠密 | 条带 | MOCS 256，trigger，档位 2 | 18.18 | 1.000 | bank |
   | 稠密 | 条带（`ocp-8ch`） | MOCS 256，trigger，档位 2 | 36.23 | 1.000 | bank |
   | 稠密 | 条带（`ocp-8ch`） | MOCS 256，trigger，档位 1 | 48.22 | 1.000 | 链路 |
   | 5% 按块选择 | 条带 | 不限 | 3.41 | 0.532 | bank |
   | 5% 按块选择 | 层连续 | 不限 | 4.34 | 0.417 | bank |
   | 5% 按块选择 | vllm_interleaved | 不限 | 28.92 | 0.062 | bank |
   | 5% 按块选择 | 条带 | MOCS 256，fcfs，档位 2 | 31.29 | 0.532 | 额度 |
   | 5% 按块选择 | 条带（`ocp-8ch`） | MOCS 256，trigger，档位 1 / 2 | 5.68 | 0.637 | bank |

   利用率一列是按 bank 轮次计的利用率，不含协议约束；受额度限制的两行实际利用率为 1/16 左右（稠密 0.0625）。依 §5.1.2.1，UCIe 档位 1 最多支持 8 个主机通道（`ocp-8ch`：512 bank、512 GB/s 核心通道带宽），稠密读为 48.22 ms：相对同一 8 通道几何下的档位 2/3（36.23 ms）慢 33%，相对 16 通道 `ocp`（18.18 ms）慢 2.65 倍。

### 第 2 项：压缩后带宽提升多少

条带布局，128K，B=8，单 stack。表中每格为"HBF 读时间加速比 / TPOT 加速比"（相对 BF16），3 个种子的平均：

| 读模式 | BF16 HBF 读 ms | fp8 | int4 | kivi4 | kivi2 | think | kivi2 + think | quest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 稠密 | 144.51 | 2.00 / 1.97 | 3.74 / 3.58 | 3.19 / 3.08 | 5.31 / 4.97 | 1.33 / 1.33 | 7.05 / 6.43 | 0.94 / 0.94 |
| sink + 近期 5% | 7.46 | 1.97 / 1.60 | 1.86 / 1.54 | 0.96 / 0.97 | 2.40 / 1.80 | 1.01 / 1.01 | 2.20 / 1.71 | 0.36 / 0.42 |
| 按 16 token 块选 1% | 2.97 | 1.00 / 1.00 | 0.70 / 0.81 | 0.61 / 0.74 | 0.62 / 0.75 | 0.97 / 0.98 | 0.64 / 0.76 | 0.24 / 0.37 |
| 按 16 token 块选 5% | 10.48 | 1.04 / 1.03 | 0.90 / 0.91 | 0.69 / 0.73 | 0.74 / 0.78 | 1.00 / 1.00 | 0.84 / 0.86 | 0.53 / 0.58 |
| 按 128 token 块选 5% | 9.74 | 1.73 / 1.52 | 2.18 / 1.77 | 2.03 / 1.69 | 2.75 / 2.05 | 1.22 / 1.17 | 2.79 / 2.07 | 0.50 / 0.55 |

1. **稠密读：读时间按字节比例缩短。**
   - kivi2 每 token 96 B，其中 32 B 是 16 bit 的 scale 和 zero，所以收益是 5.3 倍而不是 8 倍。
   - TPOT 的收益略低，因为线性层和反量化的计算量不变。
   - 32K 和 B=1 时规律相同（见完整表格）。
2. **按 16 token 块做稀疏选择时，压缩几乎没有收益，甚至变慢（kivi2 选 1% 时为 0.62）。**
   - 压缩后，一页 2 bit 的 K 装 128 个 token，选中的 16 个 token 仍要读整页，读放大从 1 变成 8。
   - 量化参数还各占一组页，所以每个选中的块要读 4 页，BF16 只要 2 页。
   - 这是"压缩"和"按页选择"在感测粒度上的冲突，第 1 阶段的按页抽样已有迹象。
3. **把选择块放大到 128 token，即压缩后一页的 token 数，收益回到 2–2.8 倍。**压缩格式要和选择粒度一起设计，有两种做法：一是选择块不小于压缩后一页的 token 数；二是把量化参数和数据放进同一页。后者是布局问题，留作后续工作。
4. **Quest 的页摘要每步都要整体读。**
   - 在 1% 预算下，摘要的读取量（K 的 12.5%）比被选中的数据还大，读时间是不读摘要时的 4 倍（0.24）。
   - 摘要放 HBM 也不便宜：每 token 32 B，8B 模型 128K 上下文每个请求约 1 GiB。
   - 这类元数据应该放在 HBM，或者做得更小，或者分层选择。
5. **sink + 近期窗口模式下，收益随布局和格式变化，规律不明显。**例如条带布局下 kivi2 为 2.40、kivi4 为 0.96；层连续布局下 kivi2 为 1.53、fp8 为 1.00（见完整表格）。近期窗口只占少数页，轮数由最忙的 bank 决定，与页数的关系受对齐影响很大。

### 第 5 项（读路径）：现有 KV cache 方案直接用于 HBF 的冲突

| 冲突 | 代价（8B，单 stack） | 依据 |
|---|---|---|
| vLLM 式块交错布局与静态地址映射：每层只用到 1/16 的 bank | 稠密读慢 16 倍，5% 选择慢 8.5 倍 | item1-layout-protocol |
| 主机把一页的 64 条 64 B 命令一起排队（fcfs），MOCS 256 | 稠密读慢 16 倍，5% 选择慢 9.2 倍；先发触发命令则没有代价 | 同上 |
| 16 token 选择块对上压缩后的大页 | 选 1% 时整数和 KIVI 格式慢 1.4–1.6 倍（fp8、think 约持平）；选 5% 时慢 1.0–1.4 倍 | item2 |
| Quest 页摘要每步整体读 | 1% 预算下读时间是不读摘要时的 4 倍 | item2 |
| UCIe 档位 1（最多 8 通道、每通道 48 GB/s，§5.1.2.1） | 稠密读相对同 8 通道几何（`ocp-8ch`）档位 2/3 慢 33%（48.22 对 36.23 ms）；相对 16 通道 `ocp`（档位 2/3）慢 2.65 倍（48.22 对 18.18 ms） | item1-layout-protocol |
| 预取到 HBM 再读 | HBF 的带宽被 HBM 读回抵消 | item1-system |

写路径上的冲突放在阶段 B。初步结果（`--check-writes`）：
- vLLM 式块池和按层优先预留，在追加生成 token 后都违反 §5.4.1 的块内顺序写；
- 多个请求逐个写 prefill 时，按层优先的布局同样违规；
- 每个请求独占整段（`dedicated`）和全局写前沿（`packed`）是合法的。

### MQSim 核对

[evidence/scripts/mqsim_confirm.sh](evidence/scripts/mqsim_confirm.sh) 在 `ocp` 上跑了 15 个代表点，除批近似那一点为 1 层外都是 2 层，先整体 prefill，共 29 层，逐层与估算对照（[evidence/phase2a/mqsim-confirm.csv](evidence/phase2a/mqsim-confirm.csv)）。覆盖的配置：
- 新格式：int4、kivi2、quest；
- 批近似：32K、B=8，每层 262,144 页；
- 按块选择：1% 和 5%，16 和 128 token 块；
- 三档核心通道带宽：384、1,024、3,072 GB/s；
- 条带错开布局。

结果：
- 校准范围内，估算误差在 −4.7% 到 0 之间，从不高于 MQSim。
- 超出校准范围的两组：tR = 2,048 ns 的稠密读低估 5.3%；tR = 8,192 ns 的稀疏读误差在 0.6% 以内。
- 每次 MQSim 用时 13–15 s，峰值内存 4.15 GB。

### 设计讨论与审查记录

**设计讨论**（实现前，规划子代理）
- 先做 M7 提速，再把读时间放进分析引擎，因为它依赖逐 die、逐通道的轮次结构。
- 压缩格式用 token 区间窗口和存储层级来表达，例如 KIVI 的残差窗口放在 HBM。
- GPU 模型做成作用于每层行的纯函数，可以对已有运行重新评估。
- 校准用逐轮流水线公式，不用"每层常数开销"。后者无法同时拟合稀疏读和受核心通道限制的稠密读。

**参数调研**（子代理）
- 规范对 tR、编程和擦除时间、块大小、P/E 上限、读干扰阈值都写"由产品自定"，厂商也未公开。
- 现有 HBF 时序都来自 SLC 研究芯片（tR 约 4 µs）。
- 按档位 3 的 3.072 TB/s 和 1,024 个 bank 反推，需要 tR ≈ 1.4 µs，快于已发表的任何 SLC。
- 扫描范围见 params 汇总（阶段 B 一并入库）。

**审查**（子代理）

| # | 问题 | 处理 |
|---|---|---|
| H1 | `--batch` 配 `--access-csv` 时，只分析了请求 0，却按 b 个请求计算，TPOT 低估约 4 倍 | 导入的访问记录必须覆盖每个请求的 KV 头，否则报错 |
| H2 | 对没有系统元数据的运行重新评估时，默认 b = ceil(B/N)，同样低估 | 从 meta.json 的形状推出分析批，形状必须严格等于"模型 KV 头 × 分析批" |
| M1 | 只读 HBM 层的步会让报告崩溃；两个引擎在这类层上行数不同 | 报告对缺失值显示 "—"；不加 `--timing` 时这类层不出行，与参考实现一致 |
| M2 | 量化参数张量被计入被注意的 token，均匀按页抽样时注意力 FLOPs 高估约 3.4 倍 | 新增 `aux` 角色；按每个数据张量的 token × 维数累计 `attended_elements` |
| M3 | HBF 与 HBM 的效率折算不对称，没有说明 | 报告写明：公平的比较是 est 对 HBM；bound 对 HBM 偏向 HBF |
| L1–L5 | 不给 `--model` 时 `attn_dims` 错误；tR ≠ 4,096 等情况未标"未经验证"；晚出错时留下半个运行目录；每步的块计数一直保存；文档细节 | 逐条修正并加测试 |
| S1 | 扫描的有效带宽把 KIVI 在 HBM 的残差也算作 HBF 读，短上下文高估 50% | 新增 `hbf_elements`，有效带宽只计 HBF 读的元素 |
| S2 | 工作进程被系统杀掉时扫描会永远等待 | 改用 ProcessPoolExecutor，进程池损坏时把未完成的点记为错误 |
| S3 | 并行组不含 1 时汇总为空却标记为正常 | 记为不可行并给出原因 |

修正后，审查子代理做了两组随机核对：590 例引擎对照、1,500 例需求量暴力复算，都没有发现差异。

### 局限

- GPU 计算用 roofline 和假设的效率，没有实测。
- 批是"层优先、上下文相同"的近似，各请求按需分配、上下文不同的情况见阶段 B。
- 多 stack 时按请求划分。B < N 时有 stack 空闲；单个请求跨 stack 交织的情况没有建模。
- tR、核心通道带宽等器件参数规范未给出，结论按 tR = 4.1 µs、1,024 GB/s 的假设成立，变化趋势见 tR 扫描。
- 估算在 tR ≠ 4,096、联动组小于 die、协议拉长轮数时没有 MQSim 对照。
- decode 追加写及其对读的干扰、寿命，见阶段 B。

## 第 2 阶段 B（一）：多请求放置、写入干扰与寿命估算

这一阶段回答第 3 项（KV 直接存放在 HBF 上的寿命）和第 4 项（压缩带来的寿命收益），以及第 5 项中写路径上的冲突。寿命的结果来自稳态解析模型。逐事件的模拟（排队、保持期刷新、空间回收、前缀共享）放在阶段 B（二），用来核对这里的解析结果。

新增三块能力：
- **多请求放置（M2，`hbf/batch.py`）**：各请求上下文不同、按到达逐个写入时的地址分配，以及写顺序检查（`--alloc`、`--partition`、`--check-writes`）。
- **写入干扰（M5，`hbf/interference.py`）**：写入、刷新和擦除让每步变长多少。
- **寿命解析模型（M4a，`hbf/endurance.py --analytic`）**：用阶段 A 得到的每步时间推算写入速率、读干扰刷新和 P/E 消耗。负载预设见 `hbf/configs/workloads.json`。

分工：M2 由主代理实现，审查子代理审查后修正（见本节末尾）。M4、M5 由子代理实现，手算用例的测试通过，**但还没有独立审查**（PLAN.md 中的 T-B6），本节数字应视为初步结果。所有实验在本机运行（提交 `fb9631d`）。全量测试 350 项通过、3 项 xfail，与已知偏差一致（[测试日志](evidence/pytest/local-20261127T165020.log)）。数据见 [evidence/phase2b/](evidence/phase2b/)，运行方法见 [docs/local-runs.md](docs/local-runs.md)。

### 方法与设定

**放置方式**（单 stack，`ocp`，8B，条带布局，B ∈ {8, 32}，上下文 ∈ {4K, 32K}，生成 ∈ {0, 338, 2,048} token，共 96 个点）
- `layer_major`：阶段 A 的近似，按层为每个请求预留固定区域（静态批）。
- `dedicated`：每个请求独占一段连续的 superblock。
- `packed`：所有请求共用一个全局写前沿，按到达顺序追加写。
- `vllm_pool`：vLLM 式块池，16 token 一块；在 M2 静态快照里空闲块按递增编号分配（`fresh`，未建 LRU 复用），`shuffled` 表示空闲块编号被打乱，模拟长时间运行后的碎片。
- 分区：`channel` 把请求限制在单个主机通道内，`die` 限制在一个 die 内。
- decode 生成的 token 先在 HBM 缓冲，攒满一页再写入。`flush_order rotate` 把各层的新页轮流分到不同 bank；`layer` 按层顺序写出。

**写顺序检查**（`--check-writes`，规范 §5.4.1）：块内必须从第 0 页起按页顺序写，写第 0 页会自动擦除整块。分别统计跳页、回写（写到前面的地址）、覆盖已写页，以及擦除仍有有效数据的块。

**读并行度**：κ = 实际读轮数 ÷ 最少读轮数。最少读轮数是同样的页数均匀分到 1,024 个 bank 时的轮数，κ = 1 表示没有因放置而损失并行度。

**写入干扰**（4 个基点 × 4 种写入情景 × 4 个 tPROG × 3 个 tBERS × 5 种模式，共 960 个组合）
- 基点：8B 的 B=8 和 B=32（32K，BF16），8B 的 B=8 kivi2，70B 的 B=8。
- 写入情景：只有 decode 写入；decode 加读干扰刷新（X = 1e6 或 1e5 时，按稠密读带宽 × 每块页数 / X 计算）；decode 加 prefill 写入 2.66 GB/s（8B、B=8、32K 请求持续到达时的写入速率）。
- 模式：
  - `fifo`：写入排在读前面；
  - `gaps`：写入只占用 bank 在两次读之间的空闲时间；
  - `suspend`：读到来时暂停编程或擦除，暂停开销 5 µs 或 20 µs；
  - `random`：擦除随机落在各 die 上，与读的节奏不协调。
- tPROG 取 75、100、200、350 µs，tBERS 取 1、3.5、10 ms（来源见 [docs/research/params.md](docs/research/params.md)）。

**寿命解析模型**
- 闭环，并发 8 个请求（agentic 因会话间的思考时间，平均批约 6–7）。每步时间按阶段 A 的分析结果拟合，随批大小和上下文变化。
- 负载预设：
  - `chat`：输入中位数 1,020、输出 129 token（Azure 对话轨迹）；
  - `long32k`：32K 输入、338 输出；
  - `agentic`：每会话 64 次调用，初始历史 32K token，每次新增 875 输入、214 输出，调用间隔 5 s；
  - `reasoning`：输入同 chat，输出 2,048 token。
- 写入：每个请求的 prompt KV 和生成的 KV 都写进 HBF，没有跨请求前缀缓存（`agentic` 在单会话的 64 次调用间复用会话历史）。
- 读干扰：一个块被读 X 次后整块重写一次。稠密读时每步每块被读"每块页数"次，所以阈值折算成步数是 **X / 每块页数**，与每步时间和压缩格式都无关（ppb 256 时为 390.6、3,906、39,063 步）。
- 保持期 24 h，单 stack 512 GiB，`packed` 放置，理想磨损均衡（报告平均 P/E）。每次写第 0 页都计一次擦除，包括从未写过的块（规范字面理解，偏保守）。
- 扫描的参数：读干扰阈值 X ∈ {1e5, 1e6, 1e7}，P/E 上限 ∈ {1e4, 1e5, 1e6}，每块页数 ∈ {192, 256, 512, 1,024}。这些参数规范都写"由产品自定"，结论以曲线形式给出。

### 第 5 项（写路径）：放置方式与 §5.4.1 的冲突

写顺序合法性（[evidence/phase2b/placement.csv](evidence/phase2b/placement.csv)）：

| 放置方式 | 合法的点 | 违规原因 |
|---|---:|---|
| `packed`（含 `layer` 写出顺序、`die` 分区） | 36 / 36 | — |
| `dedicated`（含 `channel` 分区） | 24 / 24 | — |
| `layer_major` | 0 / 12 | 请求逐个写 prefill 时，各请求的同一层交错排在一起，后到的请求写回前面的地址，并擦除仍有数据的块；没有生成 token 时也违规 |
| `vllm_pool` | 2 / 12 | 只有 32K、不生成、空闲块全新时合法；4K 时不生成也有跳页和回写。追加生成 token 后，新 token 写进不连续的块，每个点都违规 |
| `vllm_pool shuffled` | 0 / 12 | 空闲块乱序复用，每个点都有数十万到数千万次违规 |

读并行度 κ（生成 0 / 338 / 2,048 token 中的最大值）：

| 放置方式 | B=8，4K | B=8，32K | B=32，4K | B=32，32K |
|---|---:|---:|---:|---:|
| `packed` | 1.03 | 1.00 | 1.04 | 1.00 |
| `dedicated` | 1.33 | 1.06 | 1.33 | 1.06 |
| `packed`，按层顺序写出 | 11.33 | 2.82 | 11.33 | 2.82 |
| `dedicated`，`channel` 分区 | 2.00 | 2.00 | 1.01 | 1.00 |
| `packed`，`die` 分区 | 8.00 | 8.00 | 2.00 | 2.00 |
| `layer_major`、`vllm_pool` | ≤ 1.03 | 1.00 | ≤ 1.01 | 1.00 |

1. **只有"全局追加写"和"每请求独占整段"两类放置合法。**vLLM 的块池复用和按层预留的静态批都与"块内顺序写、写第 0 页即擦除"冲突。规范又没有 GC（§11.4），违规无法由设备在内部吸收，主机侧的 KV 管理器要按"段"分配、整段回收。
2. **违规的两种放置读并行度并不差（κ ≈ 1）**，冲突完全在写路径上，阶段 A 只看读时看不出来。
3. **生成 token 的写出顺序必须轮换。**按层顺序写出时，同一层的新页都落在 32 个 bank 上，κ 最高 11.3。轮换后 `packed` 的 κ 不超过 1.04。
4. **分区会损失并行度。**B=8 时 `channel` 分区只用到一半通道（κ = 2），`die` 分区 κ = 8；B=32 时请求足够多，κ 回到 1 和 2。分区可以隔离写入和擦除对其他请求的干扰，代价是请求少时读变慢。
5. **短请求的起点对齐（T-B7）。**`dedicated` 和 `packed` 原来让每个请求从 bank 行的边界开始。上下文短时，各请求的同一层落在同一批 bank 上：kivi2、2K、B=8 时 κ = 5.33。现在默认把各请求的起点错开（`--align stagger`），同一点降到 1.67，写顺序仍合法；`--align row` 保留原行为。上表在这次修改前运行，`dedicated` 的 1.33 等数字需要重跑确认。

### 写入对读的干扰

每格为 TPOT 增加的百分比，tPROG = 100 µs，tBERS = 1 ms（最后一列为 10 ms），[evidence/phase2b/interference.csv](evidence/phase2b/interference.csv)：

| 基点 | 写入情景 | fifo | gaps | suspend 5 µs | random | random，tBERS 10 ms |
|---|---|---:|---:|---:|---:|---:|
| 8B，B=8 | 只有 decode | 0.07 | 0.00 | 0.01 | 0.24 | 6.4 |
| 8B，B=8 | 刷新，X = 1e5 | 6.06 | 1.79 | 0.40 | 11.5 | 384 |
| 8B，B=8 | prefill 2.66 GB/s | 7.13 | 2.11 | 1.06 | 12.6 | 422 |
| 8B，B=32 | prefill 2.66 GB/s | 7.13 | 5.42 | 5.54 | 3.3 | 111 |
| 8B，B=8，kivi2 | prefill 2.66 GB/s | 7.11 | 1.58 | 0.48 | 52.2 | 1,745 |
| 70B，B=8 | prefill 2.66 GB/s | 7.12 | 0.10 | 0.12 | 10.8 | 362 |

1. **decode 写入单独看可以忽略。**在 HBM 攒满整行再写，各种协调的模式下 TPOT 增加不超过 0.1%。
2. **prefill 级别的写入（或 X = 1e5 时的刷新）有可见代价。**
   - `fifo` 增加 5–10%（tPROG 350 µs 时最多 33%）。
   - `gaps` 的代价取决于读之间有多少空闲：tPROG = 100 µs（tBERS 1–10 ms）下，70B 计算时间长，约 0.1–2.4%；8B B=8 约 2–5%；8B B=32 的 bank 几乎一直在读，5–8%；全参数扫描到 tPROG = 350 µs 时三者最高分别达 8.2%、25.3%、31.0%。
   - `suspend` 5 µs 与 `gaps` 相当或更好，B=32 时一样受限于空闲时间。
3. **不协调的擦除代价很大。**擦除随机落在各 die 上时，tBERS 1 ms 下增加 3–52%，10 ms 下增加 100–1,750%。kivi2 每步最短，擦除阻塞所占比例最大。规范把刷新交给主机（§11.5），主机必须把擦除集中安排在读的空隙里，或者按 die 轮流进行。
4. 写入只占编程带宽的 5–6%（寿命模型的 `program_bw_share`），问题不在写带宽，而在写入和擦除阻塞了读。

### 第 3 项：KV 直接存放在 HBF 上的寿命

单 stack，512 GiB，并发 8，`packed`，每块 256 页，[evidence/phase2b/endurance-analytic.csv](evidence/phase2b/endurance-analytic.csv)：

| 模型 | 负载 | X | 写入 GB/s（prefill / decode / 刷新） | 刷新 ÷ KV 写入 | P/E 每天 | 到 P/E 1e4 | 到 1e5 | 到 1e6 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 8B | chat | 任意 | 1.97 / 0.25 / 0 | 0 | 353 | 28 天 | 284 天 | 7.8 年 |
| 8B | long32k | 任意 | 2.62 / 0.03 / 0 | 0 | 416 | 24 天 | 241 天 | 6.6 年 |
| 8B | reasoning | 1e5 | 0.14 / 0.20 / 1.71 | 5.0 | 322 | 31 天 | 310 天 | 8.5 年 |
| 8B | reasoning | ≥ 1e6 | 0.14 / 0.20 / 0 | 0 | 54 | 185 天 | 5.1 年 | 51 年 |
| 8B | agentic | 1e5 | 0.08 / 0.01 / 2.20 | 23.0 | 360 | 28 天 | 278 天 | 7.6 年 |
| 8B | agentic | 1e6 | 0.08 / 0.01 / 0.18 | 1.85 | 43 | 234 天 | 6.4 年 | 64 年 |
| 8B | agentic | 1e7 | 0.08 / 0.01 / 0 | 0 | 15 | 1.8 年 | 18 年 | 182 年 |
| 70B | chat | 任意 | 0.78 / 0.10 / 0 | 0 | 139 | 72 天 | 1.97 年 | 19.7 年 |
| 70B | long32k | 任意 | 2.25 / 0.02 / 0 | 0 | 357 | 28 天 | 280 天 | 7.7 年 |
| 70B | agentic | 1e6 | 0.08 / 0.01 / 0.17 | 1.85 | 42 | 239 天 | 6.5 年 | 65 年 |

1. **满负载时，写入速率由 prefill 决定，寿命按"容量 × P/E 上限 ÷ 写入速率"计算。**
   - 8B 的 long32k 每秒写入 2.6 GB，每天约 416 次 P/E。若 P/E 上限是 SLC 常见的 1e5，约 8 个月用完；若是 1e4，不到一个月。
   - 70B 的 chat 每步慢、写得少，寿命约是 8B 的 2.5 倍。70B 的 long32k 同样受 prefill 写入限制，与 8B 接近。
   - decode 写入很小：long32k 只占 1%，chat 占 11%。
2. **读干扰刷新只影响数据被反复读的负载。**chat（输出中位数 129、对数正态均值 177.6 token）和 long32k（输出 338 token）都短于阈值（ppb 256、X = 1e5 时为 390 步），自己的 KV 不需要刷新。reasoning（2,048 步）在 X = 1e5 时每个请求刷新 5 次。agentic 的历史在 64 次调用中反复被读，X = 1e5 时刷新写入是 KV 写入的 23 倍。
3. **"不需要刷新"的阈值（`break_even_x` = 数据一生被读的步数 × 每块页数）**：chat 4.5 万（按均值 177.6 步算），long32k 8.7 万，reasoning 52 万，agentic 234 万。产品的读干扰阈值低于这个值，就要为刷新付出写入和擦除。
4. **块越大，读干扰越早触发。**刷新阈值折算的步数与每块页数成反比。agentic、X = 1e6 时，每块 192、256、512、1,024 页对应每天 34、43、80、150 次 P/E，1,024 页是 192 页的 4.5 倍。以 prefill 写入为主的负载不受块大小影响（`packed` 没有填充）。
5. 这些都是理想磨损均衡下的平均值，最坏块和保持期刷新要靠事件模拟给出。

### 第 4 项：压缩后的寿命收益

8B，单 stack，并发 8，每块 256 页（[evidence/phase2b/endurance-formats.csv](evidence/phase2b/endurance-formats.csv)）：

| 格式 | HBF 每 token 字节 | long32k 吞吐 token/s | long32k P/E 每天 | 到 1e5 | 每百万 token 的 P/E | agentic X = 1e6 P/E 每天 | 到 1e5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 131,072 | 206 | 416 | 241 天 | 23.4 | 42.8 | 6.4 年 |
| fp8 | 65,536 | 387 | 391 | 256 天 | 11.7 | 39.4 | 7.0 年 |
| int4 | 34,816 | 579 | 310 | 322 天 | 6.2 | 26.9 | 10.2 年 |
| kivi4 | 40,960 | 535 | 336 | 298 天 | 7.3 | 30.9 | 8.9 年 |
| kivi2 | 24,576 | 766 | 288 | 347 天 | 4.4 | 21.9 | 12.5 年 |
| kivi2 + think 0.5 | 18,432 | 860 | 243 | 412 天 | 3.3 | 16.7 | 16.4 年 |

KIVI 的字节数只算 HBF 上的部分，残差窗口在 HBM。汇总表 `kv_bytes_per_token` 一列对 KIVI 误把残差也按每 token 计入（例如 kivi2 为 155,648），磨损数字不受影响，这一列在阶段 B（二）修正。

1. **按处理的 token 算，压缩的寿命收益等于字节比例。**每百万 token 的 P/E 从 23.4 降到 4.4（kivi2，5.4 倍）和 3.3（kivi2 + think，7.1 倍）。
2. **按天算，满负载时收益小得多：大部分收益变成了吞吐。**压缩后读得少，每步变快，同样并发下每秒处理的 token 更多（kivi2 为 3.7 倍），写入速率只降了 31%。long32k 到 1e5 的天数从 241 天到 347 天，只多 44%。若吞吐固定不变，寿命才按字节比例延长。
3. **压缩不改变读干扰阈值折算的步数。**压缩后一页仍是每步读一次，刷新 ÷ KV 写入在各格式下都相同（agentic 在 X = 1e6 时为 1.85）。压缩按比例减少了刷新的字节数，但没有减少刷新的需要。agentic 以刷新为主，kivi2 的寿命约为 bf16 的 2 倍（12.5 年对 6.4 年）。
4. 与第 2 项一致：稠密读时压缩同时带来带宽和寿命收益。按 16 token 块做稀疏选择时，压缩对读几乎没有收益（阶段 A），但写入量仍按字节比例减少。

### 设计讨论与审查记录

**设计**（实现前）
- 设计子代理写出方案（[docs/design/phase2b-design.md](docs/design/phase2b-design.md)），主代理答复其中的开放问题并分工（[docs/design/phase2b-decisions.md](docs/design/phase2b-decisions.md)）。
- 主要决定：
  - §5.4.1 的顺序写按 NAND 块理解；
  - 每次写第 0 页都计一次擦除；
  - 默认 `packed` 放置，满了先驱逐；
  - 写入干扰默认 `gaps`，按 die 阻塞；
  - 寿命的时钟用阶段 A 的结果拟合，并与阶段 A 逐点核对（0.5% 以内）。
- 阶段 B 分成两部分：
  - （一）放置、写入干扰、寿命解析模型；
  - （二）事件模拟。
  
  原实现子代理留下的事件模拟草稿保存在 `docs/design/drafts/`。

**M2 审查**（子代理）

| # | 问题 | 处理 |
|---|---|---|
| B-H1 | KIVI 的窗口张量在请求被接纳之前就写出了 | 新增 `fill_context`、`full_pages`，接纳后才写出 |
| B-H2 | 各请求 `--contexts` 相同时被忽略 | 改为设定存储的上下文 |
| B-M1 | 容量只按单个请求检查；页表用 int64 占内存 | 按全部请求检查；页表改为 int32 |
| B-M2 | `request_bytes` 用了统一上下文 | 按每个请求的上下文计算 |
| B-M3 | 写顺序检查把回写和覆盖混在一起 | 按擦除轮次区分回写、覆盖和擦除仍有数据的块 |
| B-M4 | 写顺序检查默认运行，大点很慢 | 只在 `--check-writes` 时运行 |
| B-L1、B-L2 | `layer_major` 的填充没有说明；vLLM 专用参数可以用在其他放置上 | 写入文档；其他放置用这些参数时报错 |

报告的措辞另有 7 处修正（R1–R7）。

### 局限

- 寿命是稳态解析结果。以下几项都要等阶段 B（二）的事件模拟：
  - 排队；
  - 保持期刷新的实际触发；
  - 空间满时的回收与驱逐；
  - 前缀共享；
  - 最坏块的 P/E。
- M4、M5 还没有独立审查。
- 读干扰阈值、P/E 上限、tPROG、tBERS、每块页数都是产品参数，规范没有给出，结论以扫描范围内的曲线成立。
- 写入干扰的写入量是设定的情景，用来给出影响范围；真实负载的写入量由寿命模型给出（prefill 级别约 2–2.6 GB/s，与情景一致）。
- 只有单个 stack，理想磨损均衡，没有前缀缓存。prefill 计算不经过 HBF，只计它的写入。
- 放置实验在 T-B7 之前运行，`dedicated` 和 `packed` 的 κ 需要重跑确认；写顺序合法性不受影响。
- 寿命模型中 `Device(banks=1024)` 与 `blocking_units(dies=64)` 写死了 16 主机通道（`ocp`），因此 `ocp-8ch`（8 主机通道、512 bank）暂时不能用于寿命模型。

## 规范一致性审查（OCP HBF v0.7.0）

第 1 阶段之后，由两个子代理先后对照 OCP HBF 规范 v0.7.0 审查并互相核实，关键数字另行复算。页码取自规范 PDF 页脚。结论：独立 plane 的结果都有效；页在 plane 内的偏移不符合规范，影响所有 lockstep 和条带布局的结论；另有两项配置能力缺失和若干文档措辞问题。

| # | 问题 | 规范依据 | 影响 | 处理 | 状态 |
|---|---|---|---|---|---|
| S1 | 页在 plane 内的偏移由 MQSim 按写入或首次读的顺序分配；规范是地址决定位置（无页级 FTL） | §4.6 p22–23 地址位序与图 4；§5.4.1 p58 块内连续写、禁止跳地址、写 page 0 自动擦除；§5.7 p62–63 由地址算 page-0；§11.4 p117 无 GC、只做块级重映射 | lockstep 轮数：4 层 × 32K、5% 选页、32 联动时，不加 prefill 43 轮，按规范 122 轮；独立 plane 不受影响 | 分析默认用静态偏移；prefill 按地址顺序写满；多 plane 运行要求页已写入 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S2 | 读未写过的页在规范中返回错误，MQSim 却分配新页并成功返回 | 表 12 p60 "Read from ErasePage"，错误码 0x7；§11.5.1.7 p120 | 同 S1 | 统计并拒绝（多 plane 时）读未写页 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S3 | 条带布局的填充地址没有写入，MQSim 把后续页的偏移压紧 | §5.4.1 第 4 条 "No skipping or jumping of 4KiB local address" | 条带布局 lockstep-64 每层按规范 18 轮，报告写成 32 轮；"条带化在 lockstep 下无效"的结论不成立 | prefill 连同填充地址一起写 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S4 | README 的 lockstep 示例没加 `--prefill` | 同 S1 | 示例结果低估约 2.8 倍 | 改示例 | 已改：分析默认静态位置；prefill 按地址顺序写满（含填充页）；多 plane 运行拒绝读未写页；条带结论已重测 |
| S5 | 规范示例层级没有现成配置：每主机通道 2 条 core die 通道 × 每通道 2 die × 每 die 16 bank，共 1,024 bank；地址先 bank 后 die | 图 4 p23（"for example"）；表 3 p16 只给 cube 汇总，信息不全 | 现有 T-PLANE 用 SPLASH 的 16 die × 64 plane，分组不同 | 新增规范示例拓扑和地址顺序 | 已改：新增 `--topology ocp`（32 通道 × 2 die × 16 bank，规范地址顺序） |
| S6 | 规范三档带宽 0.384/1.536/3.072 TB/s 无法配置：16 通道下每通道 24/96/192 GB/s 不是 2 的幂，配置检查拒绝 | 表 2、表 4 p16 | 无法做带宽档位扫描 | 放宽检查，传输时间与 MQSim 的整数算法一致 | 已改：放宽检查，`transfer_ns` 与 MQSim 的整数算法一致，新增 `--aggregate-gb-s`；三档的单页时延与 MQSim 逐 ns 相等 |
| S7 | 文档措辞：缓冲个数写成"要求"；"16x16 对应规范示例"；"16 die × 64 plane 和 HBF 一致"；"标称 1 TB/s"；512 GiB 是物理容量而规范表 3 是用户容量 | §5.3.1 第 7 条 p57；图 4；表 3、表 4 | 仅文档 | 改措辞并注明出处 | 已改：缓冲改为"支持"并引原文；16x16、16 die × 64 plane、1,024 GB/s 注明出处；容量区分物理与用户；64 B 突发注明只对主机链路成立 |
| S8 | 调度按 die 门控（T-PLANE、T-CACHE），规范按 bank 排序感测、每 bank 两个缓冲 | §5.3.1 第 5b–8 条 p57 | 多 plane 模式下时延偏保守，轮数不变 | 不改：chip 当 plane 模式即按 bank 独立调度；在文档中说明 | 不改，已写入 README"尚不支持"表 |
| S9 | 读计数、刷新、坏块、ECC、读重试、数据保持未建模 | §4.4 p19；§11.5 p118（刷新是主机责任）；§4.1；§5.3.1 第 9 条；§9 p106 | 无故障时不改变轮数和利用率 | 列为范围外 | 不改，已写入 README"尚不支持"表 |

## 局限与下一阶段接口

- MQSim 主结果沿用 SSD 的 chip 门控、固定 ONFI 命令和 PCIe/NVMe 协议。上游没有 cache read，感测和传输不能重叠；T-CACHE 为用户读补上了双缓冲，映射读和 GC 读仍走上游路径。T-CMD 的合并窗口对所有新事务统一生效，所以孤立批次和重叠批次都会多出 W 的等待。理想估算给出相应的吞吐下界模型，T-ONFI 用于独立研究四项命令时序。
- I2 只在 `NVM_PHY_ONFI_NVDDR2::Send_command_to_chip` 记录命令下发。同文件 `perform_interleaved_cmd_data_transfer` 发出的 die 交错命令没有 issue 记录，后续的 cmd_done、ready、data 事件仍有记录。目前所有配置每 chip 只有 1 个 die（多 plane 时 P>1，但 die 交错路径只在 D>1 时使用），不经过该路径。若以后用 D>1 的配置分析时间线，需要补记这一处。
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

**第 2 阶段 B（一），本机运行（提交 `fb9631d`）**：全量 pytest 共 **350 项通过，3 项 xfail**，[测试日志](evidence/pytest/local-20261127T165020.log)；3 项 xfail 不变。实验汇总表在 [evidence/phase2b/](evidence/phase2b/)，运行方法见 [docs/local-runs.md](docs/local-runs.md)。

**第 8 轮审查修正后（提交 `27ce1ca`）**：全量 pytest 共 **156 项通过，3 项 xfail**，用时 15 min 8 s，[测试日志](evidence/pytest/round8.log)（只保存了结束状态）；3 项 xfail 不变。新增 9 项：T-CACHE 空闲 chip 回归、条带布局补页与稠密均衡（5 组）、主机通道错开、核心通道下界、生成器与 CLI 边界。更早的三次全量运行（147、146、138 项）没有保存日志。

**实验排查复审修正后（提交 `9f69e3d`）**：全量 pytest 共 **147 项通过，3 项 xfail**，用时 14 min 40 s；3 项 xfail 不变。新增联动组 `group_sense` 测试，并扩充了贪心下界、HBF-EventLab 变长项和 `uniform_tokens` 边界的测试。

**实验排查修正后（提交 `e6397a6`）**：全量 pytest 共 **146 项通过，3 项 xfail**，用时 14 min 42 s；3 项 xfail 不变。新增 8 项：分段放行改为一次感测（含 HBF-EventLab MOCS 1 对照）、额度下界不超过轮数的随机测试（5 组）、`uniform_tokens`、生成模式越界报错、多 plane 不给估算与子批轮数列。

**协议约束修正后（提交 `c948424`）**：全量 pytest 共 **138 项通过，3 项 xfail**，用时 14 min 34 s；3 项 xfail 不变。比规范一致性修正时多 12 项，均为协议约束测试。

**规范一致性修正后（提交 `873b8f1`）**：全量 pytest 共 **126 项通过，3 项 xfail**，用时 14 min 27 s，[测试日志](evidence/pytest/spec-review.log)。3 项 xfail 不变。比第 1 阶段多 10 项：静态位置与 MQSim 物理位置核对（含条带布局）、多 plane 拒绝读未写页、`ocp` 地址映射与 MQSim 运行、三档带宽的单页时延。

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
