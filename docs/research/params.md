# Parameter ranges for the KV-cache-on-HBF study (phase 2)

> Parameter research for phase 2 (English, with sources and support tags). Treat numbers tagged from memory or guess as unverified; several web pages could not be opened directly when this was written.

Compiled 2026-11-27 for modules M0, M1, M3, M4 and M5 of `phase2/brief.md`.

## How to read this file

Each row gives a value, its source (title and URL), the year, what it applies to, and how well it is supported. Two columns carry most of that.

**Applies to** says what the number describes:

- **HBF**: HBF-specific, from the spec, a vendor or an HBF paper.
- **SLC**: native SLC or low-latency SLC (XL-Flash, Z-NAND).
- **pSLC**: SLC mode on a TLC/QLC die.
- **TLC**: native TLC.
- **SSD**: drive-level.
- **GPU**, **model** and **workload** rows describe those things.

**Tag** says how well the number is supported:

| Tag | Meaning |
|---|---|
| **[P]** | I read it in the primary document myself. Local copies: OCP spec text, the SPLASH and FlashAccel PDFs, MQSim source. |
| **[W]** | A web-search excerpt shows it in the primary or official source. I could not open the page myself: WebFetch could not resolve tomshardware.com, storagereview.com or arxiv.org from this session, so web numbers come from search excerpts only. |
| **[S]** | Secondary report: press, blog or a third-party table. |
| **[D]** | I derived it. The formula is shown. |
| **[M]** | From memory, not re-checked this session. Verify before citing in a paper. |
| **[G]** | Guess or assumption. No source. |

## 0. What the OCP HBF spec v0.7.0 itself fixes

Local text: `scratchpad/hbf-spec.txt`. Spec dated 03 AUG 2026.

| Item | Value in spec | Section | Tag |
|---|---|---|---|
| NAND page size | 4 KiB | §4.1 (NAND Interface Features), Table 3 | [P] |
| Write granularity / ordering | 4 KiB writes. Addresses consecutive within a NAND block, no skipping. Auto-erase when page 0 (WL-0, STR-0) of a block is written. | §5.4.1 items 3, 4, 5, 9 | [P] |
| NAND block size (pages per block) | "defined in the product specification" | §5.4.1 item 9 | [P] |
| Read granularity | 64 B to 4 KiB, must not cross a 4 KiB page. 2 cache buffers per bank. Strict sense order within a bank. | §4.1, §5.3.1 items 5–7 | [P] |
| Stack organisation | 16 host channels. 16-die stack (8-high for grade 1). Table 3: "16 dies/cube, 16 banks/channel, 4096 B page, 512 GiB". §4.6 / Fig. 4 example: 16 banks/die, 4 dies per host channel, so 1,024 banks per stack (what the repo models). | §4, §4.2 Table 3, §4.6 (Fig. 4), Table 4 | [P] |
| Host bandwidth grades | 0.384 / 1.536 / 3.072 TB/s per stack (UCIe x64 at 8 / 16 / 32 GT/s, 75 % AXI efficiency) | §4.2 Tables 2, 4 | [P] |
| Capacity | "512 GiB density or higher" per stack | §4, Table 3 | [P] |
| Positioning | "8x to 16x of the capacity of HBM, while delivering the same read bandwidth" | §3 (Introduction) | [P] |
| Power-on retention | 24 h at 85 °C. May not retain data after power-off. | §9 Table 33 | [P] |
| Refresh interval | "periodic data refresh … typically every 24–48 hours" | §11.5 | [P] |
| Endurance / max PEC | "Product Specific". Exposed in MMIO MAXPEC (0x144) and AVGPEC (0x148). | §9 Table 33 note 2, §5.9.2.17–18 | [P] |
| Read disturb | "Product Specific". Host refreshes a block after X reads. Base die keeps a per-block page-read counter. CECC/UECC carry "Read Count > Max_Read_Threshold". | Table 33 note 3, §4.4, §11.5, §11.5.1.4–5, §5.5 Table 12 | [P] |
| Lifespan / UBER / MTBF | 10 years or endurance indicator 100 %. UBER 1e-16. MTBF 20 M h, AFR 0.04 %. | §9 Table 33 | [P] |
| Garbage collection | None. Only zone remapping for wear levelling, at block-multiple granularity. | §11.4, §11.4.1 | [P] |
| tR, tPROG, tBERS, planes per die | Not given | — | [P] |

## 1. HBF and NAND timing and geometry

### 1a. HBF-specific claims and HBF papers

| Parameter | Value / range | Applies to | Source (title, URL) | Year | Tag / notes |
|---|---|---|---|---|---|
| Stack capacity | 512 GB/stack, Gen1. 16 × 256 Gb dies. | HBF | IEEE Spectrum, "High Bandwidth Flash", https://spectrum.ieee.org/high-bandwidth-flash ; StorageReview, "Sandisk tapes out its first HBF memory die, targets 2027…", https://www.storagereview.com/news/sandisk-tapes-out-its-first-hbf-memory-die-targets-2027-for-inference-product-samples | 2025–2026 | [S]. Cell type (SLC vs MLC/TLC) not disclosed. |
| Read bandwidth, SanDisk Gen1 | 1.6 TB/s per stack | HBF | same | 2025–2026 | [S] |
| Capacity vs HBM | 8–16× HBM capacity "at a similar cost point". 8 stacks = 4 TB per GPU. | HBF | TechPowerUp, "SanDisk Develops HBM Killer: High-Bandwidth Flash (HBF)…", https://techpowerup.com/332516/sandisk-develops-hbm-killer-high-bandwidth-flash-hbf-allows-4-tb-of-vram-for-ai-gpus ; Tom's Hardware, "SanDisk's new High Bandwidth Flash memory enables 4TB of VRAM…", https://www.tomshardware.com/pc-components/dram/sandisks-new-hbf-memory-enables-up-to-4tb-of-vram-on-gpus-matches-hbm-bandwidth-at-higher-capacity | 2025 | [S] |
| Read latency (vendor) | Not disclosed. "Latency remains significantly higher" than HBM. Aimed at read-heavy work. | HBF | same Tom's Hardware article; HPCwire, "What Is High Bandwidth Flash (HBF)…", https://www.hpcwire.com/2026/08/04/what-is-high-bandwidth-flash-hbf-and-can-it-overcome-the-ai-memory-wall/ | 2025–2026 | [S]. HBF-Sim and SlipStream also say vendors publish no tR. |
| OCP grades, capacity | 0.4–3.0 TB/s in 3 grades. Up to 512 GB. 8-high and 16-high. UCIe. | HBF | TrendForce, "SK hynix, Sandisk Debut HBF Standard…", https://www.trendforce.com/news/2026/08/04/sk-hynix-sandisk-debut-hbf-standard-to-challenge-ai-memory-bottlenecks-with-google-tenstorrent-support/ | 2026 | [S]. Matches the spec [P]. |
| Samples / volume | Samples 2027, mass production 2028 | HBF | TrendForce 2026-08-14, https://www.trendforce.com/news/2026/08/14/news-sandisk-reportedly-tapes-out-first-hbf-product-targets-2027-samples-and-2028-production/ | 2026 | [S] |
| SPLASH tR / tPROG / tBERS | 4.096 µs / 100 µs / 1 ms | HBF (modelled SLC) | Jonnalagadda et al., "SPLASH: Co-Designing Flash-Aware Sparse Attention with a Virtualized HBM–HBF Hierarchy", arXiv 2609.23816v3, https://arxiv.org/abs/2609.23816 (Table 1) | 2026 | [P]. Same as the repo defaults (`read_ns=4096`, `program_ns=100000`, `erase_ns=1000000`). |
| SPLASH geometry | 64 planes/die × 512 blocks/plane × 192 pages of 4 KB. 16 dies, so 1,024 planes and 384 GB per stack. 1 TB/s per stack. Die area 99.6 mm² (from the XL-Flash die shot). | HBF | same, Table 1 and §6.5 | 2026 | [P]. B = P·Sp/tR = 1024 × 4 KB / 4.096 µs ≈ 1 TB/s. A read round of 1,024 pages is 4 MB. |
| SPLASH program occupancy | "Each program occupies its plane for 100 µs". Programs scheduled after a layer's last read round. | HBF | same §5 | 2026 | [P] |
| SPLASH energy | Read 64 pJ/B, program 640 pJ/B. HBM 22.5 pJ/B. HBF idle 6.3 W/stack. | HBF | same Table 1 | 2026 | [P] |
| FlashAccel tR / tPROG | 4 µs / 75 µs | HBF (modelled SLC) | Wang et al., "FlashAccel: Leveraging High-Bandwidth Flash (HBF) for High-Throughput LLM Inference", arXiv 2607.10186v2, https://arxiv.org/abs/2607.10186 (Table III) | 2026 | [P]. Based on Kouchi ISSCC 2020. |
| FlashAccel geometry | SLC, 96 WL layers, 4 KB page, 192 pages/block, 342 blocks/plane, 96 planes/die (3 megaplanes of 32), 192 Gb/die. 8 dies + base die, 768 GB/s, 192 GB/stack. | HBF | same Table III | 2026 | [P] |
| FlashAccel write bandwidth | Read 4.6 TB/s vs peak write 245.8 GB/s for its CSI config | HBF | same §VI-D | 2026 | [P]. Write/read ≈ 1/19. |
| SlipStream latency | 20 µs request-to-return unloaded latency (default). Swept 20/10/5 µs. tR variance 3–5 µs. 4 KB page. | HBF | Saxena et al., "SlipStream: Enabling HBF for GPU Systems by Mitigating the Latency and SRAM Overheads", arXiv 2611.14776, https://arxiv.org/abs/2611.14776 | 2026 | [W]. End-to-end latency including transport, not tR. |
| SK hynix H³ | HBF holds weights and read-only shared KV. Generated KV stays in HBM. Up to 2.69× perf/W. "~100,000 write cycles" (secondary). | HBF | TrendForce 2026-02-12, https://www.trendforce.com/news/2026/02/12/news-sk-hynix-unveils-ai-chip-architecture-with-hbf-reportedly-boosts-performance-per-watt-by-up-to-2-69x/ ; all-about-industries.com summary, https://www.all-about-industries.com/sk-hynix-presents-hybrid-storage-architecture-for-better-ki-inference-a-73bfd0eb323c3a35bd0391c0571a7c59/ ; Ha et al., IEEE CAL 2026 (paywalled) | 2026 | [S]. The 100K figure is not confirmed in the paper. |
| Implied tR from bandwidth and bank count | tR = N_banks × 4 KiB / BW. With 1,024 banks: 10.9 µs (0.384 TB/s), 2.73 µs (1.536), 2.62 µs (SanDisk 1.6), 1.37 µs (3.072). With 2,048 banks the times double. | HBF | derived from spec Table 4 and SanDisk 1.6 TB/s | — | [D]. Grade 3 at 1,024 banks needs tR ≈ 1.4 µs, faster than any published SLC. Either the bank count is larger or the grade is link-limited, not array-limited. |

### 1b. Low-latency SLC, SLC and TLC reference points

| Parameter | Value / range | Applies to | Source (title, URL) | Year | Tag / notes |
|---|---|---|---|---|---|
| tR / tPROG | 4 µs / 75 µs. 128 Gb 1 b/cell, 96 WL layers. | SLC (Kioxia) | Kouchi et al., "A 128Gb 1b/Cell 96-Word-Line-Layer 3D Flash Memory to Improve Random Read Latency with tPROG=75µs and tR=4µs", ISSCC 2020 13.5 (also JSSC Oct 2020), https://dblp.org/rec/conf/isscc/KouchiKYBKIDSIA20 | 2020 | [W]. The base for FlashAccel and SPLASH. |
| tR | < 5 µs (TLC about 50 µs) | SLC (XL-FLASH) | Kioxia XL-FLASH page, https://americas.kioxia.com/en-ca/business/memory/xlflash.html ; press release 2019-08-05, https://americas.kioxia.com/en-ca/business/news/2019/memory-20190805-1.html | 2019 | [W] |
| Page / planes / die | 4 KB page, 16 planes, 128 Gb SLC die (256 Gb MLC variant) | SLC (XL-FLASH) | same; AnandTech, "Toshiba Launches XL-FLASH 3D SLC NAND", https://www.anandtech.com/show/14707 | 2019 | [W] |
| tPROG | "7 µs" | SLC (XL-FLASH) | Tom's Hardware, "Kioxia's New XL-NAND SSDs to Compete with Intel's Optane", https://www.tomshardware.com/news/kioxia-intros-ssd-optane-competitor | 2020 | [S]. Implausibly low for SLC tPROG and not in any Kioxia document. Do not use. |
| 2nd-gen XL-Flash latency | < 13 µs read, 4 µs write (drive-level) | SLC SSD | Tom's Hardware, "Custom PCIe 5.0 SSD with 3D XL-Flash debuts…", https://www.tomshardware.com/pc-components/ssds/custom-pcie-5-0-ssd-with-3d-xl-flash-debuts-special-optane-like-flash-memory-delivers-up-to-3-5-million-random-iops | 2025 | [S]. Drive latency, not array timing. |
| tR / tPROG | 3 µs / 100 µs. 64 Gb die. 2 KB or 4 KB page. | SLC (Samsung Z-NAND) | AnandTech, "The Samsung 983 ZET (Z-NAND) SSD Review", https://www.anandtech.com/show/13951/the-samsung-983-zet-znand-ssd-review/2 ; Tom's Hardware, "Samsung Announces QLC SSDs And Second-Gen Z-NAND", https://www.tomshardware.com/news/samsung-qlc-z-nand-ssd-flash,37945.html | 2018–2019 | [W] |
| tR in SLC mode | 34 µs typ. SLC block 7 MB (TLC 21 MB). 1,344 pages/block (TLC). Block erase 10 ms typ. | pSLC / TLC (SanDisk 3D Gen5 X3) | SanDisk S5E datasheet (335S00620-A), https://dosdude1.com/files/Mac-NAND/datasheets/335S00620-A_SanDisk.pdf | ~2019 | [W]. 7 MB / 16 KB is about 448 SLC pages per block [D]. |
| Block erase | 15 ms typ. 2,304 pages/block. | TLC (4 Tb) | Frontgrade UT81NDQ512G8T datasheet, https://frontgrade.com/sites/default/files/documents/Datasheet-UT81NDQ512G8T.pdf | ~2023 | [W] |
| Block erase | 2 ms typ (Micron MT29F2G08ABAGAH4). 3.5 ms typ (SkyHigh S34ML). | SLC (planar) | Micron datasheet, https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/791/MT29F2G08ABAGAH4.pdf ; SkyHigh S34ML0xGx, https://www.mouser.in/en/new/skyhigh-memory/spansion-s34ml0x/ | 2010s | [W] |
| Erase behaviour | Base erase ≥ 3.5 ms. 2–5 ISPE loops after 2K P/E. 40 µs read / 350 µs program / 3.5 ms erase in the evaluated TLC config. | TLC (3D, 160 chips) | Cho et al., "AERO: Adaptive Erase Operation for Improving Lifetime and Performance of Modern NAND Flash-Based SSDs", ASPLOS 2024, https://arxiv.org/abs/2404.10355 | 2024 | [W] |
| TLC timing used in MQSim studies | 90 µs read / 700 µs program / 5 ms erase | TLC | "Reducing Solid-State Drive Read Latency by Optimizing Read-Retry", https://arxiv.org/abs/2104.09611 | 2021 | [W] |
| Upstream MQSim defaults | 75 µs read (LSB) / 750 µs program / 3.8 ms erase. 8 KiB page, 256 pages/block. | TLC-ish SSD | `hbf/mqsim/src/exec/Flash_Parameter_Set.cpp` lines 8–22 (CMU-SAFARI MQSim), https://github.com/CMU-SAFARI/MQSim | 2018 | [P]. Shows the SSD-oriented defaults the repo overrides. |
| Pages per block (structural) | Pages per SLC block ≈ WL layers × strings. For example 96 × 2 = 192 (FlashAccel, SPLASH). 200+ layer parts give several hundred. | SLC / pSLC | derived from FlashAccel (96 WL → 192 pages), the S5E SLC block size, and spec §5.4.1 "WL-0, STR-0" | — | [D]. The spec confirms a WL × string page order but not the counts. |
| Planes per die | 16 (XL-Flash). 64 (SPLASH model). 96 (FlashAccel model). 4–6 (commodity TLC, e.g. 6-plane 232L-class parts). | SLC / HBF / TLC | rows above; commodity plane count [M] | — | HBF needs many small planes; the repo uses 16 banks per die (spec Fig. 4). |

## 2. Endurance and reliability

| Parameter | Value / range | Applies to | Source (title, URL) | Year | Tag / notes |
|---|---|---|---|---|---|
| HBF endurance | "Product specific". Host monitors AVGPEC against MAXPEC. | HBF | OCP spec §9 Table 33 | 2026 | [P] |
| HBF retention | 24 h power-on at 85 °C. Refresh every 24–48 h. | HBF | OCP spec §9, §11.5 | 2026 | [P] |
| HBF read disturb | Refresh after X (product-specific) reads per block. Base die flags "Read Count > Max_Read_Threshold". | HBF | OCP spec §11.5, §11.5.1.5 | 2026 | [P] |
| SLC P/E assumed for HBF | 1e5 | HBF (SLC mode) | SPLASH §6.6 (cites Cai et al. 2017); FlashAccel §II (cites Mohan et al., HotStorage 2010, "How I Learned to Stop Worrying and Love Flash Endurance") | 2026 | [P] |
| P/E with relaxed retention | FlashAccel assumes 10× (1e5 to 1e6). The literature claims up to 50× when retention drops from 3 years to 3 days. | HBF / SLC | FlashAccel §VI-D [P]; SPLASH §6.6 [P]; both cite Luo et al., "WARM", MSST 2015, https://ghose.cs.illinois.edu/papers/15msst_warm.pdf and Cai et al., "Flash Correct-and-Refresh", ICCD 2012, https://doi.org/10.1109/ICCD.2012.6378623 | 2012–2026 | [P] as cited. The 50× figure is truncated in the WARM excerpt I saw [W]. |
| Refresh-based lifetime gains | FCR: 31× (hybrid) and 46× (adaptive-rate). WARM: 3.24×, or 12.9× with adaptive refresh. | MLC (SSD) | FCR, ICCD 2012 (KiltHub copy: https://kilthub.cmu.edu/articles/journal_contribution/Flash_Correct-and-Refresh_Retention-Aware_Error_Management_for_Increased_Flash_Memory_Lifetime/6468821); WARM, MSST 2015 | 2012, 2015 | [W]. Measured on MLC SSD workloads, not SLC KV. |
| SPLASH write budget | 1.66 GB/s per GPU (3 TB, 1e5 P/E, 5 years, WA 1.17). Mean residency 30.8 min. Write amplification 1.17 overall: 1.85 at 10K-token contexts, 1.02 at 512K–1M. Daily refresh costs < 2 % of the budget over 5 years. | HBF | SPLASH §6.6 | 2026 | [P]. Formula L = C·N_PE / (WA·R_W). |
| Measured KV write rates | Back-to-back prefill: 0.82 GB/s per GPU (SPLASH). FlashAccel: 0.99 GB/s per GPU including prefill. | HBF | SPLASH §6.6 | 2026 | [P] |
| SLC P/E (planar) | 150K (older SLC SSDs). MLC at 1x-nm: 3K. | SLC / MLC | Cai et al., "Error Characterization, Mitigation, and Recovery in Flash-Memory-Based Solid-State Drives", Proc. IEEE 2017, https://arxiv.org/abs/1706.08642 | 2017 | [W] |
| SLC P/E (generic) | 100K. MLC 10K. eMLC 30K. | SLC | Pure Storage blog (2013), https://blog.purestorage.com/?p=190 | 2013 | [S] |
| pSLC P/E | 30K (SP Industrial, 3D NAND). 40K, from 10K native TLC (MEMXPRO PC32, Micron 64L). > 50K (Beckhoff). 100K, configurable to 150K (ATP N751Pi). | pSLC | Silicon Power, https://silicon-power.com/web/ro/technology_detail-pslc ; AnandTech, "What's TLC is SLC Again: MEMXPRO…", https://www.anandtech.com/show/15530 ; Beckhoff, https://www.beckhoff.com/en-gb/products/ipc/industrial-ssds-flash-memory/ ; StorageNewsletter, ATP N751Pi, https://storagenewsletter.com/2024/08/30/atp-industrial-n751pi-pcie-gen4-nvme-m-2-2280-ssds-with-125c-operating-temperature-range | 2020–2024 | [S]. Vendor marketing; test conditions differ. |
| TLC P/E | 2.2K (Beckhoff 3D TLC, at 1-year retention). 3K (typical). 5K, later 11K in test (ATP industrial TLC). 10K (Micron 64L industrial). | TLC | Beckhoff (above); TechPowerUp, "ATP Extends Endurance of its Industrial 3D TLC SSDs", https://www.techpowerup.com/338285/atp-extends-endurance-of-its-industrial-3d-tlc-ssds ; AnandTech 15530 | 2020–2025 | [S] |
| XL-Flash drive endurance | Kioxia FL6: 60 DWPD | SLC SSD | Tom's Hardware, Kioxia FL6 article (above) | 2020 | [S]. Over 5 years at WA ≈ 1 this implies ≥ about 1.1e5 P/E [D]. |
| Read disturb (SLC, classic) | Errors expected after about 1M reads per block. First-gen MLC about 100K. Modern MLC as few as 20K. | SLC / MLC | Cai et al., "Read Disturb Errors in MLC NAND Flash Memory: Characterization, Mitigation, and Recovery", DSN 2015, https://arxiv.org/abs/1805.03283 | 2015 | [W] |
| Read disturb (3D TLC) | Worst case 54,560 reads (one page next to the weakest WL read repeatedly). 518,420 reads under a spread pattern. Worst WL 403K. | TLC (3D) | "STRAW: A Stress-Aware WL-Based Read Reclaim Technique for High-Density NAND Flash-Based SSDs", https://arxiv.org/abs/2501.02517 | 2025 | [W]. Read disturb is per WL; the threshold depends on the access pattern. |
| Read disturb (3D SLC mode / HBF) | No published number | SLC / HBF | — | — | [G]. Fewer bits per cell are generally more disturb-tolerant (patent statements), so ≥ 1e5 is likely. Sweep it. |
| Retention requirement (SSD baseline) | Client: 1 year at 30 °C. Enterprise: 3 months at 40 °C (power-off). | SSD | JEDEC JESD218 | 2010 | [M]. Compare with HBF's 24 h power-on. This gap is the basis for the "relaxed retention, more P/E" argument. |

## 3. GPUs for the compute model (M1)

Dense means without 2:4 sparsity. NVIDIA headline numbers are usually sparse, which is 2× dense.

| GPU | Dense BF16 TFLOPS | Dense FP8 TFLOPS | HBM | HBM BW | Source (title, URL) | Year | Tag |
|---|---|---|---|---|---|---|---|
| H100 SXM | 989 | 1,979 | 80 GB HBM3 | 3.35 TB/s | NVIDIA H100 page/datasheet, https://www.nvidia.com/en-us/data-center/h100/ (sparse 1,979 / 3,958; "specifications 1/2 lower without sparsity") | 2022 | [W] |
| H200 SXM | 989 | 1,979 | 141 GB HBM3e | 4.8 TB/s | NVIDIA H200 page, https://www.nvidia.com/h200 (1,979 / 3,958 "with sparsity") | 2024 | [W] |
| B200 (HGX/DGX) | 2,250 | 4,500 | 180 GB HBM3e | 8 TB/s (DGX: 64 TB/s over 8 GPUs; some tables say 7.7) | NVIDIA DGX B200, https://www.nvidia.com/en-us/data-center/dgx-b200/ ; Lenovo Press LP2226, https://lenovopress.lenovo.com/LP2226 | 2024–2025 | [W] |
| GB200 (per GPU, NVL72) | 2,500 | 5,000 | 186 GB HBM3e (13.4 TB / 72) | 8 TB/s (576 TB/s / 72) | NVIDIA GB200 NVL72, https://www.nvidia.com/en-us/data-center/gb200-nvl72/ ; DGX GB200, https://www.nvidia.com/en-us/data-center/dgx-gb200/ | 2024 | [W]. Rack figures 720 / 360 PF are sparse, giving 5 / 2.5 PF dense per GPU [D]. |
| B300 / GB300 | about 2,250–2,500 | about 4,500–5,000 | 288 GB HBM3e (8 × 12-high) | 8 TB/s | NVIDIA, "Inside NVIDIA Blackwell Ultra…", https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era ; The Register, https://www.theregister.com/2025/03/18/nvidia_blackwell_ultra/ ; GB300 NVL72, https://www.nvidia.com/en-in/data-center/gb300-nvl72/ | 2025 | [W]/[S]. Dense FP4 15 PF is confirmed. The BF16/FP8 dense split varies by source (HGX 4.5 vs GB300 5 PF FP8). |
| Achievable fraction | FlashAttention-3 on H100: 740 TFLOPS (75 %), later 840 (85 %) BF16; FP8 1.2–1.3 PFLOPS | — | — | — | Shah et al., "FlashAttention-3", https://arxiv.org/abs/2407.08608 | 2024 | [W]. Attention-kernel peak, not end-to-end MFU. |

## 4. Model configurations (BF16 unless stated)

KV/token = 2 × layers × kv_heads × head_dim × 2 B. One (layer, KV head, K or V) row of head_dim 128 in BF16 is 256 B, so a 4 KiB HBF page holds 16 tokens (as in SPLASH §4.1).

| Model | Layers | Hidden | FFN inter. | Q heads | KV heads | head_dim | Vocab | Params / BF16 bytes | KV per token | KV at 128K | Source | Tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 4096 | 14336 | 32 | 8 | 128 | 128256 | 8.03 B / 16.1 GB | 128 KiB | 16 GiB | HF `meta-llama/Llama-3.1-8B` config.json (mirrors), https://huggingface.co/meta-llama/Llama-3.1-8B | [W] |
| Llama-3.1-70B | 80 | 8192 | 28672 | 64 | 8 | 128 | 128256 | 70.6 B / 141 GB | 320 KiB | 40 GiB | HF `meta-llama/Llama-3.1-70B` config.json (mirrors), https://huggingface.co/meta-llama/Llama-3.1-70B | [W] |
| Qwen2.5-7B | 28 | 3584 | 18944 | 28 | 4 | 128 | 152064 | 7.6 B / 15.2 GB | 56 KiB | 7 GiB | Qwen2.5 Technical Report Table 1, https://arxiv.org/abs/2412.15115 ; config mirrors | [W] |
| Qwen2.5-32B | 64 | 5120 | 27648 | 40 | 8 | 128 | 152064 | 32.5 B / 65 GB | 256 KiB | 32 GiB | Qwen2.5 TR Table 1 (layers, heads) [W]; hidden/FFN [M] | [W]/[M] |
| Qwen2.5-72B | 80 | 8192 | 29568 | 64 | 8 | 128 | 152064 | 72.7 B / 145 GB | 320 KiB | 40 GiB | Qwen2.5 TR Table 1; config mirrors | [W] |
| Qwen3-8B | 36 | 4096 | 12288 | 32 | 8 | 128 | 151936 | 8.2 B / 16.4 GB | 144 KiB | 18 GiB | HF `Qwen/Qwen3-8B` config.json, https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json ; Qwen3 TR, https://arxiv.org/abs/2505.09388 | [W] (KV=8 [M]) |
| Qwen3-32B | 64 | 5120 | 25600 | 64 | 8 | 128 | 151936 | 32.8 B / 65.6 GB | 256 KiB | 32 GiB | Qwen3-architecture config mirror (OpenBuddy SimpleChat-32B), https://huggingface.co/OpenBuddy/SimpleChat-32B-V1/blob/main/config.json | [S]. Check the official repo. |
| DeepSeek-V2 (MLA, MoE) | 60 | 5120 | MoE: 160 routed + 2 shared, 6 active | 128 | latent | kv_lora_rank 512 + RoPE key 64 = 576 per token per layer | 102400 | 236 B total / 21 B active | 67.5 KiB | 8.4 GiB | DeepSeek-V2 paper, https://arxiv.org/abs/2405.04434 (236B/21B, 93.3 % KV reduction [W]); layer counts [M] | [W]/[M] |
| DeepSeek-V2-Lite (MLA, MoE) | 27 | 2048 | dense 10944 (layer 0); MoE: 64 routed + 2 shared, 6 active, expert inter. 1408 | 16 | 1 (latent) | kv_lora_rank 512 + RoPE 64 = 576; qk_nope 128, v 128, q_lora null | 102400 | 15.7 B total / 2.4 B active (recomputed 15.71 B / 2.45 B incl. LM head) | 30.4 KiB (31,104 B) | 3.8 GiB | HF `deepseek-ai/DeepSeek-V2-Lite` config.json, https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json | [W] |
| DeepSeek-V3 (MLA, MoE) | 61 | 7168 | dense 18432 (first 3 layers); MoE 256 routed + 1 shared, 8 active, expert inter. 2048 | 128 | latent | 512 + 64 = 576; q_lora_rank 1536 | 129280 | 671 B total / 37 B active; native FP8 weights about 671 GB | 68.6 KiB (70,272 B) | 8.6 GiB | HF `deepseek-ai/DeepSeek-V3` config.json, https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json ; Raschka KV calculation, https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/ ; "Insights into DeepSeek-V3", https://arxiv.org/abs/2505.09343 (70 KB/token vs Qwen2.5-72B 327 KB, Llama-3.1-405B 516 KB) | [W] (MoE details [M]) |
| Gemma-3-27B (hybrid 5:1 `LLLLLG`, W = 1,024) | 62 (52 L + 10 G) | 5376 | 21504 | 32 | 16 | 128 | 262208 (tied) | 27.02 B text (25.60 B non-emb + 1.42 B emb; +417 M vision = 27.4 B multimodal) | 80 KiB (81,920 B) global + 416 MiB fixed local | 10.4 GiB (10 GiB global + 0.41 GiB local) | HF `google/gemma-3-27b-it` config.json (`text_config`), https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json ; Gemma 3 Technical Report Table 1, https://arxiv.org/abs/2503.19786 | [W] |
| GPT-OSS-20B (hybrid 1:1 `LG`, W = 128, MoE) | 24 (12 L + 12 G) | 2880 | MoE: 32 routed, 4 active, inter. 2880 | 64 | 8 | 64 | 201088 | 20.91 B total / 3.61 B active (native MXFP4 MoE + BF16 rest, 12.8 GiB ckpt) | 24 KiB (24,576 B) global + 3 MiB fixed local | 3.0 GiB | HF `openai/gpt-oss-20b` config.json, https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json ; OpenAI gpt-oss Model Card Table 1, https://arxiv.org/abs/2508.10925 | [W] |
| GPT-OSS-120B (hybrid 1:1 `LG`, W = 128, MoE) | 36 (18 L + 18 G) | 2880 | MoE: 128 routed, 4 active, inter. 2880 | 64 | 8 | 64 | 201088 | 116.83 B total / 5.13 B active (native MXFP4 MoE + BF16 rest, 60.8 GiB ckpt) | 36 KiB (36,864 B) global + 4.5 MiB fixed local | 4.5 GiB | HF `openai/gpt-oss-120b` config.json, https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json ; OpenAI gpt-oss Model Card Table 1, https://arxiv.org/abs/2508.10925 | [W] |
| Qwen3-Next-80B-A3B (SSM 3:1 `SSSA`, MoE) | 48 (36 S + 12 A) | 2048 | MoE: 512 routed + 1 shared, 10 active, inter. 512 | 16 (gated Q) | 2 (full attn) | 256 (full); Gated DeltaNet 16 K / 32 V heads × 128 | 151936 | 79.7 B / 80 B total (79 B non-emb), 3.0–3.9 B active (recomputed 79.67 B total, 3.56 B active incl. LM head) | 24 KiB (24,576 B) full attn + 72 MiB FP32 recurrent state (36 × 2 MiB) | 3.1 GiB (3.0 GiB attn + 72 MiB state) | HF `Qwen/Qwen3-Next-80B-A3B-Instruct` config.json, https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json | [W] |

Notes for M3:

- MLA latent per layer is 576 × 2 B = 1,152 B, which does not divide 4 KiB. 3 tokens per page wastes 16 %. Storing latent (1,024 B, 4 tokens/page) and RoPE key (128 B, 32 tokens/page) as two TensorSpecs avoids the waste [D].
- Parameter counts: recomputed from the dims (attention/SSM projections + MLP/MoE + embeddings; active parameters count active layers plus LM head, excluding the input embedding table lookup). All match published totals within 1 % (DeepSeek-V2-Lite 15.71 B vs 15.7 B, Gemma-3-27B text 27.01 B vs 27.02 B, GPT-OSS-20B 20.91 B / 3.61 B vs 20.91 B / 3.61 B, GPT-OSS-120B 116.79 B / 5.13 B vs 116.83 B / 5.13 B, Qwen3-Next-80B-A3B 79.67 B vs 79.7–80 B) [D].
- Qwen3-Next Gated DeltaNet layers carry a recurrent state of `32 × 128 × 128 × 4 B = 2 MiB` (FP32; 1 MiB in BF16) plus a 48 KiB causal conv1d state (`3 × 8,192 × 2 B`) per SSM layer [D]; the preset models the 2 MiB FP32 recurrent state in HBM.
- Default tier for sliding-window local KV (Gemma 3, GPT-OSS) is HBM (`phase3-decisions.md` #10), because sliding windows written to HBF die after W tokens; HBF is kept as an experiment variant (`local_tier="hbf"`).

## 5. KV compression and sparsity formats

| Format | Parameters | Effective bits/element (vs 16) | Source (title, URL) | Year | Tag / notes |
|---|---|---|---|---|---|
| FP8 KV (vLLM) | `fp8_e4m3` (CUDA and ROCm) or `fp8_e5m2`. One FP32 scale per Q/K/V tensor (`q/k/v_scale`). Optional per-head scales (Flash Attention backend, llm-compressor calibration). | 8 → 2.0× | vLLM docs, "Quantized KV Cache", https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ | 2024–2026 | [W] |
| KIVI-2 / KIVI-4 | K per-channel, grouped along tokens (group G). V per-token. Main setting G = 32, residual R = 128 full-precision recent tokens. R = 32 also evaluated. Ablations G ∈ {32, 64, 128} (128 degrades) and R ∈ {32, 64, 96, 128}. | 2-bit + FP16 scale and zero per G elements: 2 + 32/G = 3.0 bits at G = 32 (5.3×); 2.5 at G = 64; 2.25 at G = 128. KIVI-4 at G = 32: 5.0 bits (3.2×). Plus R tokens at 16 bits per sequence. | Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024, https://arxiv.org/abs/2402.02750 | 2024 | [W] for G and R. Scale/zero dtype FP16 is [M]; the overhead arithmetic is [D]. Later papers often use G = 128, R = 32 or 128. |
| KVQuant | Pre-RoPE per-channel K. Per-token V. Per-layer sensitivity-weighted non-uniform datatype. Dense-and-sparse: 1 % per-vector outliers kept in FP16. Main point 3-bit (< 0.1 PPL loss); 2-bit and 4-bit also evaluated. | About 3 bits + 1 % × (16-bit value + 32-bit index) ≈ 3.5 bits (about 4.6×). Paper claims 4.8× longer context. | Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024, https://arxiv.org/abs/2401.18079 | 2024 | [W]. Outlier byte cost is [D]/[G]. Sparse outliers break fixed per-page layouts. |
| ThinK | Query-driven pruning of key channels, keeping the top T of D per head. Orthogonal to quantization. With KIVI: 2.8× peak-memory reduction. "> 20 %" KV memory reduction overall. 40 % channel sparsity is a common comparison point. | Pruning a fraction λ of K channels: (2 − λ)/2 of KV. λ = 0.4 gives 1.25×, λ = 0.5 gives 1.33× (K only). | Xu et al., "ThinK: Thinner Key Cache by Query-Driven Pruning", ICLR 2025, https://arxiv.org/abs/2407.21018 ; "Dual-QK…" (compares at 40 % sparsity), https://arxiv.org/abs/2610.09827 | 2024–2026 | [W]. λ values 0.4 and 0.5 are [M]. |
| MLA latent | 576 elements per token per layer (512 latent + 64 RoPE key). Against Llama-3.1-70B GQA (2 × 8 × 128 = 2,048 per layer) that is 3.6× fewer; against MHA with 128 heads (32,768) it is 57× fewer. | — | DeepSeek-V2/V3 (above) | 2024 | [W]/[D] |
| Quest | Pages of 16 tokens (reimplementation default). Metadata per page per KV head: element-wise min and max of keys (2 × head_dim). Token budgets 256–4096 evaluated. First 2 layers show < 10 % sparsity (others > 90 %). | Metadata = 2/16 of K = 12.5 % of K (6.25 % of K + V) at equal dtype. Eight 16-token K pages of metadata fit in one 4 KiB page. | Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024, https://arxiv.org/abs/2406.10774 ; "The Sparse Frontier" (page size 16), https://arxiv.org/abs/2504.17768 | 2024–2025 | [W]. Metadata size [D]. "Keep first 2 layers dense" is [M] as a design rule. |
| SnapKV | Budgets 1024 / 2048 / 4096 on LongBench. Observation window at the prompt tail. Max-pooling clustering. Ablation: window 16, kernel 5. Code default window 32 [M]. | Prefill-time eviction. No per-step selection. | Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024, https://arxiv.org/abs/2404.14469 | 2024 | [W] |
| StreamingLLM | 4 sink tokens + recent window (e.g. 4 + 1020; perplexity 5.40 vs 5158 for window only) | Fixed budget | Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024, https://arxiv.org/abs/2309.17453 | 2024 | [W] |
| DeepSeek Sparse Attention (V3.2) | Lightning indexer (FP8, few heads, own K cache). Top-k = 2048 tokens per query token. | Fixed 2048-token budget per layer | DeepSeek-V3.2 paper, https://arxiv.org/abs/2512.02556 ; vLLM blog, https://vllm.ai/blog/2025-09-29-deepseek-v3-2 | 2025 | [W]. The indexer cache is an extra tensor to store. |
| NSA | Compression, selection and sliding branches. cuDNN selection block size {16, 32, 64(, 128)}, default 64. Paper values (compress 32 / stride 16 / select 64, 16 selected blocks, window 512) unverified. | — | NVIDIA cuDNN NSA docs, https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/nsa.html | 2025 | [W]/[M] |
| SPLASH sparse budget | Retrieval budget 1–5 % of context. Hot window and sinks in HBM. 16-token pages. Summary vector per page (centroid). | — | SPLASH §3, §4.2 | 2026 | [P] |
| vLLM block size | 16 tokens (default; 16–128 best on ShareGPT) | — | Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023, https://arxiv.org/abs/2309.06180 | 2023 | [W] |

## 6. Serving workload

| Quantity | Value | Applies to | Source (title, URL) | Year | Tag |
|---|---|---|---|---|---|
| ShareGPT lengths | Mean input about 161 tokens, output about 338 per request. 8.4× / 5.8× longer than Alpaca. | chat | vLLM paper (above), Fig. "input/output length distributions" | 2023 | [M]. Exact means not found by search. vLLM's loader uses only the first 2 turns. |
| LMSYS-Chat-1M | 1M conversations. 2.0 turns per conversation. Prompt 69.5 tokens, response 214.5 tokens per turn (Llama-2 tokenizer). | chat | Zheng et al., "LMSYS-Chat-1M", ICLR 2024, https://arxiv.org/abs/2309.11998 | 2024 | [S] (via summary; another summary gives 69.8 / 215.7) |
| Azure LLM trace (Nov 2023) | Coding: median prompt 1,500, output 13. Conversation: median prompt 1,020, output 129. | production | Patel et al., "Splitwise", ISCA 2024, https://arxiv.org/abs/2311.18677 ; Azure Public Dataset, https://github.com/Azure/AzurePublicDataset | 2023–2024 | [W] |
| Mooncake (Kimi) trace | Records arrival time, input/output length and 512-token block hashes. Averages about 7.6K input / 180 output. | production long-context | Qin et al., "Mooncake", FAST 2025, https://www.usenix.org/system/files/fast25-qin.pdf | 2025 | [M] for the averages. Compute from the trace if used. |
| Coding agents (TraceLab) | Median call: about 119K prefix tokens (cached history), 875 new input tokens, 214 output tokens. Prefix hit rate 95.7 %. Typical prefix 32K–256K. Cache misses cause 3.8× more prefill than truly new tokens. | agentic | "TraceLab: Characterizing Coding Agent Workloads for LLM Serving", https://arxiv.org/abs/2606.30560 | 2026 | [W] |
| SWE-bench Pro Codex traces | Input tokens: mean 68,329, P50 63,917, P90 114,888. Cache hit rate 94.2 %. | agentic | HF dataset card, https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces | 2026 | [S] |
| GitHub Copilot | 3.2 M users, 13 M sessions, 761 M LLM calls, 95 T tokens (June 2026 sample). Agent loops couple LLM calls about 1:1 with tool calls. | agentic | Microsoft Research, "ghcp_traces", https://www.microsoft.com/en-us/research/wp-content/uploads/2026/08/ghcp_traces-6.pdf | 2026 | [W] |
| DeepSeek V3/R1 production (24 h) | 608 B input tokens, 56.3 % (342 B) from on-disk KV cache. 168 B output tokens. Per H800 node: about 73.7K input tok/s (prefill, including hits) or 14.8K output tok/s (decode). 20–22 tok/s per request. Mean KV length per output token 4,989. 226.75 nodes on average. | production (MLA) | DeepSeek Open Source Week Day 6, "Inference System Overview" (github.com/deepseek-ai/open-infra-index); summary: https://www.analyticsvidhya.com/blog/2025/03/deepseek-inference-system-overview/ | 2025 | [S]. Derived: about 88 concurrent decode requests per GPU [D]. |
| KV written per GPU-day (DeepSeek-derived) | 22.7 TB/day per prefill GPU (≈ 263 MB/s). 26.9 TB/day per decode GPU (≈ 311 MB/s). MLA at 70 KB/token. | production | FlashAccel §II (from the DeepSeek report) | 2026 | [P] |
| Decode batch | vLLM `--max-num-seqs` default cap 256. FlashAccel: 8×H200 holds batch 30 for Llama-3.1-405B at 15K context (HBM-capacity bound); HBF raises it to about 110. Bandwidth saturates around B ≈ 128 on H100 in one study. | serving | vLLM forum, https://discuss.vllm.ai/t/why-is-ttft-worse-for-decode-1-than-decode-100/2590 ; FlashAccel §VI | 2023–2026 | [W]/[P] |
| Prefill throughput, measured anchors | Llama-3.1-8B, 1 × H100, vLLM BF16: about 18.3K tok/s at 4K input, 16.2K at 16K (mixed throughput). Llama-3.1-70B, 4 × H100: 8.2K tok/s at 16K (about 2K per GPU). Llama-3-8B: 1M-token prefill takes 30 min on 1 × A100. | GPU | Snowflake, "SwiftKV", https://www.snowflake.com/en/engineering-blog/swiftkv-llm-compute-reduction/ ; Jiang et al., "MInference 1.0", NeurIPS 2024, https://arxiv.org/abs/2407.02490 | 2024 | [W] |
| Prefill throughput, analytic | FLOP/token = 2P + 2·L·(n_q·d_h)·S (causal, mean over an S-token prompt). At 50 % of dense BF16. Llama-3.1-8B on H100: 29.8K / 20.1K / 9.8K tok/s at S = 2K / 32K / 128K; on B200 67.8K / 45.6K / 22.3K. Llama-3.1-70B per H100: 3.4K / 2.7K / 1.6K; per B200: 7.8K / 6.1K / 3.6K. | GPU | derived. The MInference check gives 155 TFLOP/s on A100 (50 % of 312), so 50 % MFU is consistent. | — | [D] |
| Prefill-bound KV write rate | BF16 KV, = prefill tok/s × KV/token. 8B on H100: 3.9 / 2.6 / 1.3 GB/s (2K / 32K / 128K); on B200 up to 8.9 GB/s. 70B per H100: 1.1 / 0.9 / 0.5 GB/s; per B200: 2.6 / 2.0 / 1.2 GB/s. | HBF write load | derived from the previous row and §4 | — | [D]. Upper bound: assumes every prefilled token goes to HBF and no prefix-cache hits. |
| Decode-bound KV write rate | = batch × (1/TPOT) × KV/token. Example: 64 seqs × 20 tok/s × 128 KiB ≈ 168 MB/s (8B). | HBF write load | derived | — | [D] |

## 7. Derived checks the simulator should reproduce

The arithmetic is in `phase2/calc/derive.py`.

**Lifetime write budget per 512 GiB stack.** At WA = 1 over 5 years:

| N_PE | Budget per stack | Full-drive writes per day |
|---|---|---|
| 3e3 | 10 MB/s | 1.6 |
| 1e4 | 35 MB/s | 5.5 |
| 3e4 | 105 MB/s | 16 |
| 1e5 | 349 MB/s | 55 |
| 1e6 | 3.5 GB/s | 548 |

Against the 0.3–4 GB/s per-GPU write rates in §6, endurance binds unless N_PE ≥ 1e5 and several stacks share the load. FlashAccel's "about 55 P/E per day" for SLC over 5 years matches the 1e5 row.

**Array program bandwidth per stack.** 1,024 × 4 KiB / tPROG = 55.9 / 41.9 / 21.0 / 10.5 GB/s at tPROG = 75 / 100 / 200 / 400 µs. This is far above KV write demand, so write throughput is not the limit; read interference (M5) is.

**Read disturb under dense decode.** This is the most sensitive endurance term. Dense decode senses every KV page once per step (one page holds one layer/head/K-or-V slice). A fully occupied block of `ppb` pages therefore sees ppb / TPOT reads per second. At 256 pages and 50 ms TPOT that is 5,120 reads/s:

| Refresh threshold (block reads) | Refresh period | Refresh rewrites per block per day |
|---|---|---|
| 1e5 | every 20 s | 4,424 |
| 1e6 | every 195 s | 442 |
| 1e7 | every 33 min | 44 |

So dense KV reads exhaust a 1e5 P/E budget in under a year unless the threshold is ≥ 1e7, the data lives for less than one refresh period, or selection is sparse. A 1–5 % budget divides the read rate by 20–100. The spec's per-block read counter and CECC "Read Count > Max_Read_Threshold" (§11.5.1.5) make this host-visible. Neither SPLASH nor FlashAccel models read disturb.

**Retention refresh.** Data that outlives 24 h is rewritten daily: 1 P/E per day, about 1,825 over 5 years. That is about 2 % of 1e5 (SPLASH says < 2 %), but 18 % of a 1e4 pSLC budget. KV regions with residency below 24 h (SPLASH mean 30.8 min) need no retention refresh.

## 8. Recommended sweep ranges

The nominal values match the repo defaults where those exist.

### HBF device (M0, M4, M5)

| Parameter | Low | Nominal | High | Reason |
|---|---|---|---|---|
| tR (array sense) | 2 µs | 4.096 µs | 10 µs (stress 20 µs end-to-end) | Low: OCP grade 2 or SanDisk 1.6 TB/s over 1,024 banks implies 2.6–2.7 µs, and Z-NAND is 3 µs. Nominal: SPLASH, FlashAccel and Kouchi 2020, and the repo default. High: XL-Flash "< 5 µs" with margin, and grade 1 at 1,024 banks implies 10.9 µs. 20 µs is SlipStream's request-to-return latency, not tR. |
| Banks per stack | 1,024 fixed (spec Fig. 4) | 1,024 | 2,048 (ablation) | Bandwidth is swept through tR. 2,048 tests whether grade 3 (3.072 TB/s) is reachable at tR ≈ 2.7 µs. |
| Page size | 4 KiB | 4 KiB | 16 KiB (off-spec ablation only) | Fixed by spec §4.1. 16 KiB is the commodity TLC page size. Use it only to show read amplification. |
| tPROG | 75 µs | 100 µs | 350 µs | Kouchi/FlashAccel; SPLASH, Z-NAND and the repo default; Aero's TLC config as a pessimistic pSLC-on-TLC bound [G for pSLC]. |
| tBERS | 1 ms | 3.5 ms | 10 ms | SPLASH and repo default are 1 ms (optimistic). 3.5 ms is the Aero 3D base erase and planar SLC 2–3.5 ms. 10 ms is the S5E 3D TLC datasheet. Auto-erase on page-0 write blocks the bank (spec §5.4.1). |
| Pages per block | 192 | 256 | 768 | SPLASH/FlashAccel (96 WL × 2). Repo/MQSim default. 200+ layer SLC block ≈ WL × strings [D]. Static mapping reclaim granularity = 1,024 × ppb × 4 KiB: 0.75 / 1 / 3 GiB. |
| N_PE (max PEC) | 1e4 | 1e5 | 1e6 | pSLC low end; SPLASH/FlashAccel/H³; FlashAccel's 10× relaxed-retention assumption (literature upper bound 50×). Add 3e3 as a "TLC HBF" ablation. |
| Read-disturb threshold (block reads before refresh) | 1e5 | 1e6 | 1e7 | 3D TLC measured 5.5e4–5.2e5 (STRAW 2025), a conservative stand-in for the unknown SLC value; classic SLC 1e6 (Cai 2015); 1e7 is [G]. Report results as a curve over this parameter; it dominates dense-KV lifetime (§7). |
| Retention refresh interval | 24 h | 24 h | 48 h | Spec §9 guarantees 24 h; §11.5 says 24–48 h. Also run an "off" ablation when KV residency is under 24 h. |
| Host-compaction WA | 1.0 | 1.17 | 2.0 | Request-granular superblocks (ideal); SPLASH overall; SPLASH 10K contexts 1.85 rounded up for vLLM-style interleaving. |
| Lifetime target | 3 yr | 5 yr | 10 yr | SPLASH and FlashAccel use 5 years; spec lifespan is 10 years (§9). |

### GPU compute (M1)

| Parameter | Low | Nominal | High | Reason |
|---|---|---|---|---|
| GPU | H100 SXM (989 / 1,979 TF dense, 80 GB, 3.35 TB/s) | B200 (2,250 / 4,500 TF, 180 GB, 8 TB/s) | B300 / GB300 (about 2,500 / 5,000 TF, 288 GB, 8 TB/s) | HBF is a 2027+ part, so B200-class is the realistic host. SPLASH uses B200. |
| Prefill MFU | 0.4 | 0.5 | 0.7 | 0.5 reproduces the MInference A100 number; FlashAttention-3 kernels reach 0.75–0.85. |
| Decode HBM efficiency | 0.7 | 0.8 | 0.9 | B ≈ 128 saturates at 84–86 % of peak HBM bandwidth (H100 study cited in §6, decode-batch search) [W]. |
| HBF stacks per GPU | 4 | 8 | 8 | SPLASH: 8 × 384 GB = 3 TB at 8 TB/s. SanDisk: 8 × 512 GB = 4 TB. |

### Model, context and batch

| Parameter | Low | Nominal | High | Reason |
|---|---|---|---|---|
| Model | Qwen2.5-7B (56 KiB/token, 4 KV heads) | Llama-3.1-8B (128 KiB) and Llama-3.1-70B (320 KiB) | Qwen3-32B (256 KiB); DeepSeek-V3 MLA (68.6 KiB, latent) as the compression extreme | Covers a 2.3× KV/token spread among GQA models, plus MLA. |
| Context | 8K | 32K | 128K (stress 1M) | 1–2K is chat (Azure, ShareGPT); 32K–128K is agentic (TraceLab median prefix 119K; SWE-bench Pro mean 68K). |
| Decode batch | 1 | 32 | 256 | 1 is the single-sequence repo mode; DeepSeek-derived about 88 per GPU; vLLM cap 256. |
| TPOT target | 20 ms | 50 ms | 100 ms | FlashAccel SLOs 50 / 100 ms; SPLASH 100 ms; 20 ms interactive [G]. |

### Compression and sparsity (M3)

| Parameter | Low | Nominal | High | Reason |
|---|---|---|---|---|
| KV format (bytes vs BF16) | FP8 (2×) | KIVI-4 G32 (3.2×) | KIVI-2 G32 R128 (5.3× on the quantized part); ThinK λ = 0.5 stacks another 1.33× on K | §5. Model scale/zero as separate small tensors, because they change pages per token. |
| Sparse budget | 1 % of context (SPLASH low) | 2048 tokens (DSA top-k; Quest 1K–4K) | 5 % or 4096 tokens | SPLASH 1–5 %; Quest 256–4096; DSA 2048. Quest pages are 16 tokens; metadata adds 12.5 % of K. |
| Sink + window kept dense (in HBM) | 4 + 0 | 4 + 128 (KIVI residual R) | 4 + 1020 | StreamingLLM 4 sinks; KIVI R = 128. |

### Workload (M2, M4)

| Parameter | Low | Nominal | High | Reason |
|---|---|---|---|---|
| Input length per request | 161 (ShareGPT mean) | 1,020–1,500 (Azure medians) | 64K–120K prefix (agentic) | §6. For agentic traces, the new tokens written per call are only about 875 (TraceLab), so prefix-cache hit rate (94–96 %) cuts KV writes about 20×. |
| Output length | 13 (Azure coding median) | 129–338 (chat) | 2K (reasoning) | Last value is [G]: reasoning traces are longer, and no trace with output lengths was found. |
| Arrival rate per GPU | — | prefill tok/s ÷ mean uncached input | — | Bound by the §6 analytic prefill rate. Example: 8B on H100 at 32K is about 20K tok/s, so about 0.6 req/s per GPU of uncached 32K prompts. |
| KV write rate per GPU | 0.17 GB/s (decode-only, 8B, B = 64) | 0.8–1.0 GB/s (SPLASH and FlashAccel measured) | 3–9 GB/s (back-to-back short-prompt prefill, 8B, H100–B200) | §6 [D]. Compare with the 0.35 GB/s per stack 5-year budget at 1e5 P/E. |

## 9. Gaps and caveats

- **Pages not opened directly.** WebFetch could not resolve tomshardware.com, storagereview.com or arxiv.org from this session. Web-sourced numbers therefore rest on search excerpts ([W]/[S]), not full-page reads. Allow those hosts in the environment's network settings to re-verify the [M] items. These are: KIVI scale dtype; ShareGPT and Mooncake means; NSA hyperparameters; ThinK λ values; Qwen2.5-32B hidden/FFN; DeepSeek-V2 dims; JESD218 retention.
- **No vendor HBF tR, tPROG, tBERS, block size, P/E or read-disturb numbers exist publicly** (spec v0.7.0 leaves them product-specific; HBF-Sim and SlipStream say the same). Every HBF timing in use today traces back to SLC research dies: Kouchi 2020 (4 µs / 75 µs) and XL-Flash (< 5 µs, 4 KB, 16 planes).
- **HBF cell type is undisclosed.** SanDisk's "16 × 256 Gb dies, 512 GB" says nothing about SLC vs MLC. SPLASH, FlashAccel and H³ assume SLC (or 1e5 cycles).
- **Spec Table 3 ("16 banks/channel") and §4.6 Fig. 4 (16 banks/die × 4 dies) disagree** on bank count. The repo follows Fig. 4 (1,024 banks).
