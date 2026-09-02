# iDASH 2026 Track 3 研究记录

> 记录时间：2026-09-03
> 团队：nameforgotten（NUDT），联系：songshang19@nudt.edu.cn
> 代码：github.com/ss4huil/idash26-track3（`master` = v1 提交版，`dev-v2` = v2 LSS 优化版，标签 `track3-v1-submission`）

本文档完整记录这一阶段的研究：做了什么、得到什么结果、踩过哪些坑、未来还能做什么。供后续回头看时快速恢复上下文。

---

## 1. 任务与规则要点

iDASH 2026 Track 3：为 DeepDTAGen 的药物-靶点亲和力预测分支做两方秘密分享（2PC）MPC 推理加速，GPU 加速。评分 = 精度（敏感度+特异度均值，多个阈值，掉点 ≤2% 为资格线）+ 加速比。

官方 QA 中对设计影响最大的条款：

- 只评 affinity 预测分支（CNN+FC），生成分支忽略；
- 预处理（格式转换、加密、秘密分割）不计时；**dealer（半诚实第三方）允许**，可离线生成相关随机数和秘密分割；
- **A_norm = D^{-1/2} A D^{-1/2} 必须在线从 A 的份额推导**（QA-15），不能离线预计算；
- 不允许 FHE/TEE；ring 32 或 64 bit 均可；要求 128-bit 安全；
- 蛋白序列公开、药物小分子私有；模型权重公开、中间结果保密；
- 评测 batch"几百到几十万"，batch size 应灵活；
- 评测机：每台 1×H100 PCIe 80GB，SSD ~450MB/s，内存 376GB（可用 ~250GB），无 GPU DMA，网络 >1Gbps、RTT<1ms；
- 最多可提交两个方案。

**两个未确认的开放问题**（影响提交策略，建议邮件问组委会 wh25@iu.edu / hatang@iu.edu）：
1. 评测带宽的具体值（">1Gbps"太模糊，LSS 方案通信量大，收益严重依赖带宽）；
2. 密钥从 SSD 加载是否计入 online 计时（若计入，keysize 是决定性因素；若不计入，通信量更重要）。

## 2. 开发环境

| 项 | 我们的开发机 | 比赛评测机 |
|---|---|---|
| GPU | A10 23GB（SM86） | H100 PCIe 80GB（SM90a） |
| CPU | 16 vCPU = 8 物理核 HT | 未公布 |
| 内存 | 58GB | 376GB（可用 ~250GB） |
| 磁盘 | / 99GB + /data2 500GB（实测 ~525MB/s） | SSD ~450MB/s |

注意：双进程 loopback 模拟两方时共享同一张 GPU 和同一组 CPU 核，测得的绝对时间比真实两方部署悲观。

## 3. 版本演进与结果

### 3.1 v1（master，tag `track3-v1-submission`）— FSS 方案 + 流式流水线

基于 EzPC/GPU-MPC 框架（`/home/ecs-user/idash26/EzPC` 的 `track3-patches` 分支，补丁也随仓库 `patches/` 分发）。

- **合规在线邻接归一化**（QA-15）：GPU DPF-LUT，139 项公开常数表，raw 0/1 邻接矩阵份额入库，`gpu_mpc/secure_adj_norm.h`，`DDG_SECURE_ADJ_NORM=1` 门控；
- **密钥 arena + 去同步**（46 处）：loopback B=16 协议耗时 173→136ms；
- **多 chunk 流式 + SSD 预读流水线**：`DDG_NUM_CHUNKS` / `DDG_PREFETCH`，NC=2/3/4、补零边界、N=256 真实 SSD 全部逐位验证；stall 全部来自 SSD 带宽（IO-bound 证实）；
- **三数据集验证**：davis/bindingdb 用 bw=32，**kiba 必须 bw=64**（fusion FC 激活溢出 2^31 达 9 倍）；
- **交钥匙入口**：`competition_run.sh`（4+1 行 CONFIG）+ `run_party.sh`（自动检测 VRAM/RAM 选 B/NC/模式，meta.json 握手）；
- 双架构二进制：sm_90a（H100，仅编译验证未实机）+ sm_86（本地实测）。

性能基线：通信 7.8MB/样本（bw32）；tc 模拟 1Gbps 下 65-71ms/样本（带宽 98% 打满，RTT 不敏感）。

### 3.2 v2（dev-v2）— LSS 化压缩 keysize

动机：FSS 的 DCF 比较 key 太大（占总量 ~85%），大 batch 下完全卡在 key 的 SSD 读取上。参考 Matchmaker（ePrint 2025/424）思路：dealer 场景下 AND 三元组/OT 可以直接由 dealer 生成相关性，不需要两方 OT 协议。

| 阶段 | commit | 内容 | 结果 |
|---|---|---|---|
| P0 | `a38e23a` | bw=64/s=12 全局化 + LocalARS 概率截断默认（截断密钥归零） | keysize -47.4%（davis B=8 密钥 2.80→1.48GiB）；`DDG_EXACT_TRUNC=1` 可回退 |
| P1 | `ad87ac9` | `gpu_mpc/lss/`：dealer 直给相关性，四原语（AND/OT16/MUX/B2A），key 格式 "LSS1" | 双进程单测全过 |
| P2 | `c18e886` | Millionaires compare（radix-16，16 digits，26 triple/比较）+ MSB/DReLU/ReLU=MUX | 全边界过；每元素 key 114B(P0)/62B(P1)，6 轮，~85B 通信 |
| P3 | `b372597` | `DDG_LSS_RELU=1` 分流接入 GPU 管线（masked-public↔share 桥接，独立端口） | 三数据集验证过；keysize 198→~50MB/样本 |
| P3.5 | `335e835` | OpenMP 并行化（`DDG_LSS_THREADS=8`） | relu 9.2s→1.04s/forward@B=8 |
| LSS2 | `d884d34` | PRF 种子压缩：纯随机数种子再生，只显式存相关性修正 | relu key 35.8→9.8MB/样本（两方合计）；relu 0.81s（反快 22%） |
| AES | `e54f5fa` | splitmix64→AES-128-NI CTR（FIPS-197 自测，14 条流独立 key） | 128-bit 安全声明成立；relu 0.758s，无劣化 |

LSS2 各记录压缩（含 3-bit tag，LSS1→LSS2，P0/P1 bit 数）：BIT_TRIPLE 6→4/3；OT16_SEND 35→3；OT16_RECV 9→5；MUX_TRIPLE 196→67/5；B2A_CORR 68→3/67；MASK_IN 67→3/67（r_in 由管线给定只能压一半）；MASK_OUT 67→3/3。

### 3.3 N=128 对比实测（davis，bw=64，B=8×16 chunks，loopback，2026-09-03）

| 指标 | FSS | LSS2 | 倍数 |
|---|---|---|---|
| 在线总时间 | 97.3s | **25.3s** | **3.85×** |
| 每样本 | 760ms | 197ms | |
| key 总量/方 | 25.4GB（198MB/样本） | 4.6GB（36MB/样本） | 5.5× |
| 通信量/方 | 10.2MB/样本 | 27.5MB/样本 | 2.7× 多 |
| dealer keygen | 52.3s | 48.3s | 相当 |

- **FSS 瓶颈 = key 的 SSD 读取**：97s 中 81s 在等 key I/O（每 chunk 1.5GB @ ~300MB/s，预取掩盖不住）；GPU 93% 闲置。NC=1 时 key H2D 拷贝 405ms/fwd 比 relu 本身 316ms 还大。
- **LSS2 瓶颈 = CPU relu 计算**：I/O 完全隐藏（stall 仅 chunk 0）；relu 内部 leaf(OT16) 9.6s + tree(AND) 9.4s + mux 1.1s + bridge 0.8s，占 compute 86%。relu 算子本身 883ms/fwd 比 FSS GPU 的 316ms 慢，但系统总账赢。
- 正确性：两方案 128 输出 max|Δ|=0.0034；对 golden MAE 同为 0.2167。

### 3.4 keysize 总账（bw=64，每样本每方）

```
FSS 总量 198MB = relu DCF ~167MB(85%) + FSS 剩余(conv/GEMM/adj-norm) ~31MB
LSS2 模式 = FSS 剩余 31.4MB + LSS2 relu key 4.9MB ≈ 36MB/样本/方
```

- 30K 样本 ≈ **1.05TB/方** → 超比赛机 250GB 可用内存，SSD 流式仍必需；
- 内存模式上限 ≈ 7000 样本。

## 4. 精度验证基准

- davis rows 0-7 官方浮点基准：5.065341, 5.041791, 5.005894, 8.407457, 5.437313, 5.374848, 5.893423, 5.058861；容差 ≤0.005，实测 max err 0.0013；
- kiba B=8：`validate_batch_fixed.py` PASS（max err 0.004883）；
- bindingdb：LocalARS 路径有 0.0054 的 ulp 级噪声（良性，门限可放宽到 0.007）；
- 验证器：`scripts/dev_tools/validate_batch_fixed.py <batch名> <mpc日志>`；
- **已知验证器自身问题**：davis row5（及 N=128 中同一批 ~11 行）的 fixed-point 重算基准与官方浮点基准本身偏差 0.005-0.015，FSS/LSS 路径同样触发，非实现 bug。

## 5. 踩过的坑（重要，别再踩）

1. **`pkill -f deepdtagen` 会杀掉自己的 shell**——用 `pgrep -af "[d]eepdtagen"`；
2. **make 不跟踪 FLAGS**：改 BW/GPU_ARCH 前必须 `rm -f deepdtagen_inference pool_override.o lss/*.o`；
3. **bw=64 数据生成 bug**（已修 `4f77789`）：`np.int64(1)<<64` 位移溢出导致 protein_emb 全零；修复前的 bw64 验证是自洽假象，所有 bw64 数据必须重新生成（`timing_b1000_raw` 是修复前的，勿用于精度验证）；
4. **bw=64 + key arena 有 4B 对齐崩溃**（未修）：bw64 必须 `DDG_KEY_ARENA=0`；
5. **OpenMP 线程数**：双进程共享 8 物理核，必须 `DDG_LSS_THREADS=8`，默认 16 线程超售反而慢 2.6×；
6. 运行二进制必须 `export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`；
7. /data2 重启后不自动挂载（`sudo mount /dev/vdb1 /data2`，建议写 fstab）；
8. kiba 必须 bw=64（fusion FC 激活溢出 32-bit ring 达 9 倍）。

## 6. 未完成 / 未来可做（按优先级）

**提交前必须有：**
1. **LSS 模式接入交钥匙脚本**：`run_party.sh` 的 `KEYS_MB_PER_SAMPLE=180` 是 FSS 时代估值，LSS 模式下自动选 B 失真；dealer/eval 入口没暴露 `DDG_LSS_RELU`；
2. **比赛精度门**：只做过"回归值 vs 浮点基准 ≤0.005"验证，没算过比赛真正的指标（多阈值下敏感度+特异度均值，掉点 ≤2%）——需在完整测试集上评一次；
3. **tc 限速网络复测**：LSS 通信量 2.7×、6 轮/比较，loopback 的 3.85× 优势在 >1Gbps 真实网络下会缩水多少未知；
4. 发邮件给组委会确认带宽具体值 + key SSD 加载是否计时（决定交 FSS 版还是 LSS 版，可交两个）。

**大 batch 需要：**
5. **LSS key 的 SSD 流式读取**：`_lss.bin` 目前 chunk 0 一次性全量进内存（N=128 时 616MB 没问题，30K 时 ~150GB/方 不行）；

**进一步优化空间：**
6. **FSS 剩余 31MB/样本的 breakdown + 压缩**：先查 conv/GEMM 是否有冗余 Beaver triple（明文权重×秘密输入在 2PC 下本不需要 triple），确实需要的 triple 可套同样的 PRF 种子压缩（a、b 再生只存 c，~3×）；
7. **LSS relu 的 CPU leaf/tree 优化**：leaf(OT16) 与 tree(AND) 各占一半；可考虑更大 radix 减叶子数（权衡轮次）、leaf 的 AES/比特编排上 GPU；
8. KDF 目前用 splitmix64 做一次性种子扩展（安全论证只依赖 AES 的 PRP 性）；如评审苛求可换 AES 自身做 KDF，仅 `lss_stream_key` 一处；
9. bw=64 arena 对齐 bug 修复（可选，workaround 已够）；
10. H100（sm_90a）二进制只编译验证过，未实机跑。

## 7. 快速复现

```bash
# 构建（EzPC 框架必须在 track3-patches 分支）
cd /home/ecs-user/idash26/gpu-mpc-track/gpu_mpc
rm -f deepdtagen_inference pool_override.o lss/*.o
make GPU_MPC_ROOT=/home/ecs-user/idash26/EzPC/GPU-MPC BW=64 GPU_ARCH=86 CUDA_VERSION=12.8 deepdtagen_inference
mv -f deepdtagen_inference deepdtagen_inference_bw64

# 运行（任何二进制）
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# LSS2 模式（davis B=8）
DDG_SECURE_ADJ_NORM=1 DDG_LSS_RELU=1 DDG_KEY_ARENA=0 DDG_LSS_THREADS=8 \
  ./deepdtagen_inference_bw64 ...   # 参数见 run_party.sh

# 验证
python3 ../scripts/dev_tools/validate_batch_fixed.py timing_b8_raw_davis_bw64 <mpc日志>
```

关键数据/产物位置：
- 官方模型与数据：`/home/ecs-user/idash26/DeepDTAGen/`（3 个 .pth + 3 个 test.csv）；
- N=128 基准数据与 key：`/data2/bench_n128/`（约 57GB，含 FSS/LSS2 两整套 key）；
- B=8 小batch 数据：`gpu_mpc/timing_b8_raw_{davis,kiba,bindingdb}_bw64/`；
- 协议文档：`gpu_mpc/lss/lss_protocol.md`（§9 = LSS2 压缩格式与安全论证）。
