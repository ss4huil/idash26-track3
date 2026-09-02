# LSS 在线原语协议规范（v2 Phase 1）

> 本文档是 `gpu_mpc/lss/` 的协议唯一权威定义。写代码前先定公式；
> 任何实现与本文档不一致时以本文档为准（或先改本文档再改代码）。
> 协议语义参考 SCI（CrypTFlow2/SIRNN），但**不链接、不依赖 SCI**，
> 全部干净重写：dealer 直给相关性，无 OT extension、无 split-OT。
>
> 安全假设（与比赛 Q3 一致）：dealer **半诚实且不与任一方合谋**；
> 在线阶段无 dealer。两方均为半诚实。环 = Z_{2^64}。

## 记号

- party ∈ {0, 1}；`[i==0]` 表示 party 0 为 1、party 1 为 0 的常量。
- 布尔份额：`x = x0 ⊕ x1`（XOR，单比特）。
- 算术份额：`x = x0 + x1 mod 2^64`。
- `open(v)`：双方交换各自份额并重构（XOR 或加），1 轮通信。

## 原语 1：AND（布尔 Beaver 三元组，记录类型 BIT_TRIPLE）

Dealer：采样 a, b ← {0,1}，c = a∧b；XOR 拆成布尔份额发给双方。
每方记录 = (a_i, b_i, c_i)，3 bit。

在线（输入布尔份额 x_i, y_i；输出 z = x∧y 的布尔份额 z_i）：

    e = open(x ⊕ a)          // 每方发 1 bit：e_i = x_i ⊕ a_i
    f = open(y ⊕ b)          // 同上，f_i = y_i ⊕ b_i（与 e 同轮发送）
    z_i = c_i ⊕ (f ∧ a_i) ⊕ (e ∧ b_i) ⊕ (e ∧ f ∧ [i==0])

正确性：z = (a⊕e)(b⊕f) = ab ⊕ af ⊕ eb ⊕ ef，ef 项只加给一方。
通信：双向各 2 bit，1 轮。语义对应 SCI `traverse_and_compute_ANDs` 的 Beaver open。

## 原语 2：1oo16 OT（Millionaires 叶子层，记录类型 OT16_SEND / OT16_RECV）

Dealer：采样 r ← [0,16)，16 个 pad k_0..k_15 ← {0,1,2,3}（各 2 bit）。
- sender 方记录：k[0..15]（32 bit）；
- receiver 方记录：(r, k_r)（4+2=6 bit）。

在线：receiver 持选择 d ∈ [0,16)；sender 持明文函数 g: [0,16) → {0..3}
（在 Millionaires 中 g(i) = ((digit>i) ^ mask_cmp)<<1 | ((digit==i) ^ mask_eq)，
mask 由 sender 本地采样，作为其输出份额 s；sender 令 f(i) = g(i) ⊕ s）。

    receiver → sender: δ = (r − d) mod 16        // 4 bit
    sender → receiver: c_i = f(i) ⊕ k_{(i+δ) mod 16},  i = 0..15   // 16×2 bit
    receiver 输出: c_d ⊕ k_r  =  f(d) = g(d) ⊕ s
    sender 输出: s

下标验证：d + δ ≡ d + (r − d) ≡ r (mod 16)，故 c_d 用的 pad 恰为 k_r。✓
⚠️ 方向：δ 是 **(r − d)** 不是 (d − r)。这与 SCI `split-kkot.h:got_recv_online`
的 `a = (r_off − r) & mask` 完全一致（r_off=离线选择，r=真实选择）。
此公式是本目录单元测试的重点验证对象。

## 原语 3：MUX（比特 × 环元素，记录类型 MUX_TRIPLE）

Dealer：采样 a ← {0,1}，b ← Z_{2^64}，c = a·b。
发双方：a 的**布尔份额** a^b_i（1 bit）、a 的**算术份额** a^a_i
（dealer 取 a^a_0 ← {0,1}，a^a_1 = a − a^a_0 mod 2^64，故 a^a_i ∈ {0, 1, 2^64−1}，
按完整 u64 存）、b 的算术份额 b_i、c 的算术份额 c_i。

在线（输入布尔份额 β_i 和算术份额 x_i；输出 z = β·x 的算术份额）：

    e = open(β ⊕ a^b)        // 1 bit/方
    f = open(x − b)          // 64 bit/方，与 e 同轮
    若 e == 0:  z_i = c_i + a^a_i · f
    若 e == 1:  z_i = x_i − c_i − a^a_i · f

正确性：e=0 时 β=a，z = c + a·f = a(b+f) = a·x ✓；
e=1 时 β=1−a，z = x − a·x = (1−a)·x ✓。
**为什么 a 要存两种份额**：a·f 在环上分配需要算术份额（a^a_0·f + a^a_1·f = a·f），
布尔份额下 a0⊕a1 ≠ a0+a1，本地不可算——这是与 SCI 用 2 个 COT 的 multiplexer
等价的单轮替代，代价是 key 里多 8 B。
通信：双向各 65 bit，1 轮。语义对应 SCI `AuxProtocols::multiplexer`。

## 原语 4：B2A（布尔份额 → 算术份额，记录类型 B2A_CORR）

Dealer：采样 r ← {0,1}；发双方 r 的布尔份额 rb_i（1 bit）和算术份额 ra_i（u64）。

在线（输入布尔份额 β_i；输出 z = β 的算术份额，CrypTFlow2 公式）：

    e = open(β ⊕ rb)         // 1 bit/方
    若 e == 0:  z_i = ra_i
    若 e == 1:  z_i = [i==0] − ra_i

正确性：e=0 时 β=r，z = r ✓；e=1 时 β = 1−r，z0+z1 = 1 − r ✓。
通信：双向各 1 bit，1 轮。

## key 文件格式 v1（`lss_keys_party{0,1}.bin`）

- 与 FSS key 文件**分离**（P3 再决定是否合并）。两侧记录严格同序。
- Header（32 B，字节对齐）：magic "LSS1"(4B) | version u32=1 | party u32 |
  num_records u64 | payload_bits u64。
- 记录流：**单一连续比特流，LSB-first 打包**（bit i → byte i/8 的第 i%8 位）。
  每条记录以 3-bit 类型标签开头：
  0=BIT_TRIPLE, 1=OT16_SEND, 2=OT16_RECV, 3=MUX_TRIPLE, 4=B2A_CORR。
  标签用途：消费方校验 == 期望类型，第一时间暴露 keygen/eval 图序错位
  （设计文档 §9 风险登记第 1 条）。
- 记录负载（标签之后）：
  - BIT_TRIPLE：a,b,c 各 1 bit → 共 6 bit = 0.75 B
  - OT16_SEND：16×2 bit pad → 35 bit = 4.375 B
  - OT16_RECV：r(4) + pad_r(2) → 9 bit = 1.125 B
  - MUX_TRIPLE：a^b(1) + a^a(64) + b(64) + c(64) → 196 bit = 24.5 B
  - B2A_CORR：rb(1) + ra(64) → 68 bit = 8.5 B
- Trailer（24 B，字节对齐）：num_records u64 | payload_bytes u64 |
  checksum u64（payload 字节的 FNV-1a 64）。

## PRG 说明

dealer 用 `std::mt19937_64`（种子由参数给定，确定性可复现）。
统计上均匀独立，对"dealer 直给"模型的半诚实安全性足够；
**生产环境建议换 AES-CTR**（TODO，见汇报遗留问题）。
