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

## 5. Millionaires 比较 compare_gt（P2 算子层）

语义：party0 持明文 a、party1 持明文 b（各自本地已知，实际用途是各自的
share 值），输出 1{a > b} 的布尔份额。对齐 SCI `millionaire.h` 的
`compare(greater_than=true)`；1{x<y} 由双方互换输入得到。party0 恒为
OT sender。

结构（bitlength ∈ [1,64]，D = ceil(bitlength/4) 个 digit，radix-2^4；
顶层 digit 可能只有 r = bitlength mod 4 个有效 bit，仍用 OT16，只用前
2^r 个入口）：

- **叶子层**（digit i 从 LSB 编号 0..D−1）：每 digit 一个 OT16，
  sender 的明文函数 g(k) = ((t>k)<<1)|(t==k)（t = 本方 digit），
  掩码 s（2 bit）来自 sender 本地 PRNG、同时即 sender 的份额；
  receiver choice = 本方 digit。产出布尔份额对：
  cmp_i = 1{a 的 digit_i > b 的 digit_i}（payload 高 bit），
  eq_i = 1{digit 相等}（低 bit；digit 0 的 eq 不使用——严格大于的
  组合中 LSB 的 eq 无意义，与 SCI 一致）。
  全部 n×D 个 OT 一批调用，**1 轮**。
- **AND 树**（stride i = 1,2,4,...，ceil(log2 D) 层，每层 1 轮）：
  组 j 覆盖 digit 段 [j, j+i)。同层所有 AND 收集后一次 batched
  and_open，再统一写回（组内先读后写安全）：
  - j==0 组（LSB 组，无 eq）：cmp[j] ← (cmp[j] ∧ eq[j+i]) ⊕ cmp[j+i]
  - j>0 组：eq[j] ← eq[j] ∧ eq[j+i]；
            cmp[j] ← (cmp[j] ∧ eq[j+i]) ⊕ cmp[j+i]
  不变式：cmp[j] = 1{该段 a > 该段 b}，eq[j] = 1{该段相等}；
  最终 cmp[0] = 1{a>b}。
  门数（D=16）：15+7+3+1 = **26 AND/比较**（`lss_and_gates_per_compare`，
  与设计文档 §3.3 的 26 triple 一致）。

key/通信/轮次（每次比较，D=16）：key = 16 OT16 + 26 BIT_TRIPLE；
通信 = sender→receiver 16×32 bit（OT 回复）+ 26×2 bit（AND open），
receiver→sender 16×4 bit（δ）+ 26×2 bit；轮次 = 1 + 4 = **5**。

## 6. MSB / DReLU（wrap 推导）

x = x0 + x1 mod 2^bw。写 x_i = m_i·2^(bw−1) + b_i（m_i = msb(x_i)，
b_i = 低 bw−1 位）。设 c = 1{b0 + b1 ≥ 2^(bw−1)}（低位相加的进位），则

    x mod 2^bw = ((m0 + m1 + c) mod 2)·2^(bw−1) + (b0 + b1 mod 2^(bw−1))
    ⟹  msb(x) = m0 ⊕ m1 ⊕ c

c 由比较得到：c = 1{b0 ≥ 2^(bw−1) − b1} = 1{b0 > (2^(bw−1)−1) − b1}。
即：party1 本地计算 v1 = (2^(bw−1)−1) − b1（mod 2^(bw−1)），双方做
(bw−1)-bit compare_gt(b0, v1)，各方再本地 XOR 上自己 share 的 msb。
（与 SCI `AuxProtocols::MSB`，aux-protocols.cpp:175-198 完全一致。）

DReLU(x) = 1{x ≥ 0} = ¬msb(x)：布尔份额本地取反（仅 party0 翻转），
零通信零轮次。

## 7. ReLU

ReLU(x) = MUX(DReLU(x), x)：x ≥ 0（有符号）输出 x，否则输出 0。
= MSB（内含 1 次 63-bit 比较）+ 本地取反 + 1 次 MUX。

**每元素核算（bw=64，16 OT16 + 26 triple + 1 MUX，实测与理论一致）**：

| 项 | party0（OT sender） | party1（OT receiver） |
|---|---|---|
| key/元素 | 16×35 + 26×6 + 196 bit = **114.0 B** | 16×9 + 26×6 + 196 bit = **62.0 B** |
| 在线通信/元素 | 发送 70.5 B（不含结果 open） | 发送 14.5 B |
| 轮次 | 6（叶子 OT 1 + AND 树 4 + MUX 1） | 同左 |

⚠️ key 显式存储未达设计文档 §6 的 ~16–20 B/元素目标：OT16 pad（70 B）
与 MUX 的 b/c 份额（16 B）均为 dealer PRG 输出，可种子化压缩
（设计文档 Phase 2 的 PRF 压缩项）；显式残量 ≈ 22 B。

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

算子级 key 序列（`gen_compare` / `gen_relu`，顺序 = 在线消费顺序）：
- compare(bitlength)：先 n×D 条 OT16（叶子一批消费），再 n×gates 条
  BIT_TRIPLE（AND 树逐层消费）；
- relu(bw)：compare(bw−1) 的记录 + n 条 MUX_TRIPLE。

## PRG 说明

dealer 用 `std::mt19937_64`（种子由参数给定，确定性可复现）。
统计上均匀独立，对"dealer 直给"模型的半诚实安全性足够；
**生产环境建议换 AES-CTR**（TODO，见汇报遗留问题）。
