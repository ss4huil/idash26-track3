// lss_compare.cpp — Millionaires 比较 / MSB / DReLU / ReLU（P2 算子层）
//
// 公式与正确性证明见 lss_protocol.md §5–§7。语义参考 SCI
// src/Millionaire/millionaire.h 与 BuildingBlocks/aux-protocols.cpp 的 MSB，
// 但全部干净重写：叶子 OT 与 AND 三元组全部来自 P1 的 dealer key 流，
// 无 OT extension。party0 恒为 OT sender（SCI 的 ALICE 约定）。
//
// 轮次结构（bitlength 位，D=ceil(bitlength/4) 个 digit）：
//   叶子 OT16：全部 n×D 个一批，1 轮（1 次 RT）；
//   AND 树：ceil(log2(D)) 层，每层一次 batched and_open，1 轮/层；
//   ReLU 再 +1 轮 MUX。bw=64 的 ReLU 共 6 轮。
#include "lss_online.h"

namespace lss {

// ── Millionaires 比较：out = 1{a > b} 的布尔份额 ────────────────────
// val 为本方输入（party0: a；party1: b）。digit 从 LSB 到 MSB 编号
// 0..D−1。叶子 digit i 的 OT16 产出布尔份额对：
//   cmp_i = 1{a 的 digit_i > b 的 digit_i}（payload 高 bit）
//   eq_i  = 1{两 digit 相等}（payload 低 bit；digit 0 的 eq 不使用，
//           与 SCI 一致——严格大于的组合里 LSB 的 eq 无意义）
// AND 树（stride i = 1,2,4,...；同层全部 AND 一批 1 轮）：
//   j==0 组：cmp[j] ← (cmp[j] ∧ eq[j+i]) ⊕ cmp[j+i]
//   j>0  组：eq[j]  ← eq[j] ∧ eq[j+i]；
//            cmp[j] ← (cmp[j] ∧ eq[j+i]) ⊕ cmp[j+i]
// 不变式：第 j 组覆盖 digit [j, j+i)，cmp[j]=1{该段 a > 该段 b}，
// eq[j]=1{该段相等}；最终 cmp[0] = 1{a>b}。
void LssParty::compare_gt(uint8_t *out, const uint64_t *val, size_t n,
                          int bitlength) {
    if (bitlength < 1 || bitlength > 64)
        throw std::runtime_error("lss: compare_gt bitlength 须在 [1,64]");
    const int D = lss_num_digits(bitlength);
    const int r = bitlength % 4; // 顶层 digit 的有效 bit 数（0 = 满 4 bit）
    const size_t m = n * (size_t)D;

    std::vector<uint8_t> cmp(m), eq(m);

    // ── 叶子层：一批 OT16（1 轮）──
    if (party == 0) {
        // sender：g(i) = ((t>i)<<1)|(t==i)，t 为本方 digit；
        // 掩码 s 来自本地 PRNG，同时即本方的 (cmp,eq) 份额。
        std::vector<uint8_t> g(m * 16), s(m);
        for (size_t e = 0; e < m; e++) {
            size_t j = e / (size_t)D;
            int i = (int)(e % (size_t)D);
            uint8_t t;
            if (i == D - 1 && r != 0)
                t = (uint8_t)(val[j] >> (4 * i)) & ((1u << r) - 1);
            else
                t = (uint8_t)(val[j] >> (4 * i)) & 15;
            for (int k = 0; k < 16; k++)
                g[e * 16 + k] = (uint8_t)(((t > k) << 1) | (t == k));
            s[e] = (uint8_t)(local_rng() & 3);
            cmp[e] = (s[e] >> 1) & 1;
            eq[e] = s[e] & 1;
        }
        ot16_send(g.data(), s.data(), m);
    } else {
        // receiver：choice = 本方 digit
        std::vector<uint8_t> d(m), o(m);
        for (size_t e = 0; e < m; e++) {
            size_t j = e / (size_t)D;
            int i = (int)(e % (size_t)D);
            if (i == D - 1 && r != 0)
                d[e] = (uint8_t)(val[j] >> (4 * i)) & ((1u << r) - 1);
            else
                d[e] = (uint8_t)(val[j] >> (4 * i)) & 15;
        }
        ot16_recv(d.data(), o.data(), m);
        for (size_t e = 0; e < m; e++) {
            cmp[e] = (o[e] >> 1) & 1;
            eq[e] = o[e] & 1;
        }
    }

    // ── AND 树：逐层一批（每层 1 轮）──
    // 注意布局与叶子层一致：元素主序，cmp/eq 的下标 = k*D + digit。
    for (int i = 1; i < D; i <<= 1) {
        // 先收集本层全部 AND 的输入（读旧状态），再统一写回
        std::vector<uint8_t> xs, ys;
        std::vector<int> dst_j;   // 每个 AND 对应的组下标 j
        std::vector<bool> is_eq;  // 该 AND 是 eq 更新还是 cmp 更新
        for (int j = 0; j + i < D; j += 2 * i) {
            for (size_t k = 0; k < n; k++) {
                xs.push_back(cmp[k * D + j]);
                ys.push_back(eq[k * D + j + i]);
            }
            dst_j.push_back(j);
            is_eq.push_back(false);
            if (j > 0) {
                for (size_t k = 0; k < n; k++) {
                    xs.push_back(eq[k * D + j]);
                    ys.push_back(eq[k * D + j + i]);
                }
                dst_j.push_back(j);
                is_eq.push_back(true);
            }
        }
        std::vector<uint8_t> zs(xs.size());
        and_open(xs.data(), ys.data(), zs.data(), xs.size());
        // 写回：顺序与收集顺序一致
        size_t pos = 0;
        for (size_t gidx = 0; gidx < dst_j.size(); gidx++) {
            int j = dst_j[gidx];
            if (!is_eq[gidx]) {
                for (size_t k = 0; k < n; k++)
                    cmp[k * D + j] = zs[pos + k] ^ cmp[k * D + j + i];
            } else {
                for (size_t k = 0; k < n; k++)
                    eq[k * D + j] = zs[pos + k];
            }
            pos += n;
        }
    }

    for (size_t k = 0; k < n; k++) out[k] = cmp[k * D];
}

// ── MSB ─────────────────────────────────────────────────────────────
// x = x0 + x1 mod 2^bw。写 x_i = m_i·2^(bw−1) + b_i（m_i = msb(x_i)，
// b_i = 低 bw−1 位）。设 c = 1{b0 + b1 ≥ 2^(bw−1)}（低位的进位/wrap），则
//   x mod 2^bw = ((m0 + m1 + c) mod 2)·2^(bw−1) + (b0+b1 mod 2^(bw−1))
//   ⟹ msb(x) = m0 ⊕ m1 ⊕ c。
// 而 c = 1{b0 ≥ 2^(bw−1) − b1} = 1{b0 > (2^(bw−1)−1) − b1}，
// 即 party1 本地把 b1 换成 mask−b1 后做 (bw−1)-bit compare_gt。
// （与 SCI AuxProtocols::MSB 一致，aux-protocols.cpp:175-198。）
void LssParty::msb(const uint64_t *x, uint8_t *out, size_t n, int bw) {
    if (bw < 2 || bw > 64)
        throw std::runtime_error("lss: msb bw 须在 [2,64]");
    const int shift = bw - 1;
    const uint64_t mask = (shift == 64) ? ~0ULL : ((1ULL << shift) - 1);
    std::vector<uint64_t> v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = x[i] & mask;
        if (party == 1) v[i] = (mask - v[i]) & mask;
    }
    compare_gt(out, v.data(), n, bw - 1);
    for (size_t i = 0; i < n; i++)
        out[i] ^= (uint8_t)((x[i] >> shift) & 1);
}

// ── DReLU = ¬MSB（布尔份额的本地翻转：仅 party0 翻）────────────────
void LssParty::drelu(const uint64_t *x, uint8_t *out, size_t n, int bw) {
    msb(x, out, n, bw);
    if (party == 0)
        for (size_t i = 0; i < n; i++) out[i] ^= 1;
}

// ── ReLU = MUX(DReLU, x) ────────────────────────────────────────────
void LssParty::relu(const uint64_t *x, uint64_t *y, size_t n, int bw) {
    std::vector<uint8_t> d(n);
    drelu(x, d.data(), n, bw);
    mux(d.data(), x, y, n);
}

// ── P3 桥接版 ReLU（masked-public ↔ share 转换 + 重构出口）───────────
void LssParty::relu_bridged(const uint64_t *h_m, uint64_t *h_out, size_t n,
                            int bw) {
    if (keys.exhausted()) keys.rewind(); // benchmark 迭代复用（见头注释）

    // 入口：m → x 的算术份额
    std::vector<uint64_t> x(n);
    for (size_t i = 0; i < n; i++) {
        keys.expect(REC_MASK_IN);
        uint64_t rp = keys.reader.get(64);
        x[i] = (party == 0) ? (h_m[i] - rp) : (0ULL - rp);
    }
    // LSS relu
    std::vector<uint64_t> y(n);
    relu(x.data(), y.data(), n, bw);
    // 出口：加新 mask 份额并重构（双方交换 w_p，各得 m' = relu(x) + r'）
    std::vector<uint64_t> w(n), wp(n);
    for (size_t i = 0; i < n; i++) {
        keys.expect(REC_MASK_OUT);
        uint64_t rp = keys.reader.get(64);
        w[i] = y[i] + rp;
    }
    chan.exchange(w.data(), wp.data(), n * 8);
    for (size_t i = 0; i < n; i++) h_out[i] = w[i] + wp[i];
}

} // namespace lss
