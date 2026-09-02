// lss_keygen.h — LSS dealer key 生成器（gpu-mpc-track v2 Phase 1）
//
// 安全假设（与比赛 Q3 一致、设计文档 §4.3）：
//   dealer 半诚实且不与任一方合谋；在线阶段无 dealer。
//   本生成器产出的是 random-OT 相关随机数 / Beaver 三元组份额，
//   与在线输入完全无关，故可全部离线预生成。
//
// 用法：按图序依次调用 gen_*（顺序即 key 文件记录顺序，双方严格同序），
// 最后 write_files() 输出 lss_keys_party{0,1}.bin。
#pragma once

#include "lss_keys.h"
#include <random>
#include <string>

namespace lss {

class LssKeygen {
public:
    // seed 给定后输出完全确定、可复现（keygen/eval 对齐调试依赖这一点）。
    // PRG 用 std::mt19937_64：统计均匀独立，对 dealer 直给模型足够；
    // TODO(production): 换 AES-CTR（见 lss_protocol.md PRG 说明）。
    explicit LssKeygen(uint64_t seed);

    // 布尔 Beaver 三元组：a,b ← {0,1}, c = a∧b，双方各得 XOR 份额 (a_i,b_i,c_i)。
    void gen_bit_triples(uint64_t count);

    // 1oo16 OT 相关性：r ← [0,16)，k_0..k_15 ← {0..3}。
    // sender_party 方得 16 个 pad（REC_OT16_SEND），
    // 另一方得 (r, k_r)（REC_OT16_RECV）。
    void gen_ot16(uint64_t count, int sender_party);

    // MUX 三元组：a ← {0,1}, b ← Z_2^64, c = a·b；
    // 双方得 a 的布尔份额 + a 的算术份额 + b 份额 + c 份额。
    void gen_mux(uint64_t count);

    // B2A 相关性：r ← {0,1}，双方得 r 的布尔份额 + 算术份额。
    void gen_b2a(uint64_t count);

    // ── 算子级入口（P2）：记录顺序 = 在线消费顺序，严格同序 ──────────
    // 一次 bitlength 位 Millionaires 比较（1{x>y} 布尔份额）所需记录：
    //   count×num_digits 条 OT16（叶子一批）+ count×and_gates 条 BIT_TRIPLE
    //   （AND 树逐层消费；两者内部均按记录流顺序，与在线游标一致）。
    void gen_compare(uint64_t count, int bitlength, int sender_party = 0);

    // 一次 bw 位 ReLU（MSB(63-bit 比较) + MUX）所需记录：
    //   gen_compare(count, bw−1) + count 条 MUX_TRIPLE。
    void gen_relu(uint64_t count, int bw = 64, int sender_party = 0);

    // 写出双份 key 文件（含 header/trailer 校验）。
    void write_files(const std::string &path0, const std::string &path1) const;

    uint64_t num_records() const { return nrecs_; }
    uint64_t payload_bits(int party) const { return w_[party].nbits; }

private:
    std::mt19937_64 rng_;
    BitWriter w_[2];
    uint64_t nrecs_ = 0;

    uint64_t rand64() { return rng_(); }
    uint8_t rand_bit();           // 从 bit 池取 1 bit
    uint8_t rand_bits(unsigned n); // n ≤ 8，取低 n 位

    uint64_t bit_pool_ = 0;
    unsigned pool_left_ = 0;
};

} // namespace lss
