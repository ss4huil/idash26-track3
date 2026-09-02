// lss_keygen.h — LSS dealer key 生成器（gpu-mpc-track v2，LSS2 种子压缩格式）
//
// 安全假设（与比赛 Q3 一致、设计文档 §4.3）：
//   dealer 半诚实且不与任一方合谋；在线阶段无 dealer。
//   本生成器产出的是 random-OT 相关随机数 / Beaver 三元组份额，
//   与在线输入完全无关，故可全部离线预生成。
//
// LSS2（lss_protocol.md §9）：master seed 派生 2×7 条计数器模式 PRF 流
// （每方每记录类型一条），记录里纯随机的份额全部由 PRF 再生、不落盘；
// 文件只存相关性修正（如 BIT_TRIPLE 的 c0、OT16_RECV 的 k_r、MUX 的 cc0
// 等）。dealer 生成顺序 = 先 PRF 再生出各方份额，再反解显式修正值。
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
    // seed 给定后输出完全确定、可复现（keygen/eval 对齐调试依赖这一点；
    // 两个 dealer 进程用同一 master seed ⇒ 派生流种子与记录流一致）。
    // PRF 用 splitmix64 计数器模式：见 lss_keys.h 的 TODO(production)，
    // 最终提交需换 AES-CTR 以满足 128-bit 安全声明。
    explicit LssKeygen(uint64_t seed);

    // 布尔 Beaver 三元组：a_i,b_i 双方 PRF 再生，c₁ 由 P1 流再生，
    // c₀=(a∧b)⊕c₁ 显式存 P0（1 bit）。⇒ P0 4 bit/条，P1 3 bit/条。
    void gen_bit_triples(uint64_t count);

    // 1oo16 OT 相关性：sender 方 16 个 pad 全 PRF 再生（tag only）；
    // receiver 方 r 由本方流再生，k_r 显式存（2 bit）。
    void gen_ot16(uint64_t count, int sender_party);

    // MUX 三元组：ab、aa0、bb 双方 PRF 再生；cc₁ 由 P1 流再生，
    // cc₀=c−cc₁ 显式存 P0（64 bit）；aa₁=a−aa0∈{0,1,2⁶⁴−1} 编码 2 bit 存 P1。
    void gen_mux(uint64_t count);

    // B2A 相关性：rb 双方 PRF 再生；ra0 由 P0 流再生，ra1=r−ra0 显式存 P1。
    void gen_b2a(uint64_t count);

    // ── 算子级入口（P2）：记录顺序 = 在线消费顺序，严格同序 ──────────
    // 一次 bitlength 位 Millionaires 比较（1{x>y} 布尔份额）所需记录：
    //   count×num_digits 条 OT16（叶子一批）+ count×and_gates 条 BIT_TRIPLE
    //   （AND 树逐层消费；两者内部均按记录流顺序，与在线游标一致）。
    void gen_compare(uint64_t count, int bitlength, int sender_party = 0);

    // 一次 bw 位 ReLU（MSB(63-bit 比较) + MUX）所需记录：
    //   gen_compare(count, bw−1) + count 条 MUX_TRIPLE。
    void gen_relu(uint64_t count, int bw = 64, int sender_party = 0);

    // ── GPU 管线桥接版 ReLU（P3，设计文档 §5.2 最小侵入桥接）─────────
    // dealer 在 keygen forward 中调用：r_in 为本算子输入张量的 mask
    // （keygen 模式下 in.d_data 跟踪的正是 mask；两个 dealer 进程随机流
    // 确定性一致，故 r_in 相同）。记录顺序 = eval 消费顺序：
    //   n × MASK_IN → compare 记录 → n × MUX → n × MASK_OUT。
    // MASK_IN：r_in 由管线给定不可压缩——P0 份额 PRF 再生，P1 显式存
    //   r_in−r0（64 bit）。
    // MASK_OUT：r'_out 由 dealer 定义为双方 PRF 份额之和（s0+s1），零显式
    //   字节；本函数把 r_out 填出（dealer 把它作为输出张量的 mask 继续跟踪）。
    void gen_relu_bridged(uint64_t n, int bw, const uint64_t *r_in,
                          uint64_t *r_out, int sender_party = 0);

    // 桥接入口 mask 记录（REC_MASK_IN；P0 份额 PRF 再生，P1 显式）。
    // REC_MASK_OUT 由 gen_relu_bridged 内部以零显式字节形式生成，本入口
    // 拒绝（value 外部给定时无法双方都压缩）。
    void gen_mask_records(RecordType type, const uint64_t *values, uint64_t n);

    // 写出双份 key 文件（含 header/trailer 校验）。
    void write_files(const std::string &path0, const std::string &path1) const;
    // 单方写出：两个 dealer 进程各自只写自己的 key 文件（P3 集成用；
    // 两进程相同 seed ⇒ 记录流一致）。
    void write_file(const std::string &path, int party) const;

    uint64_t num_records() const { return nrecs_; }
    uint64_t payload_bits(int party) const { return w_[party].nbits; }

private:
    std::mt19937_64 rng_;               // 仅用于构造时派生各流种子
    uint64_t seeds_[2][LSS_NUM_SEEDS];  // [party][记录类型] PRF 流种子
    uint64_t ctr_[LSS_NUM_SEEDS] = {};  // 每类型已生成记录数（PRF 索引）
    BitWriter w_[2];
    uint64_t nrecs_ = 0;

    // party p 的类型 slot 流中第 idx 条记录的第 word 个 PRF 输出字
    uint64_t prf(int p, uint8_t slot, uint64_t idx, unsigned word) const {
        return lss_prf(seeds_[p][slot], idx * LSS_PRF_STRIDE + word);
    }
};

} // namespace lss
