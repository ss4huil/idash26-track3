// lss_keygen.cpp — LSS dealer key 生成器实现（LSS2 种子压缩格式）
//
// 生成纪律（与 eval 侧严格对偶）：每类记录维护流内序号 ctr_，dealer 先按
// 序号 PRF 再生出双方能再生的份额，再反解出需要显式存储的修正值写盘。
#include "lss_keygen.h"

namespace lss {

LssKeygen::LssKeygen(uint64_t seed) : rng_(seed) {
    // master seed → 2×7 条流种子（确定性；两个 dealer 进程一致），
    // 每条流种子经 KDF 预展开成 AES-128 轮密钥
    lss_require_aesni();
    for (int p = 0; p < 2; p++)
        for (size_t t = 0; t < LSS_NUM_SEEDS; t++) {
            seeds_[p][t] = rng_();
            keys_[p][t] = lss_stream_key(seeds_[p][t]);
        }
}

void LssKeygen::gen_bit_triples(uint64_t count) {
    for (uint64_t k = 0; k < count; k++) {
        uint64_t i = ctr_[REC_BIT_TRIPLE] + k;
        uint64_t w0 = prf(0, REC_BIT_TRIPLE, i, 0);
        uint64_t w1 = prf(1, REC_BIT_TRIPLE, i, 0);
        uint8_t a0 = w0 & 1, b0 = (w0 >> 1) & 1;
        uint8_t a1 = w1 & 1, b1 = (w1 >> 1) & 1, c1 = (w1 >> 2) & 1;
        uint8_t c0 = uint8_t(((a0 ^ a1) & (b0 ^ b1)) ^ c1);
        w_[0].put(REC_BIT_TRIPLE, 3);
        w_[0].put(c0, 1);
        w_[1].put(REC_BIT_TRIPLE, 3);
    }
    ctr_[REC_BIT_TRIPLE] += count;
    nrecs_ += count;
}

void LssKeygen::gen_ot16(uint64_t count, int sender_party) {
    if (sender_party != 0 && sender_party != 1)
        throw std::runtime_error("lss: gen_ot16 sender_party 必须是 0 或 1");
    int recv_party = 1 - sender_party;
    for (uint64_t k = 0; k < count; k++) {
        // OT16 SEND/RECV 共用一条逻辑流（lss_keys.h stream_slot）
        uint64_t i = ctr_[REC_OT16_SEND] + k;
        uint32_t pads = (uint32_t)prf(sender_party, REC_OT16_SEND, i, 0);
        uint8_t r = (uint8_t)(prf(recv_party, REC_OT16_RECV, i, 0) & 15);
        uint8_t kr = (uint8_t)((pads >> (2 * r)) & 3);
        w_[sender_party].put(REC_OT16_SEND, 3);
        w_[recv_party].put(REC_OT16_RECV, 3);
        w_[recv_party].put(kr, 2);
    }
    ctr_[REC_OT16_SEND] += count;
    nrecs_ += count;
}

void LssKeygen::gen_mux(uint64_t count) {
    for (uint64_t k = 0; k < count; k++) {
        uint64_t i = ctr_[REC_MUX_TRIPLE] + k;
        uint64_t w00 = prf(0, REC_MUX_TRIPLE, i, 0);
        uint64_t bb0 = prf(0, REC_MUX_TRIPLE, i, 1);
        uint64_t w10 = prf(1, REC_MUX_TRIPLE, i, 0);
        uint64_t bb1 = prf(1, REC_MUX_TRIPLE, i, 1);
        uint64_t cc1 = prf(1, REC_MUX_TRIPLE, i, 2);
        uint8_t ab0 = w00 & 1, ab1 = w10 & 1;
        uint64_t aa0 = (w00 >> 1) & 1;
        uint64_t a = ab0 ^ ab1;   // a ∈ {0,1}
        uint64_t b = bb0 + bb1;
        uint64_t c = a * b;
        uint64_t cc0 = c - cc1;
        // aa1 = a − aa0 (mod 2^64) ∈ {0, 1, 2^64−1}，2 bit 编码存 P1
        uint64_t aa1 = a - aa0;
        uint64_t enc = (aa1 == 0) ? 0 : (aa1 == 1 ? 1 : 2);
        w_[0].put(REC_MUX_TRIPLE, 3);
        w_[0].put(cc0, 64);
        w_[1].put(REC_MUX_TRIPLE, 3);
        w_[1].put(enc, 2);
    }
    ctr_[REC_MUX_TRIPLE] += count;
    nrecs_ += count;
}

void LssKeygen::gen_b2a(uint64_t count) {
    for (uint64_t k = 0; k < count; k++) {
        uint64_t i = ctr_[REC_B2A_CORR] + k;
        uint8_t rb0 = (uint8_t)(prf(0, REC_B2A_CORR, i, 0) & 1);
        uint8_t rb1 = (uint8_t)(prf(1, REC_B2A_CORR, i, 0) & 1);
        uint64_t ra0 = prf(0, REC_B2A_CORR, i, 1);
        uint64_t ra1 = (uint64_t)(rb0 ^ rb1) - ra0;
        w_[0].put(REC_B2A_CORR, 3);
        w_[1].put(REC_B2A_CORR, 3);
        w_[1].put(ra1, 64);
    }
    ctr_[REC_B2A_CORR] += count;
    nrecs_ += count;
}

// ── 算子级入口（P2）────────────────────────────────────────────────
void LssKeygen::gen_compare(uint64_t count, int bitlength, int sender_party) {
    if (bitlength < 1 || bitlength > 64)
        throw std::runtime_error("lss: gen_compare bitlength 须在 [1,64]");
    int D = lss_num_digits(bitlength);
    gen_ot16(count * (uint64_t)D, sender_party);
    gen_bit_triples(count * lss_and_gates_per_compare(D));
}

void LssKeygen::gen_relu(uint64_t count, int bw, int sender_party) {
    gen_compare(count, bw - 1, sender_party);
    gen_mux(count);
}

// ── P3 桥接 ─────────────────────────────────────────────────────────
void LssKeygen::gen_mask_records(RecordType type, const uint64_t *values,
                                 uint64_t n) {
    if (type != REC_MASK_IN)
        throw std::runtime_error(
            "lss: gen_mask_records 仅支持 REC_MASK_IN（外部给定 mask，P0 份额 "
            "PRF 再生 + P1 显式）；REC_MASK_OUT 由 gen_relu_bridged 内部以"
            "双方 PRF 再生、零显式字节生成");
    for (uint64_t k = 0; k < n; k++) {
        uint64_t i = ctr_[REC_MASK_IN] + k;
        uint64_t s0 = prf(0, REC_MASK_IN, i, 0);
        w_[0].put(REC_MASK_IN, 3);
        w_[1].put(REC_MASK_IN, 3);
        w_[1].put(values[k] - s0, 64);
    }
    ctr_[REC_MASK_IN] += n;
    nrecs_ += n;
}

void LssKeygen::gen_relu_bridged(uint64_t n, int bw, const uint64_t *r_in,
                                 uint64_t *r_out, int sender_party) {
    gen_mask_records(REC_MASK_IN, r_in, n);
    gen_compare(n, bw - 1, sender_party);
    gen_mux(n);
    // 输出 mask r' 定义为双方 PRF 份额之和：r' = s0 + s1，零显式字节
    for (uint64_t k = 0; k < n; k++) {
        uint64_t i = ctr_[REC_MASK_OUT] + k;
        uint64_t s0 = prf(0, REC_MASK_OUT, i, 0);
        uint64_t s1 = prf(1, REC_MASK_OUT, i, 0);
        r_out[k] = s0 + s1;
        w_[0].put(REC_MASK_OUT, 3);
        w_[1].put(REC_MASK_OUT, 3);
    }
    ctr_[REC_MASK_OUT] += n;
    nrecs_ += n;
}

void LssKeygen::write_files(const std::string &path0, const std::string &path1) const {
    write_file(path0, 0);
    write_file(path1, 1);
}

void LssKeygen::write_file(const std::string &path, int party) const {
    if (party != 0 && party != 1)
        throw std::runtime_error("lss: write_file party 必须是 0 或 1");
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("lss: 无法写 key 文件 " + path);
    LssHeader h;
    h.party = (uint32_t)party;
    h.num_records = nrecs_;
    h.payload_bits = w_[party].nbits;
    memcpy(h.seeds, seeds_[party], sizeof(h.seeds));
    write_header(f, h);
    const BitWriter &w = w_[party];
    if (!w.buf.empty() &&
        fwrite(w.buf.data(), 1, w.buf.size(), f) != w.buf.size())
        throw std::runtime_error("lss: 写 payload 失败 " + path);
    write_trailer(f, nrecs_, w.payload_bytes(),
                  fnv1a64(w.buf.data(), w.buf.size()));
    fclose(f);
}

} // namespace lss
