// lss_keygen.cpp — LSS dealer key 生成器实现
#include "lss_keygen.h"

namespace lss {

LssKeygen::LssKeygen(uint64_t seed) : rng_(seed) {}

uint8_t LssKeygen::rand_bit() {
    if (pool_left_ == 0) {
        bit_pool_ = rng_();
        pool_left_ = 64;
    }
    uint8_t b = bit_pool_ & 1;
    bit_pool_ >>= 1;
    pool_left_--;
    return b;
}

uint8_t LssKeygen::rand_bits(unsigned n) {
    uint8_t v = 0;
    for (unsigned k = 0; k < n; k++) v |= uint8_t(rand_bit() << k);
    return v;
}

void LssKeygen::gen_bit_triples(uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        uint8_t a = rand_bit(), b = rand_bit();
        uint8_t c = a & b;
        uint8_t a0 = rand_bit(), b0 = rand_bit(), c0 = rand_bit();
        w_[0].put(REC_BIT_TRIPLE, 3);
        w_[0].put(a0, 1);
        w_[0].put(b0, 1);
        w_[0].put(c0, 1);
        w_[1].put(REC_BIT_TRIPLE, 3);
        w_[1].put(a ^ a0, 1);
        w_[1].put(b ^ b0, 1);
        w_[1].put(c ^ c0, 1);
    }
    nrecs_ += count;
}

void LssKeygen::gen_ot16(uint64_t count, int sender_party) {
    if (sender_party != 0 && sender_party != 1)
        throw std::runtime_error("lss: gen_ot16 sender_party 必须是 0 或 1");
    int recv_party = 1 - sender_party;
    for (uint64_t i = 0; i < count; i++) {
        uint8_t r = rand_bits(4);
        uint8_t k[16];
        for (int j = 0; j < 16; j++) k[j] = rand_bits(2);
        w_[sender_party].put(REC_OT16_SEND, 3);
        for (int j = 0; j < 16; j++) w_[sender_party].put(k[j], 2);
        w_[recv_party].put(REC_OT16_RECV, 3);
        w_[recv_party].put(r, 4);
        w_[recv_party].put(k[r], 2);
    }
    nrecs_ += count;
}

void LssKeygen::gen_mux(uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        uint64_t a = rand_bit();
        uint64_t b = rand64();
        uint64_t c = a * b;
        // a 的布尔份额
        uint8_t ab0 = rand_bit();
        // a 的算术份额：aa0 ∈ {0,1}，aa1 = a − aa0 (mod 2^64) ∈ {0,1,2^64−1}
        uint64_t aa0 = rand_bit();
        uint64_t b0 = rand64();
        uint64_t c0 = rand64();
        w_[0].put(REC_MUX_TRIPLE, 3);
        w_[0].put(ab0, 1);
        w_[0].put(aa0, 64);
        w_[0].put(b0, 64);
        w_[0].put(c0, 64);
        w_[1].put(REC_MUX_TRIPLE, 3);
        w_[1].put(ab0 ^ (uint8_t)a, 1);
        w_[1].put(a - aa0, 64);
        w_[1].put(b - b0, 64);
        w_[1].put(c - c0, 64);
    }
    nrecs_ += count;
}

void LssKeygen::gen_b2a(uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        uint8_t r = rand_bit();
        uint8_t rb0 = rand_bit();
        uint64_t ra0 = rand64();
        for (int p = 0; p < 2; p++) {
            w_[p].put(REC_B2A_CORR, 3);
            if (p == 0) {
                w_[p].put(rb0, 1);
                w_[p].put(ra0, 64);
            } else {
                w_[p].put(rb0 ^ r, 1);
                w_[p].put(uint64_t(r) - ra0, 64);
            }
        }
    }
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
    // values 为完整 mask；party0 份额随机采样，party1 = values − share0。
    for (uint64_t i = 0; i < n; i++) {
        uint64_t s0 = rand64();
        w_[0].put(type, 3);
        w_[0].put(s0, 64);
        w_[1].put(type, 3);
        w_[1].put(values[i] - s0, 64);
    }
    nrecs_ += n;
}

void LssKeygen::gen_relu_bridged(uint64_t n, int bw, const uint64_t *r_in,
                                 uint64_t *r_out, int sender_party) {
    gen_mask_records(REC_MASK_IN, r_in, n);
    gen_compare(n, bw - 1, sender_party);
    gen_mux(n);
    // 采样输出 mask r' 并写份额
    std::vector<uint64_t> rout(n);
    for (uint64_t i = 0; i < n; i++) rout[i] = rand64();
    gen_mask_records(REC_MASK_OUT, rout.data(), n);
    for (uint64_t i = 0; i < n; i++) r_out[i] = rout[i];
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
