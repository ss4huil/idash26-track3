// lss_online.cpp — LSS 在线原语实现（公式见 lss_protocol.md）
//
// 并行化（v2 P3 性能项）：所有元素级循环用 OpenMP 并行（_OPENMP 宏守卫，
// 无 OpenMP 时退化为串行，语义不变）。key 记录定长 ⇒ 用 BitReader::get_at
// 随机只读访问并行读记录，读完后一次性推进游标；通信（open/exchange）
// 仍单线程集中做——轮次结构与串行版完全一致。线程数：DDG_LSS_THREADS 覆盖，
// 默认 hardware_concurrency。
#include "lss_online.h"
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lss {

// ── 原语 1：AND ─────────────────────────────────────────────────────
// e = open(x ⊕ a), f = open(y ⊕ b)
// z_i = c_i ⊕ (f∧a_i) ⊕ (e∧b_i) ⊕ (e∧f∧[i==0])
// LSS2：a_i,b_i 双方 PRF 再生；c₁ 由 P1 流再生，c₀ 显式存 P0（1 bit）。
void LssParty::and_open(const uint8_t *x, const uint8_t *y, uint8_t *z,
                        size_t n) {
    // BIT_TRIPLE 记录：P0 = 3(tag)+1(c0) = 4 bit/条；P1 = 3 bit/条
    const unsigned rec_bits = (party == 0) ? 4 : 3;
    const uint64_t base = keys.reader.pos;
    const uint64_t idx0 = keys.stream_pos(REC_BIT_TRIPLE);
    check_record_tags(keys.reader, base, rec_bits, REC_BIT_TRIPLE, n);

    std::vector<uint8_t> ef(2 * n), abc(n);
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t i = 0; i < n; i++) {
        uint64_t w = keys.prf(REC_BIT_TRIPLE, idx0 + i, 0);
        uint8_t a = (uint8_t)(w & 1);
        uint8_t b = (uint8_t)((w >> 1) & 1);
        uint8_t c = (party == 0)
                        ? (uint8_t)keys.reader.get_at(base + 3 + i * 4, 1)
                        : (uint8_t)((w >> 2) & 1);
        ef[2 * i] = x[i] ^ a;
        ef[2 * i + 1] = y[i] ^ b;
        abc[i] = (a << 2) | (b << 1) | c;
    }
    keys.reader.pos = base + (uint64_t)rec_bits * n;
    keys.stream_advance(REC_BIT_TRIPLE, n);

    std::vector<uint8_t> snd((2 * n + 7) / 8), rcv(snd.size());
    pack_small(snd.data(), ef.data(), 2 * n, 1);
    chan.exchange(snd.data(), rcv.data(), snd.size());
    std::vector<uint8_t> ef_peer(2 * n);
    unpack_small(rcv.data(), ef_peer.data(), 2 * n, 1);

    const uint8_t me = (party == 0) ? 1 : 0;
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t i = 0; i < n; i++) {
        uint8_t a = abc[i] >> 2, b = (abc[i] >> 1) & 1, c = abc[i] & 1;
        uint8_t e = ef_peer[2 * i] ^ ef[2 * i];
        uint8_t f = ef_peer[2 * i + 1] ^ ef[2 * i + 1];
        z[i] = c ^ (f & a) ^ (e & b) ^ (e & f & me);
    }
}

// ── 原语 2：1oo16 OT ────────────────────────────────────────────────
// receiver → sender: δ = (r − d) mod 16（4 bit）
// sender → receiver: c_i = f(i) ⊕ k_{(i+δ) mod 16}，f(i) = g(i) ⊕ s
// receiver 输出 c_d ⊕ k_r。下标：d+δ ≡ r (mod 16) ⇒ c_d 用 pad k_r。✓
void LssParty::ot16_send(const uint8_t *g, const uint8_t *s, size_t n) {
    // 收 δ（4 bit/元素，打包）
    size_t dbytes = (n * 4 + 7) / 8;
    std::vector<uint8_t> dbuf(dbytes);
    chan.recv_all(dbuf.data(), dbytes);
    std::vector<uint8_t> deltas(n);
    unpack_small(dbuf.data(), deltas.data(), n, 4);

    // OT16_SEND 记录（LSS2）：tag only，3 bit/条；16 个 pad 由本方流再生。
    // 回复 16×2 bit = 4 B/元素
    const uint64_t base = keys.reader.pos;
    const uint64_t idx0 = keys.stream_pos(REC_OT16_SEND);
    check_record_tags(keys.reader, base, 3, REC_OT16_SEND, n);
    std::vector<uint8_t> reply(n * 4);
#ifdef _OPENMP
#pragma omp parallel for if(n > 4096)
#endif
    for (size_t e = 0; e < n; e++) {
        uint32_t pads = (uint32_t)keys.prf(REC_OT16_SEND, idx0 + e, 0);
        uint8_t sv = s[e] & 3;
        uint8_t delta = deltas[e];
        uint8_t c[16];
        for (int i = 0; i < 16; i++) {
            uint8_t k = (uint8_t)(pads >> (2 * ((i + delta) & 15))) & 3;
            c[i] = ((g[e * 16 + i] & 3) ^ sv) ^ k;
        }
        reply[e * 4 + 0] = c[0] | (c[1] << 2) | (c[2] << 4) | (c[3] << 6);
        reply[e * 4 + 1] = c[4] | (c[5] << 2) | (c[6] << 4) | (c[7] << 6);
        reply[e * 4 + 2] = c[8] | (c[9] << 2) | (c[10] << 4) | (c[11] << 6);
        reply[e * 4 + 3] = c[12] | (c[13] << 2) | (c[14] << 4) | (c[15] << 6);
    }
    keys.reader.pos = base + 3 * n;
    keys.stream_advance(REC_OT16_SEND, n);
    chan.send_all(reply.data(), reply.size());
}

void LssParty::ot16_recv(const uint8_t *d, uint8_t *out, size_t n) {
    // OT16_RECV 记录（LSS2）：3(tag)+2(k_r) = 5 bit/条；r 由本方流再生
    const uint64_t base = keys.reader.pos;
    const uint64_t idx0 = keys.stream_pos(REC_OT16_RECV);
    check_record_tags(keys.reader, base, 5, REC_OT16_RECV, n);
    std::vector<uint8_t> deltas(n), kr(n);
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t e = 0; e < n; e++) {
        uint8_t r = (uint8_t)(keys.prf(REC_OT16_RECV, idx0 + e, 0) & 15);
        kr[e] = (uint8_t)keys.reader.get_at(base + 3 + e * 5, 2);
        deltas[e] = (r - d[e]) & 15;
    }
    keys.reader.pos = base + 5 * n;
    keys.stream_advance(REC_OT16_RECV, n);

    std::vector<uint8_t> snd((n * 4 + 7) / 8);
    pack_small(snd.data(), deltas.data(), n, 4);
    chan.send_all(snd.data(), snd.size());
    // 收 16×2 bit/元素（4 B）
    std::vector<uint8_t> cbuf(n * 4);
    chan.recv_all(cbuf.data(), cbuf.size());
#ifdef _OPENMP
#pragma omp parallel for if(n > 4096)
#endif
    for (size_t e = 0; e < n; e++) {
        uint8_t de = d[e] & 15;
        uint8_t cd = (cbuf[e * 4 + (de >> 2)] >> ((de & 3) * 2)) & 3;
        out[e] = cd ^ kr[e];
    }
}

// ── 原语 3：MUX ─────────────────────────────────────────────────────
// e = open(β ⊕ a^b), f = open(x − b)
// e==0: z_i = c_i + a^a_i·f ;  e==1: z_i = x_i − c_i − a^a_i·f
// LSS2：ab/aa0/bb 双方 PRF 再生；P0 显式 cc0（64 bit），P1 显式 aa1（2 bit
// 编码：0→0, 1→1, 2→2⁶⁴−1），cc1 由 P1 流再生。
void LssParty::mux(const uint8_t *b, const uint64_t *x, uint64_t *z, size_t n) {
    // MUX_TRIPLE 记录：P0 = 3+64 = 67 bit/条；P1 = 3+2 = 5 bit/条
    const unsigned rec_bits = (party == 0) ? 67 : 5;
    const uint64_t base = keys.reader.pos;
    const uint64_t idx0 = keys.stream_pos(REC_MUX_TRIPLE);
    check_record_tags(keys.reader, base, rec_bits, REC_MUX_TRIPLE, n);

    const size_t ebytes = (n + 7) / 8;
    std::vector<uint8_t> snd(ebytes + n * 8), rcv(snd.size());
    std::vector<uint64_t> aa(n), cc(n);
    std::vector<uint8_t> ab(n);
    const int me_p = party;
#ifdef _OPENMP
#pragma omp parallel for if(n > 4096)
#endif
    for (size_t i = 0; i < n; i++) {
        uint64_t idx = idx0 + i;
        uint64_t w0 = keys.prf(REC_MUX_TRIPLE, idx, 0);
        ab[i] = (uint8_t)(w0 & 1);
        uint64_t bb = keys.prf(REC_MUX_TRIPLE, idx, 1);
        if (me_p == 0) {
            aa[i] = (w0 >> 1) & 1;
            cc[i] = keys.reader.get_at(base + 3 + i * 67, 64);
        } else {
            uint64_t enc = keys.reader.get_at(base + 3 + i * 5, 2);
            aa[i] = (enc == 0) ? 0 : (enc == 1 ? 1 : ~0ULL); // 2 → 2^64−1
            cc[i] = keys.prf(REC_MUX_TRIPLE, idx, 2);
        }
        uint64_t f = x[i] - bb;
        memcpy(snd.data() + ebytes + i * 8, &f, 8);
    }
    keys.reader.pos = base + (uint64_t)rec_bits * n;
    keys.stream_advance(REC_MUX_TRIPLE, n);
    // e 位 = b ⊕ a^b，打包（1 bit/元素）放在 f 数组之前
    std::vector<uint8_t> ebits(n);
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t i = 0; i < n; i++) ebits[i] = b[i] ^ ab[i];
    pack_small(snd.data(), ebits.data(), n, 1);

    chan.exchange(snd.data(), rcv.data(), snd.size());
    std::vector<uint8_t> epeer(n);
    unpack_small(rcv.data(), epeer.data(), n, 1);
#ifdef _OPENMP
#pragma omp parallel for if(n > 4096)
#endif
    for (size_t i = 0; i < n; i++) {
        uint8_t e = epeer[i] ^ ebits[i];
        uint64_t f_peer, f;
        memcpy(&f_peer, rcv.data() + ebytes + i * 8, 8);
        memcpy(&f, snd.data() + ebytes + i * 8, 8);
        f += f_peer;
        if (e == 0)
            z[i] = cc[i] + aa[i] * f;
        else
            z[i] = x[i] - cc[i] - aa[i] * f;
    }
}

// ── 原语 4：B2A ─────────────────────────────────────────────────────
// e = open(β ⊕ rb)；e==0: z_i = ra_i；e==1: z_i = [i==0] − ra_i
// LSS2：rb 双方 PRF 再生；ra0 由 P0 流再生，ra1 显式存 P1（64 bit）。
void LssParty::b2a(const uint8_t *b, uint64_t *z, size_t n) {
    // B2A_CORR 记录：P0 = 3 bit/条；P1 = 3+64 = 67 bit/条
    const unsigned rec_bits = (party == 0) ? 3 : 67;
    const uint64_t base = keys.reader.pos;
    const uint64_t idx0 = keys.stream_pos(REC_B2A_CORR);
    check_record_tags(keys.reader, base, rec_bits, REC_B2A_CORR, n);

    std::vector<uint8_t> ebits(n);
    std::vector<uint64_t> ra(n);
    const int me_p = party;
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t i = 0; i < n; i++) {
        uint64_t idx = idx0 + i;
        uint8_t rb = (uint8_t)(keys.prf(REC_B2A_CORR, idx, 0) & 1);
        ra[i] = (me_p == 0) ? keys.prf(REC_B2A_CORR, idx, 1)
                            : keys.reader.get_at(base + 3 + i * 67, 64);
        ebits[i] = b[i] ^ rb;
    }
    keys.reader.pos = base + (uint64_t)rec_bits * n;
    keys.stream_advance(REC_B2A_CORR, n);

    std::vector<uint8_t> snd((n + 7) / 8), rcv(snd.size());
    pack_small(snd.data(), ebits.data(), n, 1);
    chan.exchange(snd.data(), rcv.data(), snd.size());
    std::vector<uint8_t> epeer(n);
    unpack_small(rcv.data(), epeer.data(), n, 1);
    const uint64_t me = (party == 0) ? 1 : 0;
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t i = 0; i < n; i++) {
        uint8_t e = epeer[i] ^ ebits[i];
        z[i] = (e == 0) ? ra[i] : (me - ra[i]);
    }
}

// ── open 辅助 ───────────────────────────────────────────────────────
void LssParty::open_bits(const uint8_t *in, uint8_t *out, size_t n) {
    std::vector<uint8_t> peer(n);
    chan.exchange(in, peer.data(), n);
#ifdef _OPENMP
#pragma omp parallel for if(n > 65536)
#endif
    for (size_t i = 0; i < n; i++) out[i] = in[i] ^ peer[i];
}

void LssParty::open_u64(const uint64_t *in, uint64_t *out, size_t n) {
    std::vector<uint64_t> peer(n);
    chan.exchange(in, peer.data(), n * 8);
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192)
#endif
    for (size_t i = 0; i < n; i++) out[i] = in[i] + peer[i];
}

} // namespace lss
