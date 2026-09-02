// lss_online.cpp — LSS 在线原语实现（公式见 lss_protocol.md）
#include "lss_online.h"

namespace lss {

// ── 原语 1：AND ─────────────────────────────────────────────────────
// e = open(x ⊕ a), f = open(y ⊕ b)
// z_i = c_i ⊕ (f∧a_i) ⊕ (e∧b_i) ⊕ (e∧f∧[i==0])
void LssParty::and_open(const uint8_t *x, const uint8_t *y, uint8_t *z,
                        size_t n) {
    std::vector<uint8_t> a(n), b(n), c(n);
    for (size_t i = 0; i < n; i++) {
        keys.expect(REC_BIT_TRIPLE);
        a[i] = (uint8_t)keys.reader.get(1);
        b[i] = (uint8_t)keys.reader.get(1);
        c[i] = (uint8_t)keys.reader.get(1);
    }
    // 打包 e_i | f_i<<1（各 1 bit/元素）
    BitWriter w;
    for (size_t i = 0; i < n; i++) {
        w.put(x[i] ^ a[i], 1);
        w.put(y[i] ^ b[i], 1);
    }
    std::vector<uint8_t> peer(w.buf.size());
    chan.exchange(w.buf.data(), peer.data(), w.buf.size());
    BitReader r(peer.data(), n * 2);
    uint8_t me = (party == 0) ? 1 : 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t e = (uint8_t)r.get(1) ^ (x[i] ^ a[i]);
        uint8_t f = (uint8_t)r.get(1) ^ (y[i] ^ b[i]);
        z[i] = c[i] ^ (f & a[i]) ^ (e & b[i]) ^ (e & f & me);
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
    BitReader dr(dbuf.data(), n * 4);

    BitWriter w; // 16×2 bit = 4 B/元素
    for (size_t e = 0; e < n; e++) {
        keys.expect(REC_OT16_SEND);
        uint8_t k[16];
        for (int j = 0; j < 16; j++) k[j] = (uint8_t)keys.reader.get(2);
        uint8_t delta = (uint8_t)dr.get(4);
        uint8_t sv = s[e] & 3;
        for (int i = 0; i < 16; i++) {
            uint8_t f = (g[e * 16 + i] & 3) ^ sv;
            w.put(f ^ k[(i + delta) & 15], 2);
        }
    }
    chan.send_all(w.buf.data(), w.buf.size());
}

void LssParty::ot16_recv(const uint8_t *d, uint8_t *out, size_t n) {
    // 读自己的 (r, k_r)，发 δ = (r − d) mod 16
    BitWriter w;
    std::vector<uint8_t> kr(n);
    for (size_t e = 0; e < n; e++) {
        keys.expect(REC_OT16_RECV);
        uint8_t r = (uint8_t)keys.reader.get(4);
        kr[e] = (uint8_t)keys.reader.get(2);
        w.put((r - d[e]) & 15, 4);
    }
    chan.send_all(w.buf.data(), w.buf.size());
    // 收 16×2 bit/元素
    size_t cbytes = n * 4;
    std::vector<uint8_t> cbuf(cbytes);
    chan.recv_all(cbuf.data(), cbytes);
    BitReader cr(cbuf.data(), n * 32);
    for (size_t e = 0; e < n; e++) {
        uint8_t cd = 0;
        for (int i = 0; i < 16; i++) {
            uint8_t ci = (uint8_t)cr.get(2);
            if (i == d[e]) cd = ci;
        }
        out[e] = cd ^ kr[e];
    }
}

// ── 原语 3：MUX ─────────────────────────────────────────────────────
// e = open(β ⊕ a^b), f = open(x − b)
// e==0: z_i = c_i + a^a_i·f ;  e==1: z_i = x_i − c_i − a^a_i·f
void LssParty::mux(const uint8_t *b, const uint64_t *x, uint64_t *z, size_t n) {
    std::vector<uint8_t> ab(n);
    std::vector<uint64_t> aa(n), bb(n), cc(n);
    for (size_t i = 0; i < n; i++) {
        keys.expect(REC_MUX_TRIPLE);
        ab[i] = (uint8_t)keys.reader.get(1);
        aa[i] = keys.reader.get(64);
        bb[i] = keys.reader.get(64);
        cc[i] = keys.reader.get(64);
    }
    // 交换：e 比特打包（n bit）+ f 数组（n×8 B）
    BitWriter w;
    for (size_t i = 0; i < n; i++) w.put(b[i] ^ ab[i], 1);
    size_t ebytes = w.buf.size();
    std::vector<uint8_t> snd(ebytes + n * 8), rcv(snd.size());
    memcpy(snd.data(), w.buf.data(), ebytes);
    for (size_t i = 0; i < n; i++) {
        uint64_t f = x[i] - bb[i];
        memcpy(snd.data() + ebytes + i * 8, &f, 8);
    }
    chan.exchange(snd.data(), rcv.data(), snd.size());
    BitReader er(rcv.data(), n);
    for (size_t i = 0; i < n; i++) {
        uint8_t e = (uint8_t)er.get(1) ^ (b[i] ^ ab[i]);
        uint64_t f_peer;
        memcpy(&f_peer, rcv.data() + ebytes + i * 8, 8);
        uint64_t f = (x[i] - bb[i]) + f_peer;
        if (e == 0)
            z[i] = cc[i] + aa[i] * f;
        else
            z[i] = x[i] - cc[i] - aa[i] * f;
    }
}

// ── 原语 4：B2A ─────────────────────────────────────────────────────
// e = open(β ⊕ rb)；e==0: z_i = ra_i；e==1: z_i = [i==0] − ra_i
void LssParty::b2a(const uint8_t *b, uint64_t *z, size_t n) {
    std::vector<uint8_t> rb(n);
    std::vector<uint64_t> ra(n);
    for (size_t i = 0; i < n; i++) {
        keys.expect(REC_B2A_CORR);
        rb[i] = (uint8_t)keys.reader.get(1);
        ra[i] = keys.reader.get(64);
    }
    BitWriter w;
    for (size_t i = 0; i < n; i++) w.put(b[i] ^ rb[i], 1);
    std::vector<uint8_t> peer(w.buf.size());
    chan.exchange(w.buf.data(), peer.data(), peer.size());
    BitReader er(peer.data(), n);
    uint64_t me = (party == 0) ? 1 : 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t e = (uint8_t)er.get(1) ^ (b[i] ^ rb[i]);
        z[i] = (e == 0) ? ra[i] : (me - ra[i]);
    }
}

// ── open 辅助 ───────────────────────────────────────────────────────
void LssParty::open_bits(const uint8_t *in, uint8_t *out, size_t n) {
    std::vector<uint8_t> peer(n);
    chan.exchange(in, peer.data(), n);
    for (size_t i = 0; i < n; i++) out[i] = in[i] ^ peer[i];
}

void LssParty::open_u64(const uint64_t *in, uint64_t *out, size_t n) {
    std::vector<uint64_t> peer(n);
    chan.exchange(in, peer.data(), n * 8);
    for (size_t i = 0; i < n; i++) out[i] = in[i] + peer[i];
}

} // namespace lss
