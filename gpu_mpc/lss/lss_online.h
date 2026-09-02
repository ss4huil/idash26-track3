// lss_online.h — LSS 在线原语（gpu-mpc-track v2 Phase 1）
//
// 公式定义见 lss_protocol.md（唯一权威）。双方各持 key 文件游标
// （KeyFileReader 顺序消费，标签校验防图序错位）+ 一条 Channel。
// 全部为 batched 接口：一次调用消费 n 条记录、1 轮通信。
#pragma once

#include "lss_keys.h"
#include "lss_channel.h"

namespace lss {

class LssParty {
public:
    const int party; // 0 或 1
    KeyFileReader keys;
    Channel &chan;

    LssParty(int party, const std::string &key_path, Channel &chan)
        : party(party), keys(key_path), chan(chan) {}

    // ── 原语 1：AND ─────────────────────────────────────────────────
    // x,y,z 为布尔份额（每元素 0/1）。z = AND(x,y) 的布尔份额。
    void and_open(const uint8_t *x, const uint8_t *y, uint8_t *z, size_t n);

    // ── 原语 2：1oo16 OT ────────────────────────────────────────────
    // sender 角色：g[e][0..15] ∈ {0..3} 为明文函数（未掩码），
    //   s[e] ∈ {0..3} 为 sender 本地选择的输出份额（掩码）；
    //   sender 无返回值——其输出份额即 s。
    // receiver 角色：d[e] ∈ [0,16) 为选择；out[e] = g[e][d[e]] ⊕ s[e]（2 bit）。
    // 两侧交换后的 XOR 份额满足 share_s ⊕ share_r = g(d)。
    void ot16_send(const uint8_t *g, const uint8_t *s, size_t n);
    void ot16_recv(const uint8_t *d, uint8_t *out, size_t n);

    // ── 原语 3：MUX ─────────────────────────────────────────────────
    // b 布尔份额（0/1），x 算术份额；z = b·x 的算术份额。1 轮 open。
    void mux(const uint8_t *b, const uint64_t *x, uint64_t *z, size_t n);

    // ── 原语 4：B2A ─────────────────────────────────────────────────
    // b 布尔份额；z = b 的算术份额（z0 + z1 ≡ β mod 2^64）。
    void b2a(const uint8_t *b, uint64_t *z, size_t n);

    // ── 测试/调试辅助：open ─────────────────────────────────────────
    void open_bits(const uint8_t *in, uint8_t *out, size_t n);   // XOR 重构
    void open_u64(const uint64_t *in, uint64_t *out, size_t n);  // 加法重构
};

} // namespace lss
