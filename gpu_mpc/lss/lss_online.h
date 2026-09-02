// lss_online.h — LSS 在线原语（gpu-mpc-track v2 Phase 1）
//
// 公式定义见 lss_protocol.md（唯一权威）。双方各持 key 文件游标
// （KeyFileReader 顺序消费，标签校验防图序错位）+ 一条 Channel。
// 全部为 batched 接口：一次调用消费 n 条记录、1 轮通信。
#pragma once

#include "lss_keys.h"
#include "lss_channel.h"
#include <chrono>
#include <cstdlib>
#include <random>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lss {

// 线程数：DDG_LSS_THREADS 覆盖，默认 hardware_concurrency（评测机 16 核）
inline int lss_num_threads() {
    const char *e = std::getenv("DDG_LSS_THREADS");
    if (e && atoi(e) > 0) return atoi(e);
    unsigned h = std::thread::hardware_concurrency();
    return h ? (int)h : 1;
}

inline void lss_init_threads() {
#ifdef _OPENMP
    omp_set_num_threads(lss_num_threads());
#endif
}

class LssParty {
public:
    const int party; // 0 或 1
    KeyFileReader keys;
    Channel &chan;

    LssParty(int party, const std::string &key_path, Channel &chan,
             uint64_t local_seed = 0)
        : party(party), keys(key_path), chan(chan),
          local_rng(local_seed ? (local_seed ^ (0x9E3779B97F4A7C15ULL * (party + 1)))
                               : (uint64_t)std::random_device{}()) {
        lss_init_threads();
    }

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

    // ── 算子层（P2，实现在 lss_compare.cpp，公式见 lss_protocol.md §5-7）──
    // Millionaires 比较（radix-2^4，party0 恒为 OT sender）：
    //   val 为本方明文输入（party0 的 a / party1 的 b），
    //   out = 1{a > b} 的布尔份额。语义对齐 SCI millionaire.h compare
    //   （greater_than=true）；1{x<y} 可由双方互换输入获得。
    void compare_gt(uint8_t *out, const uint64_t *val, size_t n, int bitlength);

    // MSB：msb(x) = msb(x0) ⊕ msb(x1) ⊕ wrap，wrap = 1{b0+b1 ≥ 2^(bw−1)}
    // 由 (bw−1)-bit compare_gt 得到（推导见 lss_protocol.md §6）。
    void msb(const uint64_t *x, uint8_t *out, size_t n, int bw = 64);

    // DReLU(x) = 1{x ≥ 0}（有符号语义）= ¬msb(x) 的布尔份额（本地翻转）。
    void drelu(const uint64_t *x, uint8_t *out, size_t n, int bw = 64);

    // ReLU(x) = MUX(DReLU(x), x)：x ≥ 0 输出 x，否则 0（算术份额）。
    void relu(const uint64_t *x, uint64_t *y, size_t n, int bw = 64);

    // ── GPU 管线桥接版 ReLU（P3，设计文档 §5.2）─────────────────────
    // 输入 h_m = masked-public 值 m = x + r（双方同持，已从 GPU 拉到 host）。
    //   入口：读 n 条 MASK_IN 份额 r_p；P0: x0 = m − r0，P1: x1 = −r1；
    //   LSS relu（compare + mux）；
    //   出口：读 n 条 MASK_OUT 份额 r'_p，w_p = y_p + r'_p，双方交换重构
    //         ⇒ h_out = relu(x) + r'（masked-public，+1 轮）。
    // 若 key 流已耗尽（上一迭代刚好消费完）则自动回卷复用——与 FSS 侧
    // 每迭代 keyBuf = startPtr 复位的 benchmark 纪律一致（仅用于计时/
    // 精度验证，生产必须每批新 key）。
    void relu_bridged(const uint64_t *h_m, uint64_t *h_out, size_t n,
                      int bw = 64);

    // ── 测试/调试辅助：open ─────────────────────────────────────────
    void open_bits(const uint8_t *in, uint8_t *out, size_t n);   // XOR 重构
    void open_u64(const uint64_t *in, uint64_t *out, size_t n);  // 加法重构

    // 本地 PRNG：OT16 sender 的掩码 s 等本地随机性（对端不可见，
    // 无需与 key 流对齐）。seed=0 时用 random_device。
    std::mt19937_64 local_rng;

    // ── 分阶段耗时统计（微秒，性能分析用；退出时由集成层打印）─────────
    uint64_t us_leaf = 0;    // 叶子 OT16（含 g 表/回复构造）
    uint64_t us_tree = 0;    // AND 树
    uint64_t us_mux = 0;     // MUX
    uint64_t us_bridge = 0;  // 桥接入口/出口（mask 转换 + 出口重构）
    uint64_t us_comm = 0;    // socket 收发（嵌套于上述各阶段内）
};

// 简单计时器
struct LssTimer {
    std::chrono::high_resolution_clock::time_point t0;
    LssTimer() : t0(std::chrono::high_resolution_clock::now()) {}
    uint64_t us() const {
        return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - t0)
            .count();
    }
};

} // namespace lss
