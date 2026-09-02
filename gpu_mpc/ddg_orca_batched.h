//
// Optimized DDG Orca backend with batched protocol operations and reduced-round truncation.
// This is a LOCAL FORK — does NOT modify upstream EzPC/GPU-MPC.
//
// Truncation modes (v2, design doc §2 decision 2):
//   - LocalARS (DEFAULT since v2): 0 comm rounds, 0 key bytes, pure local
//     arithmetic right shift on each share. Safe ONLY at bw=64/scale=12 —
//     catastrophic wrap probability ~|x|·2^(2s-bw) = |x|·2^-40 (appendix A).
//     At bw=32 it destroys accuracy (the old "BROKEN: MSE~5200" note was a
//     bw=32 measurement, not an implementation bug).
//   - TrFloor (DDG_EXACT_TRUNC=1): exact FSS truncation, 4 rounds. Kept as
//     the fallback / A-B comparison switch.
//   - TrWithSlack (DDG_SLACK_TRUNC=1): DEPRECATED, superseded by LocalARS.
//
// NOTE: Cross-layer ReLU batching is NOT possible for this architecture — the
// network is strictly sequential (GCN1→GCN2→GCN3→ffc1→ffc2→ffc3), each ReLU
// depends on the prior layer's output. The batch dim (B=8) is already folded
// into every ReLU call (N = B × nodes × features), so batch amortization is
// already exploited by Sigma's DPF eval.
//
// Usage:
//   (default)          — LocalARS probabilistic truncation (v2 production)
//   DDG_EXACT_TRUNC=1  — FSS TrFloor exact truncation (fallback)
//   DDG_SLACK_TRUNC=1  — DEPRECATED: TrWithSlack (2 rounds), kept for old data
//   DDG_LOCAL_TRUNC=1  — legacy alias for the default LocalARS path
//

#pragma once

#include "ddg_orca.h"
#include <vector>

// ── v2 Phase 3: LSS ReLU 分流（DDG_LSS_RELU=1）────────────────────────────
// 默认关闭（保持 P0 行为可回归）。开启后 mode!=2 的 relu 走 CPU LSS 路径
// （dealer 直给相关性 + Millionaires/MSB/MUX，gpu_mpc/lss/），不再读写
// keyBuf 中的 DPF relu key；mode==2（ReluExtend）仍走 FSS 原路径。
// 协议与 key 格式见 lss/lss_protocol.md。LSS 通道端口：DDG_LSS_PORT
// （默认 43003，与 GpuPeer 的 42003 并行）。
#include "lss/lss_gpu_bridge.h"

namespace ddg_lss_detail {
// dealer 侧 LSS key 的退出时落盘：keygen 后端的 close() 在基类
// （非 virtual，静态派发），无法挂钩子；进程正常退出时由本 RAII 写文件。
struct KeygenFinalizer {
    lss::LssKeygen *kg = nullptr;
    std::string path;
    int party = -1;
    ~KeygenFinalizer() {
        if (kg) {
            kg->write_file(path, party);
            fprintf(stderr, "[lss-keygen] wrote %s\n", path.c_str());
        }
    }
};
inline KeygenFinalizer &finalizer() {
    static KeygenFinalizer f;
    return f;
}
inline int lss_port() {
    const char *e = std::getenv("DDG_LSS_PORT");
    return e ? atoi(e) : 43003;
}
// eval 侧 LSS 通道统计（atexit 打印；GpuPeer 的 Comm(B) 不含 LSS 通道）
inline lss::Channel *g_eval_chan = nullptr;
inline lss::LssParty *g_eval_lss = nullptr;
inline int g_eval_party = -1;
inline void print_eval_stats_trampoline() {
    if (g_eval_chan)
        fprintf(stderr,
                "[lss-eval] party%d: LSS channel total sent=%llu B recv=%llu B\n",
                g_eval_party, (unsigned long long)g_eval_chan->bytes_sent,
                (unsigned long long)g_eval_chan->bytes_recv);
    if (g_eval_lss)
        fprintf(stderr,
                "[lss-eval] party%d: LSS 分阶段耗时(ms) leaf=%.1f tree=%.1f "
                "mux=%.1f bridge=%.1f\n",
                g_eval_party, g_eval_lss->us_leaf / 1e3, g_eval_lss->us_tree / 1e3,
                g_eval_lss->us_mux / 1e3, g_eval_lss->us_bridge / 1e3);
}
} // namespace ddg_lss_detail

// ============================================================================
// Optimized Eval Backend
// ============================================================================

template <typename T>
class DDGOrcaBatched : public DDGOrca<T>
{
private:
    bool use_slack_trunc = false;
    bool use_exact_trunc = false;

    // ── v2 P3: LSS ReLU 状态（lazy init，首次 LSS relu 时建连/读 key）──
    bool use_lss_relu = false;
    std::string lss_ip;
    std::string lss_key_path;
    lss::Channel *lss_chan_ = nullptr;
    lss::LssParty *lss_ = nullptr;

    void ensureLssEval()
    {
        if (lss_) return;
        // party0 listen、party1 connect（eval 启动顺序：party1 先起，
        // Channel::connect 自带 ~5s 重试）
        if (this->party == SERVER0)
            lss_chan_ = new lss::Channel(
                lss::Channel::listen_and_accept(ddg_lss_detail::lss_port()));
        else
            lss_chan_ = new lss::Channel(
                lss::Channel::connect(lss_ip, ddg_lss_detail::lss_port()));
        lss_ = new lss::LssParty(this->party, lss_key_path, *lss_chan_);
        static bool stats_registered = false;
        if (!stats_registered) {   // 进程退出时打印 LSS 通道总量
            stats_registered = true;
            std::atexit(ddg_lss_detail::print_eval_stats_trampoline);
        }
        ddg_lss_detail::g_eval_chan = lss_chan_;
        ddg_lss_detail::g_eval_lss = lss_;
        ddg_lss_detail::g_eval_party = this->party;
        fprintf(stderr, "[lss-eval] party%d: LSS channel up, key=%s\n",
                this->party, lss_key_path.c_str());
    }

public:
    DDGOrcaBatched() : DDGOrca<T>() {}

    DDGOrcaBatched(int party, std::string ip, int bw, int scale, std::string keyFile = "")
        : DDGOrca<T>(party, ip, bw, scale, keyFile)
    {
        // Truncation mode: LocalARS is the default (v2 production).
        // DDG_EXACT_TRUNC=1 falls back to exact FSS TrFloor.
        // DDG_SLACK_TRUNC is DEPRECATED (superseded by LocalARS) but honoured.
        use_slack_trunc = (std::getenv("DDG_SLACK_TRUNC") != nullptr);
        use_exact_trunc = (std::getenv("DDG_EXACT_TRUNC") != nullptr);

        use_lss_relu = (std::getenv("DDG_LSS_RELU") != nullptr);
        if (use_lss_relu) {
            lss_ip = ip;
            lss_key_path = keyFile + "_lss.bin";
        }
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) override
    {
        if (use_lss_relu && mode != 2)
        {
            auto start = std::chrono::high_resolution_clock::now();
            ensureLssEval();
            // mode==3 时输入在 bw−scale 环上（镜像 keygen 的 tmpBw 约定）
            int lss_bw = this->bw - (mode == 3 ? (int)scale : 0);
            out.d_data = lssReluEvalGpu<T>(lss_, in.d_data, in.size(), lss_bw,
                                           &this->s);
            auto end = std::chrono::high_resolution_clock::now();
            this->s.relu_time +=
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            return;
        }
        DDGOrca<T>::relu(in, out, drelu, scale, mode);
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0) override
    {
        if (use_slack_trunc) {
            // DEPRECATED. Sigma TrWithSlack: 2 comm rounds (DReLU + folded MSB
            // corr) vs TrFloor's 4. Correct with bounded probabilistic error;
            // data must fit [-2^(k-2), 2^(k-2)).
            auto start = std::chrono::high_resolution_clock::now();

            auto k = readGPUTruncateKey<T>(TruncateType::TrWithSlack, &(this->keyBuf));
            in.d_data = gpuTruncate<T, T>(this->bw, this->bw, TruncateType::TrWithSlack, k,
                                          (int)shift, this->peer, this->party, in.size(),
                                          in.d_data, &(this->g), &(this->s));

            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            this->s.truncate_time += us;
        } else if (use_exact_trunc) {
            // Fallback: standard exact FSS TrFloor (4 rounds)
            DDGOrca<T>::truncateForward(in, shift, mode);
        } else {
            // DEFAULT (v2): Sigma's LocalARS — 0 communication, 0 key bytes,
            // local arithmetic right shift. Safe at bw=64/scale=12 only.
            auto start = std::chrono::high_resolution_clock::now();

            // LocalARS keygen wrote 0 bytes; eval must consume 0 bytes to keep
            // keyBuf aligned with subsequent matmul/relu keys. gpuTruncate's
            // LocalARS branch ignores `k` entirely (calls gpuLocalTr directly),
            // so we pass an empty key without advancing keyBuf.
            GPUTruncateKey<T> k;

            // Call Sigma's LocalARS truncate (built-in, no protocol overhead)
            in.d_data = gpuTruncate<T, T>(this->bw, this->bw, TruncateType::LocalARS, k,
                                          (int)shift, this->peer, this->party, in.size(),
                                          in.d_data, &(this->g), &(this->s));

            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            this->s.truncate_time += us;
        }
    }
};

// ============================================================================
// Optimized Keygen Backend
// ============================================================================

template <typename T>
class DDGOrcaKeygenBatched : public DDGOrcaKeygen<T>
{
private:
    bool use_slack_trunc = false;
    bool use_exact_trunc = false;

    // ── v2 P3: LSS dealer keygen 状态 ────────────────────────────────
    // 固定 seed：两个 dealer 进程生成完全一致的记录流（与 FSS keygen 的
    // 确定性随机流同款纪律），各自只写自己 party 的文件。
    bool use_lss_relu = false;
    std::string lss_key_path;
    lss::LssKeygen *lss_kg_ = nullptr;

    void ensureLssKeygen()
    {
        if (lss_kg_) return;
        lss_kg_ = new lss::LssKeygen(/*seed=*/0x4C53532D7632ULL); // "LSS-v2"
        auto &fin = ddg_lss_detail::finalizer();
        fin.kg = lss_kg_;
        fin.path = lss_key_path;
        fin.party = this->party;
        fprintf(stderr, "[lss-keygen] party%d: LSS keygen -> %s\n",
                this->party, lss_key_path.c_str());
    }

public:
    DDGOrcaKeygenBatched(int party, int bw, int scale, std::string keyFile)
        : DDGOrcaKeygen<T>(party, bw, scale, keyFile)
    {
        // Must mirror DDGOrcaBatched's truncation mode selection exactly —
        // keygen and eval read the same env on both sides.
        use_slack_trunc = (std::getenv("DDG_SLACK_TRUNC") != nullptr);
        use_exact_trunc = (std::getenv("DDG_EXACT_TRUNC") != nullptr);

        use_lss_relu = (std::getenv("DDG_LSS_RELU") != nullptr);
        if (use_lss_relu)
            lss_key_path = keyFile + "_lss.bin";
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) override
    {
        if (use_lss_relu && mode != 2)
        {
            // keygen 模式下 in.d_data 即输入 mask r；dealer 写 LSS 记录并
            // 返回新输出 mask r'（GPU），由后续算子继续跟踪。
            ensureLssKeygen();
            int lss_bw = this->bw - (mode == 3 ? (int)scale : 0);
            out.d_data = lssReluKeygenGpu<T>(lss_kg_, in.d_data, in.size(),
                                             lss_bw);
            return;
        }
        DDGOrcaKeygen<T>::relu(in, out, drelu, scale, mode);
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0) override
    {
        if (use_slack_trunc) {
            // DEPRECATED. Generate TrWithSlack keys (DReLU bit + LSB/MSB
            // correction tables)
            in.d_data = genGPUTruncateKey<T, T>(
                &(this->keyBuf), this->party, TruncateType::TrWithSlack,
                this->bw, this->bw, (int)shift, in.size(), in.d_data, &(this->g));
        } else if (use_exact_trunc) {
            // Fallback: standard TrFloor keygen
            DDGOrcaKeygen<T>::truncateForward(in, shift, mode);
        } else {
            // DEFAULT (v2): LocalARS — writes 0 key bytes
            in.d_data = genGPUTruncateKey<T, T>(
                &(this->keyBuf), this->party, TruncateType::LocalARS,
                this->bw, this->bw, (int)shift, in.size(), in.d_data, &(this->g));
        }
    }
};
