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

// ============================================================================
// Optimized Eval Backend
// ============================================================================

template <typename T>
class DDGOrcaBatched : public DDGOrca<T>
{
private:
    bool use_slack_trunc = false;
    bool use_exact_trunc = false;

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

public:
    DDGOrcaKeygenBatched(int party, int bw, int scale, std::string keyFile)
        : DDGOrcaKeygen<T>(party, bw, scale, keyFile)
    {
        // Must mirror DDGOrcaBatched's truncation mode selection exactly —
        // keygen and eval read the same env on both sides.
        use_slack_trunc = (std::getenv("DDG_SLACK_TRUNC") != nullptr);
        use_exact_trunc = (std::getenv("DDG_EXACT_TRUNC") != nullptr);
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
