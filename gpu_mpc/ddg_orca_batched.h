//
// Optimized DDG Orca backend with batched protocol operations and reduced-round truncation.
// This is a LOCAL FORK — does NOT modify upstream EzPC/GPU-MPC.
//
// Optimizations:
//   1. TrWithSlack: 2 comm rounds (vs TrFloor 4 rounds), bounded error, correct
//   2. LocalARS: 0 comm rounds, catastrophic error (NOT RECOMMENDED)
//
// NOTE: Cross-layer ReLU batching is NOT possible for this architecture — the
// network is strictly sequential (GCN1→GCN2→GCN3→ffc1→ffc2→ffc3), each ReLU
// depends on the prior layer's output. The batch dim (B=8) is already folded
// into every ReLU call (N = B × nodes × features), so batch amortization is
// already exploited by Sigma's DPF eval.
//
// Usage:
//   DDG_SLACK_TRUNC=1   — use TrWithSlack instead of TrFloor (2 vs 4 rounds)
//   DDG_LOCAL_TRUNC=1   — use LocalARS (BROKEN: MSE~5200, DO NOT USE)
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
    bool use_local_trunc = false;

public:
    DDGOrcaBatched() : DDGOrca<T>() {}

    DDGOrcaBatched(int party, std::string ip, int bw, int scale, std::string keyFile = "")
        : DDGOrca<T>(party, ip, bw, scale, keyFile)
    {
        // Read optimization flags
        use_slack_trunc = (std::getenv("DDG_SLACK_TRUNC") != nullptr);
        use_local_trunc = (std::getenv("DDG_LOCAL_TRUNC") != nullptr);
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0) override
    {
        if (use_slack_trunc) {
            // Sigma TrWithSlack: 2 comm rounds (DReLU + folded MSB corr) vs TrFloor's 4.
            // Correct with bounded probabilistic error; data must fit [-2^(k-2), 2^(k-2)).
            auto start = std::chrono::high_resolution_clock::now();

            auto k = readGPUTruncateKey<T>(TruncateType::TrWithSlack, &(this->keyBuf));
            in.d_data = gpuTruncate<T, T>(this->bw, this->bw, TruncateType::TrWithSlack, k,
                                          (int)shift, this->peer, this->party, in.size(),
                                          in.d_data, &(this->g), &(this->s));

            auto end = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            this->s.truncate_time += us;
        } else if (use_local_trunc) {
            // Use Sigma's LocalARS: 0 communication, local arithmetic right shift
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
        } else {
            // Standard TrFloor (4 rounds)
            DDGOrca<T>::truncateForward(in, shift, mode);
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
    bool use_local_trunc = false;

public:
    DDGOrcaKeygenBatched(int party, int bw, int scale, std::string keyFile)
        : DDGOrcaKeygen<T>(party, bw, scale, keyFile)
    {
        use_slack_trunc = (std::getenv("DDG_SLACK_TRUNC") != nullptr);
        use_local_trunc = (std::getenv("DDG_LOCAL_TRUNC") != nullptr);
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0) override
    {
        if (use_slack_trunc) {
            // Generate TrWithSlack keys (DReLU bit + LSB/MSB correction tables)
            in.d_data = genGPUTruncateKey<T, T>(
                &(this->keyBuf), this->party, TruncateType::TrWithSlack,
                this->bw, this->bw, (int)shift, in.size(), in.d_data, &(this->g));
        } else if (use_local_trunc) {
            // Generate LocalARS keys (Sigma handles this internally)
            in.d_data = genGPUTruncateKey<T, T>(
                &(this->keyBuf), this->party, TruncateType::LocalARS,
                this->bw, this->bw, (int)shift, in.size(), in.d_data, &(this->g));
        } else {
            // Standard TrFloor keygen
            DDGOrcaKeygen<T>::truncateForward(in, shift, mode);
        }
    }
};
