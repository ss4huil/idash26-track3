// DPF-DCF Adapter — redirect old fss/dcf/* to new DPF-based DCF from fss/gpu_dpf.cu
//
// This adapter allows ddg_orca.h to use the newer DPF-based DCF implementation
// (Algorithm 1 from FSS-DT.pdf) without modifying EzPC open-source files.
//
// Strategy: The old fss/dcf/* functions are wrappers around base DCF primitives.
// We provide compatible wrappers that call the DPF-based DCF from gpu_dpf.cu instead.
//
// EXCEPTION: ReLU-Extend uses DCF namespace directly (dual-output semantics incompatible with DPF)
//
// Copyright (c) 2026 iDASH Track 3 submission team.

#pragma once

#include "fss/gpu_dpf.h"
#include "fss/gpu_select.h"
#include "fss/gpu_relu.h"  // Sigma's native DPF-based ReLU
#include "fss/dcf/gpu_relu.h"  // DCF-based ReLU-Extend (dual-output)
#include "utils/gpu_comms.h"
#include "utils/gpu_stats.h"

// Reuse the prologue/epilogue templates from DPF
#include "fss/gpu_dpf_templates.h"

namespace dpf_dcf
{
    // ============================================================================
    // Key Structures — maintain compatibility with old DCF API
    // ============================================================================

    // DRelu key structure (for derivative ReLU computation)
    struct GPUDReluKey
    {
        GPUDPFKey dcfKey;  // Use DPF key structure (contains DCF via doDcf)
        u32 *dReluMask;
    };

    template <typename T>
    struct GPU2RoundReLUKey
    {
        int bin, bout, N;
        GPUDReluKey dreluKey;
        GPUSelectKey<T> selectKey;
    };

    template <typename T>
    struct GPUReluExtendKey
    {
        int bin, bout, N;
        GPUDReluKey dReluKey;
        u32 *dcfMask;
        T *oneHot;
        T *outMask;
    };

    // Truncation key structures
    using GPUMaskedDCFKey = GPUDReluKey;

    template <typename T>
    struct GPUStTRKey
    {
        int bin, bout, shift, N;
        GPUMaskedDCFKey lsbKey;
        T *lsbCorr;
    };

    template <typename T>
    struct GPUSignExtendKey
    {
        int bin, bout, N;
        GPUMaskedDCFKey dcfKey;
        T *t, *p;
    };

    template <typename T>
    struct GPUTruncateKey
    {
        GPUStTRKey<T> stTRKey;
        GPUSignExtendKey<T> signExtendKey;
    };

    // Maxpool key structure
    template <typename T>
    struct GPUMaxpoolKey
    {
        GPU2RoundReLUKey<T> *reluKey;
        GPUAndKey *andKey;
    };

    // ============================================================================
    // Key Reading Functions
    // ============================================================================

    // Read DReluKey with explicit bout parameter (DPF key doesn't store bout like old DCF)
    // bout selects the packed width of the dRelu mask: TwoRoundRelu writes bout=1,
    // ReluExtend writes bout=2 (see gpuKeygenReluExtend). Must match keygen exactly.
    GPUDReluKey readGPUDReluKey(u8 **key_as_bytes, int bout = 1)
    {
        GPUDReluKey k;
        k.dcfKey = readGPUDcfKey(key_as_bytes);  // Read as DPF key (has DCF support)
        k.dReluMask = (u32 *)*key_as_bytes;
        *key_as_bytes += ((bout * k.dcfKey.M - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
        return k;
    }

    // For masked DCF in truncation, bout=1
    GPUDReluKey readGPUMaskedDCFKey(u8 **key_as_bytes)
    {
        return readGPUDReluKey(key_as_bytes, 1);
    }

    template <typename T>
    GPU2RoundReLUKey<T> readTwoRoundReluKey(u8 **key_as_bytes)
    {
        GPU2RoundReLUKey<T> k;
        k.bin = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.bout = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.N = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.dreluKey = readGPUDReluKey(key_as_bytes);
        k.selectKey = readGPUSelectKey<T>(key_as_bytes, k.N);
        return k;
    }

    template <typename T>
    dcf::GPUReluExtendKey<T> readGPUReluExtendKey(u8 **key_as_bytes)
    {
        // Delegate to DCF namespace — keygen writes DCF-format keys
        return dcf::readGPUReluExtendKey<T>(key_as_bytes);
    }

    template <typename T>
    GPUSignExtendKey<T> readGPUSignExtendKey(u8 **key_as_bytes)
    {
        GPUSignExtendKey<T> k;
        k.bin = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.bout = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.N = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.dcfKey = readGPUMaskedDCFKey(key_as_bytes);
        size_t memSz = k.dcfKey.dcfKey.M * sizeof(T);
        k.t = (T *)*key_as_bytes;
        *key_as_bytes += memSz;
        k.p = (T *)*key_as_bytes;
        *key_as_bytes += 2 * memSz;
        return k;
    }

    template <typename T>
    GPUStTRKey<T> readGPUStTRKey(u8 **key_as_bytes)
    {
        GPUStTRKey<T> k;
        memcpy(&k, *key_as_bytes, 4 * sizeof(int));
        *key_as_bytes += 4 * sizeof(int);
        k.lsbKey = readGPUMaskedDCFKey(key_as_bytes);
        size_t memSz = k.N * sizeof(T);
        k.lsbCorr = (T *)*key_as_bytes;
        *key_as_bytes += 2 * memSz;
        return k;
    }

    template <typename T>
    GPUTruncateKey<T> readGPUTrStochasticKey(u8 **key_as_bytes)
    {
        GPUTruncateKey<T> k;
        k.stTRKey = readGPUStTRKey<T>(key_as_bytes);
        k.signExtendKey = readGPUSignExtendKey<T>(key_as_bytes);
        return k;
    }

    template <typename T>
    GPUMaxpoolKey<T> readGPUMaxpoolKey(MaxpoolParams p, u8 **key_as_bytes)
    {
        GPUMaxpoolKey<T> k;
        int rounds = p.FH * p.FW - 1;
        k.reluKey = new GPU2RoundReLUKey<T>[rounds + 1];
        for (int i = 0; i < rounds; i++)
        {
            k.reluKey[i + 1] = readTwoRoundReluKey<T>(key_as_bytes);
        }
        return k;
    }

    // ============================================================================
    // Runtime Functions — forward to DPF-based DCF
    // ============================================================================

    // NOTE: The old DCF namespace called gpuDcf which was in fss/dcf/gpu_dcf.cu
    // We redirect to the DPF-based gpuDcf from fss/gpu_dpf.cu which uses Algorithm 1

    template <typename T>
    std::pair<u32 *, T *> gpuTwoRoundRelu(SigmaPeer *peer, int party,
                                          GPU2RoundReLUKey<T> k, T *d_I,
                                          AESGlobalContext *gaes, Stats *s)
    {
        // Build Sigma's native GPUReluKey<T> from our adapter key, then call gpuRelu.
        // Sigma's GPUReluKey (global ns) is templated only on <T>; the p/q/flip
        // parameters are passed to gpuRelu itself (0/0/false for standard relu).
        ::GPUReluKey<T> reluKey;
        reluKey.bin = k.bin;
        reluKey.bout = k.bout;
        reluKey.numRelus = k.N;
        reluKey.dreluKey.dpfKey = k.dreluKey.dcfKey;
        reluKey.dreluKey.mask = k.dreluKey.dReluMask;
        reluKey.selectKey = k.selectKey;

        auto d_relu = gpuRelu<T, T, 0, 0, false>(peer, party, reluKey, d_I, gaes, s);
        return std::make_pair((u32 *)nullptr, d_relu);
    }

    // ReLU-Extend keygen and eval are delegated to dcf:: namespace below.
    // The helper functions (reluExtendMuxKeyKernel, genReluExtendMuxKey,
    // reluExtendMuxKernel, gpuReluExtendMux) are no longer needed in this adapter.

    template <typename T>
    std::pair<u32 *, T *> gpuReluExtend(SigmaPeer *peer, int party,
                                        dcf::GPUReluExtendKey<T> k, T *d_I,
                                        AESGlobalContext *g, Stats *s)
    {
        // Delegate to DCF namespace — ReLU-Extend requires dual-output DCF semantics
        // (dReluPrologue evaluates at x and x+2^(bin-1); dReluEpilogue<true> writes both
        //  drelu and xLTRin streams). The DPF single-point templates cannot produce this.
        return dcf::gpuReluExtend(peer, party, k, d_I, g, s);
    }

    // Truncation runtime functions — mirrored from dcf/gpu_truncate.cu using DPF-based DCF
    template <typename T>
    __global__ void selectForTruncateKernel(T *x, u32 *maskedDcfBit, T *outMask, T *p, int N, int party)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            int dcfBit = (((u32 *)maskedDcfBit)[i / 32] >> (threadIdx.x & 0x1f)) & 1;
            x[i] = (party == SERVER1) * x[i] + outMask[i] + p[2 * i + dcfBit];
        }
    }

    template <typename T>
    void gpuSelectForTruncate(int party, int N, T *d_I, u32 *d_maskedDcfBit, T *h_outMask, T *h_p, Stats *s)
    {
        size_t memSz = N * sizeof(T);
        auto d_outMask = (T *)moveToGPU((u8 *)h_outMask, memSz, s);
        auto d_p = (T *)moveToGPU((u8 *)h_p, 2 * memSz, s);
        selectForTruncateKernel<T><<<(N - 1) / 128 + 1, 128>>>(d_I, d_maskedDcfBit, d_outMask, d_p, N, party);
        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_outMask);
        gpuFree(d_p);
    }

    template <typename T>
    void gpuSignExtend(GPUSignExtendKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s)
    {
        gpuLinearComb(k.bin, k.N, d_I, T(1), d_I, T(1ULL << (k.bin - 1)));
        std::vector<u32 *> h_dcfMask = {k.dcfKey.dReluMask};
        // Use DPF-based DCF with idPrologue and maskEpilogue (same as old DCF runtime)
        auto d_maskedDcfBit = gpuDcf<T, 1, idPrologue, maskEpilogue>(k.dcfKey.dcfKey, party, d_I, g, s, &h_dcfMask);
        peer->reconstructInPlace(d_maskedDcfBit, 1, k.N, s);
        gpuSelectForTruncate(party, k.N, d_I, d_maskedDcfBit, k.t, k.p, s);
        peer->reconstructInPlace(d_I, k.bout, k.N, s);
        gpuFree(d_maskedDcfBit);
    }

    template <typename T>
    __global__ void stochasticTRKernel(int party, int bin, int bout, int shift, int N, T *d_I, u32 *d_dcf, T *lsbCorr)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            T lsb = (T)((d_dcf[i / PACKING_SIZE] >> (threadIdx.x & 0x1f)) & 1);
            d_I[i] = (party == SERVER1) * (d_I[i] >> shift) + lsbCorr[2 * i + lsb];
            gpuMod(d_I[i], bout);
        }
    }

    template <typename T>
    void gpuStochasticTR(GPUStTRKey<T> k, int party, SigmaPeer *peer, T *d_data, AESGlobalContext *g, Stats *s)
    {
        printf("[dpf_dcf::gpuStochasticTR] bin=%d bout=%d shift=%d N=%d\n", k.bin, k.bout, k.shift, k.N);
        printf("  lsbKey.dcfKey.bin=%d M=%d B=%d memSzOut=%lu\n",
               k.lsbKey.dcfKey.bin, k.lsbKey.dcfKey.M, k.lsbKey.dcfKey.B, k.lsbKey.dcfKey.memSzOut);
        std::vector<u32 *> h_mask = {k.lsbKey.dReluMask};
        // Use DPF-based DCF with idPrologue and maskEpilogue
        auto d_dcf = gpuDcf<T, 1, idPrologue, maskEpilogue>(k.lsbKey.dcfKey, party, d_data, g, s, &h_mask);
        peer->reconstructInPlace(d_dcf, 1, k.N, s);
        auto d_lsbCorr = (T *)moveToGPU((u8 *)k.lsbCorr, 2 * k.N * sizeof(T), s);
        stochasticTRKernel<<<(k.N - 1) / 128 + 1, 128>>>(party, k.bin, k.bout, k.shift, k.N, d_data, d_dcf, d_lsbCorr);
        peer->reconstructInPlace(d_data, k.bout, k.N, s);
        gpuFree(d_dcf);
        gpuFree(d_lsbCorr);
    }

    template <typename T>
    void gpuStochasticTruncate(GPUTruncateKey<T> k, int party, SigmaPeer *peer, T *d_data, AESGlobalContext *g, Stats *s)
    {
        gpuStochasticTR(k.stTRKey, party, peer, d_data, g, s);
        gpuSignExtend(k.signExtendKey, party, peer, d_data, g, s);
    }

    // Maxpool function
    template <typename T>
    T *gpuMaxPool(SigmaPeer *peer, int party, MaxpoolParams p, GPUMaxpoolKey<T> k,
                  T *d_I, u32 *d_oneHot, AESGlobalContext *gaes, Stats *s)
    {
        // TODO: Implement maxpool using DPF-based DCF
        assert(0 && "gpuMaxPool not yet implemented in DPF adapter");
        return nullptr;
    }

    // ============================================================================
    // Keygen Functions — use DPF-based DCF keygen
    // ============================================================================

    template <typename T>
    std::pair<T *, T *> gpuGenTwoRoundReluKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext *gaes)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        // Use Sigma's native DPF-based DReLU keygen (returns T*, not u8*)
        auto d_dreluMask = gpuKeyGenDRelu<T>(key_as_bytes, party, bin, N, d_inputMask, gaes);
        auto d_outputMask = gpuKeyGenSelect<T, T, T>(key_as_bytes, party, N, d_inputMask, d_dreluMask, bout);
        return std::make_pair(d_dreluMask, d_outputMask);
    }

    // ReLU-Extend keygen and eval are delegated to dcf:: namespace above.
    // The helper functions (reluExtendMuxKeyKernel, genReluExtendMuxKey,
    // reluExtendMuxKernel, gpuReluExtendMux) are no longer needed in this adapter.

    template <typename T>
    std::pair<u8 *, T *> gpuKeygenReluExtend(u8 **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext* g)
    {
        // Delegate to DCF namespace — ReLU-Extend requires dual-output DCF semantics
        return dcf::gpuKeygenReluExtend(key_as_bytes, party, bin, bout, N, d_inputMask, g);
    }

    // Truncation keygen — mirrored from dcf/gpu_truncate.cu using DPF-based DCF
    template <typename T>
    __global__ void signExtendKeyKernel(int bin, int bout, int N, T *inMask, u8 *dcfMask, T *t, T *p, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            t[i] = outMask[i] - inMask[i] - (T(1) << (bin - 1));
            gpuMod<T>(t[i], bout);
            assert(dcfMask[i] == 0 || dcfMask[i] == 1);
            int idx0 = dcfMask[i];
            int idx1 = 1 - idx0;
            p[2 * i + idx0] = 0;
            p[2 * i + idx1] = (T(1) << bin);
        }
    }

    template <typename T>
    T *genSignExtendKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext *gaes)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        // DPF-based DCF: signature is (key_as_bytes, party, bin, N, d_rin, gaes)
        // Old DCF had (key_as_bytes, party, bin, bout=1, N, d_rin, payload=T(1), gaes, leq=false)
        // The DPF version infers bout=bin and doesn't take payload/leq
        gpuKeyGenDCF<T>(key_as_bytes, party, bin, N, d_inputMask, gaes);
        auto d_dcfMask = randomGEOnGpu<u8>(N, 1);
        writeShares<u8, u8>(key_as_bytes, party, N, d_dcfMask, 1);
        auto d_outputMask = randomGEOnGpu<T>(N, bout);
        auto d_T = (T *)gpuMalloc(N * sizeof(T));
        auto d_p = (T *)gpuMalloc(2 * N * sizeof(T));
        signExtendKeyKernel<<<(N - 1) / 256 + 1, 256>>>(bin, bout, N, d_inputMask, d_dcfMask, d_T, d_p, d_outputMask);
        writeShares<T, T>(key_as_bytes, party, N, d_T, bout);
        writeShares<T, T>(key_as_bytes, party, 2 * N, d_p, bout);
        gpuFree(d_dcfMask);
        gpuFree(d_T);
        gpuFree(d_p);
        return d_outputMask;
    }

    template <typename T>
    __global__ void keygenStTRKernel(int party, int bin, int bout, int shift, int N, T *inputMask, T *rHat, u8 *lsbMask, T *lsbCorr, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            auto temp = inputMask[i];
            gpuMod(temp, shift);
            auto corr = 1 - 1 * (rHat[i] < temp) - (inputMask[i] >> shift) + outMask[i];
            gpuMod(corr, bout);
            auto corrM1 = corr - 1;
            gpuMod(corrM1, bout);
            lsbCorr[2 * i + lsbMask[i]] = corr;
            lsbCorr[2 * i + (lsbMask[i] ^ 1)] = corrM1;
        }
    }

    template <typename T>
    T *genGPUStTRKey(u8 **key_as_bytes, int party, int bin, int bout, int shift, int N, T *d_inputMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, shift);
        writeInt(key_as_bytes, N);
        auto d_rHat = randomGEOnGpu<T>(N, shift);
        if (h_r)
            moveIntoCPUMem((u8 *)h_r, (u8 *)d_rHat, N * sizeof(T), NULL);
        gpuLinearComb(shift, N, d_rHat, T(1), d_rHat, T(1), d_inputMask);
        // DPF-based DCF: old DCF called (key_as_bytes, party, shift, bout=1, N, d_rHat, payload=T(1), gaes, leq=true)
        // DPF version: (key_as_bytes, party, shift, N, d_rHat, gaes) — no bout/payload/leq
        gpuKeyGenDCF(key_as_bytes, party, shift, N, d_rHat, gaes);
        auto d_lsbMask = randomGEOnGpu<u8>(N, 1);
        writeShares<u8, u8>(key_as_bytes, party, N, d_lsbMask, 1);
        auto d_outMask = randomGEOnGpu<T>(N, bout);
        auto d_lsbCorr = (T *)gpuMalloc(2 * N * sizeof(T));
        keygenStTRKernel<<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, shift, N, d_inputMask, d_rHat, d_lsbMask, d_lsbCorr, d_outMask);
        writeShares<T, T>(key_as_bytes, party, 2 * N, d_lsbCorr, bout);
        gpuFree(d_inputMask);
        gpuFree(d_rHat);
        gpuFree(d_lsbMask);
        gpuFree(d_lsbCorr);
        return d_outMask;
    }

    template <typename T>
    T *genGPUStochasticTruncateKey(u8 **key_as_bytes, int party, int bin, int bout, int shift, int N, T *d_inputMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        auto d_trMask = genGPUStTRKey(key_as_bytes, party, bin, bin - shift, shift, N, d_inputMask, gaes, h_r);
        auto d_outputMask = genSignExtendKey(key_as_bytes, party, bin - shift, bout, N, d_trMask, gaes);
        gpuFree(d_trMask);
        return d_outputMask;
    }

    // Maxpool keygen stub
    template <typename T>
    T *gpuKeygenMaxpool(u8 **key_as_bytes, int party, MaxpoolParams p, T *d_inputMask, u8 *d_oneHotMask, AESGlobalContext *gaes)
    {
        // TODO: Implement if used
        assert(0 && "gpuKeygenMaxpool not yet implemented");
        return nullptr;
    }

} // namespace dpf_dcf
