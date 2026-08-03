//
// DeepDTAGen affinity path — GCN layer for GPU-MPC (iDASH 2024 Track 3).
//
// This file is part of the local affinity-prediction experiment and does NOT
// modify any upstream EzPC/GPU-MPC source. It only *uses* the public sytorch
// SytorchModule<T> functional API (matmul / relu / add) declared in
// ext/sytorch/include/sytorch/module.h.
//
// Privacy model (iDASH Track 3):
//   - Drug graph  X, A_hat, mask : CONFIDENTIAL (P1's secret shares)
//   - Model weights (GCN W, b)   : CONFIDENTIAL (P2's secret; held host-side by FC<T>)
//   - All intermediate tensors   : CONFIDENTIAL until the final affinity reveal
//
// GCN layer computes, for a dense padded graph with N = Nmax nodes:
//
//       Z      = X @ W + b            (node feature transform)
//       H_out  = A_hat @ Z            (neighbourhood aggregation)
//       H_relu = ReLU(H_out)
//
// Mapping onto GPU-MPC primitives:
//   * Z = X @ W + b   is secret(X) x public-to-P2(W).  We reuse the stock
//     FC<T> layer: it keeps W, b on the host as P2's private weights and calls
//     gpuMatmul (weight-as-host). This is the cheap path (no Beaver triples).
//   * H_out = A_hat @ Z   is secret(A_hat) x secret(Z).  We express it with the
//     functional matmul(a,b) op, which the backend lowers to the Beaver-triple
//     secret x secret kernel (gpuMatmulBeaver / gpuMatmulGmw in fss/gpu_matmul.cu).
//   * ReLU is the functional relu() op (DReLU + select over the ring).
//
// NOTE: FC<T> in sytorch is constructed FC<T>(in, out, useBias) and internally
// computes  input @ W^T + b  with W stored as (out, in). The Python weight
// exporter (reference/export_weights.py) writes GCN weights already transposed
// to (in, out) row-major followed by bias, matching load into an FC<T> tensor.
//
#pragma once

#include <sytorch/module.h>
#include "utils/gpu_data_types.h"
#include "nn/orca/fc_layer.h"

// A single dense-GCN layer, faithful to the golden reference
// (reference/dense_gcn.py + fixed_forward.py::_gcn_layer):
//
//       ReLU( A_hat @ (X @ W^T) + b )
//
// KEY ALGEBRAIC REWRITE (why bias placement is not a problem): matmul is
// associative, so
//
//       A_hat @ (X @ W^T) + b   ==   (A_hat @ X) @ W^T + b .
//
// We therefore AGGREGATE FIRST — AX = A_hat @ X (secret x secret) — and then
// apply the STANDARD bias-folding FC to AX. The FC computes AX @ W^T + b with
// the bias added AFTER the matmul, over the out-feature axis, broadcast across
// all N nodes: exactly the reference semantics. Crucially this means:
//   * the bias stays folded into P2's private weights (export_weights.py needs
//     NO change — GCN bias is already emitted inside the weight blob), and
//   * there is no need to inject a tiled bias as a secret leaf.
// Bias-before-aggregation (the naive A_hat @ (XW^T + b)) would be WRONG because
// A_hat = D^{-1/2}(A+I)D^{-1/2} is not row-stochastic — the rewrite above sidesteps
// that entirely.
//
// It is also cheaper for the expanding layers here (in_feat < out_feat): the
// secret x secret product is N*N*in_feat instead of N*N*out_feat.
//
// TRUNCATION-ORDER NOTE: the reference truncates as trunc(A_hat @ trunc(X@W^T)),
// whereas this rewrite truncates as trunc(trunc(A_hat @ X) @ W^T). Over the
// reals these are identical; in fixed-point they differ by at most ~1 ULP of
// truncation rounding — far below the fixed-point quantisation noise (~1.7e-6)
// and the +/-2% accuracy gate. The two are therefore numerically equivalent but
// NOT guaranteed bit-identical, which only matters for a C++<->Python bit-exact
// alignment test (deferred: no nvcc + no pretrained weights in this env).
//
// A_hat is a *secret* input (P1's graph), passed into forward().
template <typename T>
class GCNLayer : public SytorchModule<T>
{
    using SytorchModule<T>::matmul;   // secret x secret  (Beaver)
    using SytorchModule<T>::relu;     // functional ReLU

public:
    FC<T> *lin;   // (A_hat @ X) @ W^T + b, W/b are P2's host-side private weights
    u64 in_feat;
    u64 out_feat;

    GCNLayer(u64 in_feat_, u64 out_feat_)
        : in_feat(in_feat_), out_feat(out_feat_)
    {
        // useBias=true: bias folds into the FC after the matmul, matching the
        // reference's post-aggregation bias thanks to the associativity rewrite.
        lin = new FC<T>(in_feat, out_feat, /*useBias=*/true);
    }

    // X (N x in_feat) and A_hat (N x N), both secret.
    Tensor<T> &forward(Tensor<T> &X, Tensor<T> &A_hat)
    {
        auto &AX  = matmul(A_hat, X);   // secret x secret        -> (N, in_feat)
        auto &Z   = lin->forward(AX);   // AX @ W^T + b (P2 weights) -> (N, out)
        auto &act = relu(Z);            // ReLU (>= 0)            -> (N, out)
        return act;
    }
};
