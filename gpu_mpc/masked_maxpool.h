//
// DeepDTAGen affinity path — masked global max-pool for GPU-MPC.
//
// Reduces the per-node GCN features H (Nmax x F) to a single graph embedding
// (1 x F): for each feature channel, the maximum over the REAL nodes only,
// padding nodes excluded via the secret binary mask. Faithful to the golden
// reference reference/masked_maxpool.py / fixed_forward.py (sentinel select
// then column-wise max).
//
// Equivalence note: the reference forces padding rows to a NEG sentinel then
// takes the max. Since the preceding GCN layer applies ReLU, every real entry
// is >= 0, so zeroing padding rows (mask-multiply) is EXACTLY equivalent:
// padding contributes 0, which can only tie the max when the true masked max is
// also 0. We use mask-multiply because it is a single Hadamard product.
//
// SHAPE CONTRACTS discovered from sytorch source (layers/layers.h):
//   * _Mul asserts identical shapes on every axis — NO broadcast. Hence the
//     mask must arrive pre-tiled to (Nmax x F), not (Nmax x 1). The Python
//     sharer (share_data.py) emits maskTiled at the model scale.
//   * View(idx) selects row idx AND erases axis 0, yielding a 1-D (F,) tensor.
//     The pairwise fold therefore runs on 1-D vectors (add/relu/scalarmul are
//     all shape-preserving elementwise ops, so this is fine), and we unsqueeze
//     the (F,) result back to (1 x F) before the downstream FC, whose _resize
//     hard-asserts a 2-D input.
//
// The reduction is a STATIC left-fold of pairwise maxima, unrolled at graph
// build time (the sytorch graph is data-independent; a fixed-trip-count loop is
// legal). Cost: Nmax-1 ReLUs over an F-vector.
//     max(a, b) = a + ReLU(b - a)
//
// No upstream file is modified; this only composes the public functional API.
//
#pragma once

#include <sytorch/module.h>
#include "utils/gpu_data_types.h"

template <typename T>
struct MaskedGlobalMaxPool
{
    SytorchModule<T> *owner;   // supplies functional ops so nodes join its graph
    u64 Nmax;
    u64 F;

    MaskedGlobalMaxPool(SytorchModule<T> *owner_, u64 Nmax_, u64 F_)
        : owner(owner_), Nmax(Nmax_), F(F_) {}

    // b - a  via  b + (-1)*a
    Tensor<T> &sub(Tensor<T> &b, Tensor<T> &a)
    {
        auto &nega = owner->scalarmul(a, -1.0);
        return owner->add(b, nega);
    }

    // max(a, b) = a + ReLU(b - a), elementwise over the (F,) vectors.
    Tensor<T> &pairwise_max(Tensor<T> &a, Tensor<T> &b)
    {
        auto &d = sub(b, a);
        auto &r = owner->relu(d);
        return owner->add(a, r);
    }

    // H:        (Nmax x F) secret post-ReLU GCN features
    // maskTiled: (Nmax x F) secret {0,1} node mask tiled across F channels,
    //            carrying 1.0 in the same fixed-point scale as H.
    // Returns (1 x F) graph embedding.
    Tensor<T> &forward(Tensor<T> &H, Tensor<T> &maskTiled)
    {
        auto &masked = owner->mul(H, maskTiled);   // (Nmax x F), padding rows -> 0

        Tensor<T> *acc = &owner->view(masked, 0);  // (F,)
        for (u64 i = 1; i < Nmax; ++i)
        {
            auto &row = owner->view(masked, (i64)i);   // (F,)
            acc = &pairwise_max(*acc, row);            // (F,)
        }
        return owner->unsqueeze(*acc);   // (F,) -> (1 x F) for downstream FC
    }
};
