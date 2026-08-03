//
// DeepDTAGen affinity path — full model for GPU-MPC (iDASH 2024 Track 3).
//
// Scope: ONLY the drug-target affinity PREDICTION path. The drug-generation
// branch (VAE + Transformer decoder) is intentionally excluded.
//
// This mirrors, layer-for-layer, the Python fixed-point reference
// (idash/mpc/reference/fixedpoint.py). The two MUST stay in lockstep: the
// exported weight blob (reference/export_weights.py) is laid out in this exact
// forward order, and the accuracy gate is validated against this graph.
//
// Layout (Nmax = 138 padded nodes, FEAT_DIM = 94):
//
//   Drug branch  (secret, MPC):
//     GCN1 : ReLU(A_hat @ (X   @ W1 + b1))   94  -> 188
//     GCN2 : ReLU(A_hat @ (H1  @ W2 + b2))   188 -> 282
//     GCN3 : ReLU(A_hat @ (H2  @ W3 + b3))   282 -> 376
//     pool : masked global max over nodes    (138 x 376) -> (1 x 376)
//     dfc1 : ReLU(p   @ Wd1 + bd1)           376  -> 1024
//     dfc2 :      d1  @ Wd2 + bd2            1024 -> 128   (drug embedding)
//
//   Protein branch (public sequence): GatedCNN evaluated OUTSIDE MPC by P2,
//     entering here as a 128-vector secret share (proteinEmb).
//
//   Fusion branch (secret, MPC):
//     concat[d2, proteinEmb]                 -> 256
//     ffc1 : ReLU(. @ Wf1 + bf1)             256  -> 1024
//     ffc2 : ReLU(. @ Wf2 + bf2)             1024 -> 512
//     ffc3 : ReLU(. @ Wf3 + bf3)             512  -> 256
//     out  :      . @ Wo  + bo               256  -> 1     (affinity, revealed)
//
#pragma once

#include <sytorch/module.h>
#include "utils/gpu_data_types.h"
#include "nn/orca/fc_layer.h"
#include "nn/orca/relu_layer.h"

#include "gcn_layer.h"
#include "masked_maxpool.h"

#ifndef DDG_NMAX
#define DDG_NMAX 138
#endif
#ifndef DDG_FEAT
#define DDG_FEAT 94
#endif

template <typename T>
class DeepDTAGenAffinity : public SytorchModule<T>
{
    using SytorchModule<T>::concat;
    using SytorchModule<T>::view;

public:
    // Drug GCN stack (weights are P2's private, held host-side by FC<T>).
    GCNLayer<T> *gcn1;
    GCNLayer<T> *gcn2;
    GCNLayer<T> *gcn3;

    // Drug embedding FCs.
    FC<T> *dfc1; ReLU<T> *drelu1;
    FC<T> *dfc2;

    // Fusion FCs.
    FC<T> *ffc1; ReLU<T> *frelu1;
    FC<T> *ffc2; ReLU<T> *frelu2;
    FC<T> *ffc3; ReLU<T> *frelu3;
    FC<T> *fout;

    u64 Nmax  = DDG_NMAX;
    u64 feat  = DDG_FEAT;

    // Secret side-inputs, set per-sample before forward (see setSample()).
    //  * A_hat, maskTiled : P1's drug-graph secrets.
    //  * proteinEmb       : P2's public GatedCNN output injected as a share.
    // GCN biases fold into P2's private FC weights (associativity rewrite in
    // gcn_layer.h), so no tiled bias leaf is needed. maskTiled carries {0,1} at
    // the model fixed-point scale (share_data.py).
    Tensor<T> *A_hat       = nullptr;   // (Nmax x Nmax)
    Tensor<T> *maskTiled   = nullptr;   // (Nmax x 376)
    Tensor<T> *proteinEmb  = nullptr;   // (1 x 128)

    DeepDTAGenAffinity()
    {
        gcn1 = new GCNLayer<T>(feat, 188);
        gcn2 = new GCNLayer<T>(188, 282);
        gcn3 = new GCNLayer<T>(282, 376);

        dfc1 = new FC<T>(376, 1024, true);  drelu1 = new ReLU<T>();
        dfc2 = new FC<T>(1024, 128, true);

        ffc1 = new FC<T>(256, 1024, true);  frelu1 = new ReLU<T>();
        ffc2 = new FC<T>(1024, 512, true);  frelu2 = new ReLU<T>();
        ffc3 = new FC<T>(512, 256, true);   frelu3 = new ReLU<T>();
        fout = new FC<T>(256, 1, true);
    }

    void setSample(Tensor<T> *A_hat_, Tensor<T> *maskTiled_, Tensor<T> *proteinEmb_)
    {
        A_hat = A_hat_;
        maskTiled = maskTiled_;
        proteinEmb = proteinEmb_;
    }

    // input = X, the (Nmax x FEAT) padded node-feature matrix (P1's secret).
    // Side-inputs must have been set via setSample().
    Tensor<T> &_forward(Tensor<T> &X)
    {
        // ---- drug GCN stack (aggregate-first; bias folds into FC) ----
        auto &h1 = gcn1->forward(X,  *A_hat);   // (Nmax x 188)
        auto &h2 = gcn2->forward(h1, *A_hat);   // (Nmax x 282)
        auto &h3 = gcn3->forward(h2, *A_hat);   // (Nmax x 376)

        // ---- masked global max-pool over nodes ----
        MaskedGlobalMaxPool<T> pool(this, Nmax, 376);
        auto &pooled = pool.forward(h3, *maskTiled);   // (1 x 376)

        // ---- drug embedding ----
        auto &d1  = dfc1->forward(pooled);       // (1 x 1024)
        auto &d1r = drelu1->forward(d1);
        auto &d2  = dfc2->forward(d1r);          // (1 x 128) drug embedding

        // ---- fusion ----
        auto &fused = concat(d2, *proteinEmb);   // (1 x 256)
        auto &f1 = frelu1->forward(ffc1->forward(fused));   // (1 x 1024)
        auto &f2 = frelu2->forward(ffc2->forward(f1));      // (1 x 512)
        auto &f3 = frelu3->forward(ffc3->forward(f2));      // (1 x 256)
        auto &out = fout->forward(f3);                      // (1 x 1) affinity
        return out;
    }
};
