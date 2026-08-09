// Forked from EzPC/GPU-MPC nn/orca_opt.h (commit 66d9cddc, Aug 2024)
// Extended for DeepDTAGen graph model: added mode-propagation cases for
// _MatMul, _Mul, _ScalarMul, View, Unsqueeze (GCN + MaskedGlobalMaxPool ops).
//
// Original Authors: Neha Jawalkar
// Copyright (c) 2024 Microsoft Research
// Licensed under MIT (see original file header for full text).
//
// Modifications Copyright (c) 2026 iDASH Track 3 submission team.
// This forked version adds support for functional graph layers not present in
// the upstream CNN-focused optimizer.

#pragma once

#include <sytorch/tensor.h>
#include <sytorch/graph.h>
#include "utils/helper_cuda.h"

template <typename T>
void ddgOrcaOpt(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
{
    // ReLU-MaxPool optimization (unchanged from upstream)
    if (node->layer->name == "ReLU")
    {
        auto child = node->children[0];
        auto cLayer = node->children[0]->layer;
        if (node->children.size() == 1 && child->parents.size() == 1 && cLayer->name == "MaxPool2D")
        {
            child->layer = node->layer;
            node->layer = cLayer;

            cLayer->node = node;
            child->layer->node = child;

            node->currTensor = &(node->layer->activation);
            child->layer->resize({node->layer->activation.shape});
            child->currTensor = &(child->layer->activation);
        }
    }

    // Mode propagation rules (unchanged from upstream comments):
    // 0: layer takes ℓ bits input → ℓ bits output
    // 1: layer takes ℓ bits input → ℓ-scale bits output
    // 2: layer takes ℓ-scale bits input → ℓ bits output
    // 3: layer takes ℓ-scale bits input → ℓ-scale bits output

    if (node->layer->name == "Conv2D" || node->layer->name == "FC" || node->layer->name == "BatchNorm2dInference" || node->layer->name == "AvgPool2D" || node->layer->name == "GlobalAvgPool2D")
    {
        // Linear layer: check parent mode for sign-extension need
        auto parent = node->parents[0];
        // BUG FIX: Sigma truncateForward outputs FULL 32-bit, not reduced bitwidth.
        // When parent is _MatMul (which has doTruncationForward=true), its truncate
        // outputs 32-bit, so FC must NOT sign-extend from 20-bit → corruption.
        // Only enable doPreSignExtension when parent is ReLU/MaxPool (mode 1/3) and
        // NOT a matmul that truncates.
        bool parentIsTruncatingMatmul = (parent->layer->name == "_MatMul" && parent->layer->doTruncationForward);
        if ((parent->layer->mode == 1 || parent->layer->mode == 3) && !parentIsTruncatingMatmul)
        {
            node->layer->doPreSignExtension = true;
        }
        if (!node->layer->isTrainingMode && node->children.size() == 0)
        {
            // Last node: no truncation
            node->layer->mode = 0;
            node->layer->doTruncationForward = false;
        }
        else
        {
            // BUG FIX: Sigma truncateForward keeps FULL 32-bit (no bitwidth
            // reduction). Advertise mode=0 so downstream ReLU/MaxPool run at full
            // bw (bin=32), not the Orca reduced bw (bin=ℓ-scale=20) which would
            // misread the sign bit and zero the activation. Truncation still fires
            // via doTruncationForward (set true by the layer constructor).
            node->layer->mode = 0;
            node->layer->forwardTruncationMode = 1;
        }
    }
    // ── ADDED: secret×secret matmul (same as FC/Conv2D) ─────────────────────
    else if (node->layer->name == "_MatMul")
    {
        // Secret×secret matmul: same mode-propagation as FC/Conv2D.
        // Parent at reduced bitwidth (mode 1 or 3) → need pre-sign-extension.
        // Output: mode 1 (truncate forward) unless this is the terminal node.
        auto parent = node->parents[0];
        if (parent->layer->mode == 1 || parent->layer->mode == 3)
        {
            node->layer->doPreSignExtension = true;
        }
        if (!node->layer->isTrainingMode && node->children.size() == 0)
        {
            node->layer->mode = 0;
            node->layer->doTruncationForward = false;
        }
        else
        {
            // BUG FIX: Sigma truncate keeps full 32-bit → mode=0 (see FC block above)
            node->layer->mode = 0;
            node->layer->forwardTruncationMode = 1;
        }
    }
    // ── ADDED: element-wise secret×secret multiply (DOUBLE truncates to scale 0) ──
    else if (node->layer->name == "_Mul")
    {
        // Element-wise mul: backend uses TrWithSlack (24→12 internal truncation)
        // AND layer sets doTruncationForward=true (12→0 layer truncation).
        // Net result: scale 0 output. Set mode=0 (no bitwidth reduction, no
        // sign-extension needed downstream). This prevents mode/scale mismatch.
        node->layer->mode = 0;
    }
    // ── ADDED: public-scalar multiply ───────────────────────────────────────
    else if (node->layer->name == "_ScalarMul")
    {
        // Public-scalar multiply: converts scalar to fixed-point, multiplies
        // (scale 12 × scale 12 → scale 24), then truncates 24→12 via
        // doTruncationForward=true (set by upstream constructor). Output is
        // scale 12, same bitwidth reduction as FC/Conv2D → mode 1.
        auto parent = node->parents[0];
        if (parent->layer->mode == 1 || parent->layer->mode == 3)
        {
            node->layer->doPreSignExtension = true;
        }
        if (!node->layer->isTrainingMode && node->children.size() == 0)
        {
            node->layer->mode = 0;
            node->layer->doTruncationForward = false;
        }
        else
        {
            // BUG FIX: Sigma truncate keeps full 32-bit → mode=0 (see FC block above)
            node->layer->mode = 0;
            node->layer->forwardTruncationMode = 1;
        }
    }
    // ── ADDED: structural reshapes ──────────────────────────────────────────
    else if (node->layer->name == "View" || node->layer->name == "Unsqueeze")
    {
        // Structural reshape: pass through parent mode (no computation).
        node->layer->mode = node->parents[0]->layer->mode;
    }
    // ── ADDED: fused masked max-pool (Path B+A) ─────────────────────────────
    else if (node->layer->name == "MaskedMaxPoolLayer")
    {
        // Fused tree-reduction pool. Internally runs scalarmul→truncate→add→
        // relu→add per tree level, replicating the functional fold's crypto
        // sequence exactly. Its OUTPUT is the sum of an even-plane share and a
        // relu-share (both scale-12, full 32-bit under Sigma), i.e. mode 0 —
        // same as the functional fold's terminal Add node. Advertise mode 0 so
        // the downstream FC (dfc1) sees the identical mode it saw before and
        // does NOT sign-extend. Parent (_Mul mask-multiply) is already mode 0.
        node->layer->mode = 0;
    }
    // ─────────────────────────────────────────────────────────────────────────
    else if (node->layer->name == "Add" || node->layer->name == "Concat" || node->layer->name == "_Sub")
    {
        int m = 0;
        for (auto &parent : node->parents)
        {
            if ((parent->layer->mode % 2) == 1)
            {
                m = 3;
                break;
            }
        }
        node->layer->mode = m;
    }
    else if (node->layer->name == "Flatten")
    {
        // Delete flatten and add a flag to FC instead (unchanged from upstream)
        assert(node->parents.size() == 1 && node->children.size() == 1);
        auto parent = node->parents[0];
        auto child = node->children[0];
        assert(parent->children.size() == 1);
        assert(child->parents.size() == 1);
        parent->children[0] = child;
        child->parents[0] = parent;
        always_assert(parent->currTensor->shape.size() == 4);
        always_assert(child->layer->name == "FC");
        auto fc = static_cast<FC<T> *>(child->layer);
        auto batchSz = parent->currTensor->shape[0];
        auto h = parent->currTensor->shape[1];
        auto w = parent->currTensor->shape[2];
        auto c = parent->currTensor->shape[3];
        int m = fc->out;
        assert(h * w * c == fc->in);
        parent->currTensor = new Tensor<T>(parent->layer->activation.data, parent->layer->activation.d_data, {batchSz, h * w * c});
        parent->currTensor->graphNode = parent;
        auto temp = Tensor(fc->weight.data, {fc->weight.d1, fc->weight.d2});
        temp.copy(fc->weight.as_nd(), false);
        auto temp_as_2d = temp.as_2d();
        for (int l = 0; l < m; l++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    for (int k = 0; k < c; k++)
                    {
                        fc->weight(i * w * c + j * c + k, l) = temp_as_2d(k * h * w + i * w + j, l);
                    }
                }
            }
        }
        int i;
        for (i = 0; i < node->allNodesInExecutionOrderRef->size(); i++)
        {
            if (node->allNodesInExecutionOrderRef->at(i) == node)
            {
                break;
            }
        }
        node->allNodesInExecutionOrderRef->erase(node->allNodesInExecutionOrderRef->begin() + i);
    }
    else if (node->layer->name == "MaxPool2D")
    {
        auto parentMode = node->parents[0]->layer->mode;
        if (parentMode == 1 || parentMode == 3)
        {
            node->layer->mode = 3;
        }
        else
        {
            node->layer->mode = 0;
        }
    }
    else if (node->layer->name == "ReLU")
    {
        auto parentMode = node->parents[0]->layer->mode;
        if (parentMode == 0 || parentMode == 2)
        {
            node->layer->mode = 0;
        }
        else
        {
            bool oneChildLinear = false;
            for (auto &child : node->children)
            {
                if (child->layer->name == "Conv2D" || child->layer->name == "FC" || child->layer->name == "BatchNorm2dInference" || child->layer->name == "GlobalAvgPool2D" || child->layer->name == "AvgPool2D" || child->layer->name == "Flatten")
                {
                    oneChildLinear = true;
                    break;
                }
            }
            if (oneChildLinear)
            {
                node->layer->mode = 2;
            }
            else
            {
                node->layer->mode = 3;
            }
        }
    }
    else if (node->layer->name == "Input")
    {
        // Structural input marker: no mode change
    }
    else
    {
        throw std::runtime_error("Unknown layer type: " + node->layer->name);
    }
}

template <typename T>
void pinCpuMem(LayerGraphNode<T> *n, LayerGraphNode<T> *r)
{
    // Unchanged from upstream: pin all host tensors for faster GPU transfers
    if (n->currTensor->data)
        checkCudaErrors(cudaHostRegister(n->currTensor->data, n->currTensor->size() * sizeof(T), cudaHostRegisterDefault));
    auto w = n->layer->getweights().data;
    auto b = n->layer->getbias().data;
    auto wSz = n->layer->getweights().size;
    auto bSz = n->layer->getbias().size;
    if (w)
        checkCudaErrors(cudaHostRegister(w, wSz * sizeof(T), cudaHostRegisterDefault));
    if (b)
        checkCudaErrors(cudaHostRegister(b, bSz * sizeof(T), cudaHostRegisterDefault));
    if (n->layer->name == "_MHADummy")
    {
        auto mha = static_cast<_MHADummy<T> *>(n->layer);
        checkCudaErrors(cudaHostRegister(mha->wQKV.data, mha->wQKV.size() * sizeof(T), cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(mha->bQKV.data, mha->bQKV.size() * sizeof(T), cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(mha->wProj.data, mha->wProj.size() * sizeof(T), cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(mha->bProj.data, mha->bProj.size() * sizeof(T), cudaHostRegisterDefault));
    }
}
