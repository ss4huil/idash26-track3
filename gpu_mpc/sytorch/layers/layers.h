// Shadow header for sytorch/layers/layers.h
// Fixes d_data propagation bugs in upstream functional layers.
// Renames the original broken classes via macros, then defines fixed versions.

#pragma once

// Prevent the original definitions from being used (rename them)
#define _MatMul _MatMul_ORIG
#define _Mul _Mul_ORIG
#define View View_ORIG
#define Unsqueeze Unsqueeze_ORIG
#define Concat Concat_ORIG

#include_next <sytorch/layers/layers.h>

#undef _MatMul
#undef _Mul
#undef View
#undef Unsqueeze
#undef Concat

// ── Fix 1: _MatMul doesn't copy d_data from backend matmul result ──────────
template <typename T>
class _MatMul : public Layer<T>
{
public:
    _MatMul() : Layer<T>("_MatMul")
    {
        this->doTruncationForward = true;
    }

    // Detect batched GCN aggregation: A_hat is (B*Nmax, Nmax), X is (B*Nmax, K).
    // Standard matmul needs shape0[1]==shape1[0] (Nmax==B*Nmax) which only holds
    // for B=1. Batched: shape0[0]==shape1[0] (both B*Nmax), shape0 square-per-block.
    static bool isBatched(const std::vector<u64> &s0, const std::vector<u64> &s1)
    {
        return s0.size() == 2 && s1.size() == 2 &&
               s0[0] == s1[0] && s0[1] < s0[0] && (s0[0] % s0[1]) == 0 &&
               s0[1] != s1[0];
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        if (isBatched(shape0, shape1)) return;   // batched GCN block-matmul
        always_assert(shape0[1] == shape1[0]);
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in matmul");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 2);
        auto &a0 = *a[0]; auto a0_2d = a0.as_2d();
        auto &a1 = *a[1]; auto a1_2d = a1.as_2d();
        auto act_2d = this->activation.as_2d();

        std::vector<u64> s0 = {a0_2d.d1, a0_2d.d2};
        std::vector<u64> s1 = {a1_2d.d1, a1_2d.d2};
        if (isBatched(s0, s1)) {
            // Per-sample loop: B blocks of (Nmax,Nmax) @ (Nmax,K) → (Nmax,K).
            // Both keygen and eval walk this node once and loop B times, so key
            // read/write stays in lockstep. Eval's revealIfLeaf matches the
            // per-slice A_hat/X pointers (registered as leaves in the driver).
            u64 Nmax = a0_2d.d2;
            u64 B = a0_2d.d1 / Nmax;
            u64 K = a1_2d.d2;

            // Folded output buffer (B*Nmax, K) — fill from per-slice results.
            T *folded_out = (T *)gpuMalloc(B * Nmax * K * sizeof(T));
            for (u64 b = 0; b < B; ++b) {
                Tensor2D<T> A_slice(a0_2d.data + b * Nmax * Nmax,
                                    a0_2d.d_data ? a0_2d.d_data + b * Nmax * Nmax : nullptr,
                                    Nmax, Nmax);
                Tensor2D<T> X_slice(a1_2d.data + b * Nmax * K,
                                    a1_2d.d_data ? a1_2d.d_data + b * Nmax * K : nullptr,
                                    Nmax, K);
                Tensor2D<T> out_slice(act_2d.data + b * Nmax * K, nullptr, Nmax, K);
                this->backend->matmul(A_slice, X_slice, out_slice);
                if (out_slice.d_data) {
                    checkCudaErrors(cudaMemcpy(folded_out + b * Nmax * K, out_slice.d_data,
                                               Nmax * K * sizeof(T), cudaMemcpyDeviceToDevice));
                    gpuFree(out_slice.d_data);
                }
            }
            this->activation.d_data = folded_out;
        } else {
            this->backend->matmul(a0_2d, a1_2d, act_2d);
            this->activation.d_data = act_2d.d_data;  // ← FIX: copy GPU ptr from view
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        if (isBatched(shape0, shape1)) return {shape0[0], shape1[1]};  // (B*Nmax, K)
        always_assert(shape0[1] == shape1[0]);
        return {shape0[0], shape1[1]};
    }
};

// ── Fix 2: View doesn't compute d_data offset ──────────────────────────────
template <typename T>
class View : public Layer<T>
{
public:
    i64 idx;
    View(i64 idx) : Layer<T>("View"), idx(idx) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        u64 i = (idx + a.shape[0]) % a.shape[0];
        auto v = a.view(i);
        this->activation.copy(v, false);
        // Allocate own GPU buffer via pool (match EzPC allocation/free pair).
        // Use cudaMallocAsync + cudaFreeAsync consistently.
        if (a.d_data != nullptr) {
            u64 rowsize = a.size() / a.shape[0];
            T *src = a.d_data + i * rowsize;
            this->activation.d_data = (T *)gpuMalloc(rowsize * sizeof(T));
            checkCudaErrors(cudaMemcpyAsync(this->activation.d_data, src,
                                            rowsize * sizeof(T),
                                            cudaMemcpyDeviceToDevice, 0));
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        if (inShapes[0].size() < 1)
            printshape(inShapes[0]);
        shape.erase(shape.begin());
        return shape;
    }
};

// ── Fix 3: Unsqueeze doesn't forward d_data ────────────────────────────────
template <typename T>
class Unsqueeze : public Layer<T>
{
public:
    Unsqueeze() : Layer<T>("Unsqueeze") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        u64 sz = a.size();
        for (u64 i = 0; i < sz; i++)
        {
            this->activation.data[i] = a.data[i];
        }
        // Allocate and copy GPU memory (same reasoning as View)
        if (a.d_data != nullptr) {
            this->activation.d_data = (T *)gpuMalloc(sz * sizeof(T));
            checkCudaErrors(cudaMemcpyAsync(this->activation.d_data, a.d_data,
                                            sz * sizeof(T),
                                            cudaMemcpyDeviceToDevice, 0));
            checkCudaErrors(cudaStreamSynchronize(0));
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto inShape = inShapes[0];
        inShape.insert(inShape.begin(), 1);
        return inShape;
    }
};

// ── Fix 4: Concat doesn't upload d_data after host-side copy ───────────────
template <typename T>
class Concat : public Layer<T>
{
public:
    Concat() : Layer<T>("Concat") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        auto &shape0 = shapes[0];
        for (auto &shape : shapes)
        {
            for (u64 i = 0; i < shape.size() - 1; i++)
            {
                always_assert(shape[i] == shape0[i]);
            }
        }
    }

    void _forward(std::vector<Tensor<T> *> &arr)
    {
        u64 outchannels = 0;
        u64 sz = 0;
        for (auto &t : arr)
        {
            outchannels += t->shape.back();
            sz += t->size();
        }

        // GPU-native fast path: if all inputs have d_data AND each input is a single
        // row (size()==shape.back()), then contiguous copy == last-axis interleave.
        // (For multi-row inputs the two orders differ, so fall back to host path.)
        bool all_gpu = true;
        for (auto &t : arr)
            if (t->d_data == nullptr || t->size() != t->shape.back())
            { all_gpu = false; break; }

        if (all_gpu)
        {
            // Allocate output GPU buffer
            this->activation.d_data = (T *)gpuMalloc(sz * sizeof(T));
            T *dst = this->activation.d_data;

            // Copy each input's GPU data into the output buffer
            for (auto &t : arr)
            {
                u64 t_sz = t->size();
                checkCudaErrors(cudaMemcpyAsync(dst, t->d_data,
                                                t_sz * sizeof(T),
                                                cudaMemcpyDeviceToDevice, 0));
                dst += t_sz;
            }
            checkCudaErrors(cudaStreamSynchronize(0));

            // Still populate host-side data for framework compatibility (graphGen reads it)
            u64 offset = 0;
            for (auto &t : arr)
            {
                u64 t_sz = t->size();
                std::memcpy(this->activation.data + offset, t->data, t_sz * sizeof(T));
                offset += t_sz;
            }
        }
        else
        {
            // Original host-side path
#pragma omp parallel for
            for (int i = 0; i < sz; ++i)
            {
                u64 l = i % outchannels;
                u64 rest = i / outchannels;
                for (auto &a : arr)
                {
                    if (l < a->shape.back())
                    {
                        this->activation.data[i] = a->data[rest * a->shape.back() + l];
                        break;
                    }
                    l -= a->shape.back();
                }
            }

            // Upload concatenated host data to GPU
            this->activation.d_data = (T *)moveToGPU((u8 *)this->activation.data, sz * sizeof(T), NULL);
        }
    }

    void _forward(Tensor<T> &a)
    {
        this->activation.copy(a, false);
        // Allocate and copy GPU memory
        if (a.d_data != nullptr) {
            u64 sz = a.size();
            this->activation.d_data = (T *)gpuMalloc(sz * sizeof(T));
            checkCudaErrors(cudaMemcpyAsync(this->activation.d_data, a.d_data,
                                            sz * sizeof(T),
                                            cudaMemcpyDeviceToDevice, 0));
            checkCudaErrors(cudaStreamSynchronize(0));
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        auto &shape0 = inShapes[0];
        for (auto &shape : inShapes)
        {
            for (u64 i = 0; i < shape.size() - 1; i++)
            {
                always_assert(shape[i] == shape0[i]);
            }
        }

        std::vector<u64> outShape = shape0;
        outShape.back() = 0;
        for (auto &shape : inShapes)
        {
            outShape.back() += shape.back();
        }
        return outShape;
    }
};

// ── Fix 5: _Mul double-truncates. Backend gpuMul truncates internally (24→12);
// upstream _Mul ALSO sets doTruncationForward=true (12→0), destroying precision.
// Disable the layer-level truncation so output stays at scale 12. Paired with
// ddg_orca_opt.h setting _Mul mode=0 (no downstream sign-extension). ────────────
template <typename T>
class _Mul : public Layer<T>
{
public:
    _Mul() : Layer<T>("_Mul")
    {
        this->doTruncationForward = false;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == shape1.size());
        for (size_t i = 0; i < shape0.size(); i++) {
            always_assert(shape0[i] == shape1[i]);
        }
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in mul");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 2);
        this->backend->mul(*a[0], *a[1], this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == shape1.size());
        for (size_t i = 0; i < shape0.size(); i++) {
            always_assert(shape0[i] == shape1[i]);
        }
        return shape0;
    }
};

// ── NEW: _Sub for pure ring subtraction (b - a) without truncation ─────────
// Used by MaskedGlobalMaxPool::sub() to avoid the _ScalarMul + truncate overhead.
// Computes b - a via gpuLinearComb (local, no comm, no keys). Output stays at
// input scale (no doTruncationForward). Backend::sub() uses tmpBw = bw - scale
// to match Add's modulus semantics (Sigma full-32 mode with scale 12 → mod 2^20).
template <typename T>
class _Sub : public Layer<T>
{
public:
    _Sub() : Layer<T>("_Sub")
    {
        this->doTruncationForward = false;  // Pure ring op, no truncate
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == shape1.size());
        for (size_t i = 0; i < shape0.size(); i++) {
            always_assert(shape0[i] == shape1[i]);
        }
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in sub");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 2);
        // sub(b, a) computes b - a
        this->backend->sub(*a[0], *a[1], this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == shape1.size());
        for (size_t i = 0; i < shape0.size(); i++) {
            always_assert(shape0[i] == shape1[i]);
        }
        return shape0;
    }
};
