// Shadow header for sytorch/layers/layers.h
// Fixes d_data propagation bugs in upstream functional layers.
// Renames the original broken classes via macros, then defines fixed versions.

#pragma once

// Prevent the original definitions from being used (rename them)
#define _MatMul _MatMul_ORIG
#define View View_ORIG
#define Unsqueeze Unsqueeze_ORIG
#define Concat Concat_ORIG

#include_next <sytorch/layers/layers.h>

#undef _MatMul
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

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
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
        this->backend->matmul(a0_2d, a1_2d, act_2d);
        this->activation.d_data = act_2d.d_data;  // ← FIX: copy GPU ptr from view
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
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
        // Tensor::view() uses (data,shape) ctor leaving d_data=nullptr
        // Compute GPU pointer offset manually: row i at a.d_data + i * rowsize
        u64 rowsize = a.size() / a.shape[0];
        this->activation.d_data = (a.d_data != nullptr) ? (a.d_data + i * rowsize) : nullptr;
        this->activation.isOwner = false;  // aliased pointer — don't free
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
        this->activation.d_data = a.d_data;  // ← FIX: forward GPU ptr
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

        // ← FIX: Upload concatenated host data to GPU
        this->activation.d_data = (T *)moveToGPU((u8 *)this->activation.data, sz * sizeof(T), NULL);
    }

    void _forward(Tensor<T> &a)
    {
        this->activation.copy(a, false);
        this->activation.d_data = a.d_data;  // ← FIX: forward GPU ptr
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
