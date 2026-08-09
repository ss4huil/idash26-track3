//
// MaskedMaxPoolLayer — fused tree-reduction max-pool (Path B+A optimization).
//
// Collapses the 137-step sequential fold into a single Layer node whose _forward
// implements tree reduction with batched backend crypto calls (relu/truncate).
// Both keygen and eval walk identical tree structure → keys align perfectly.
//
// Input:  masked (B*Nmax, F) — sample-major, already H ⊙ mask
// Output: pooled (B, F)       — max over Nmax nodes per sample
//
// Performance: ~8 crypto rounds (log₂ 138) vs 137 sequential, plus ~3 framework
// nodes total vs 1233. Targets the measured 9.5s framework overhead bottleneck.
//
#pragma once

#include <sytorch/layers/layers.h>
#include "utils/gpu_data_types.h"
#include "utils/gpu_comms.h"
#include <vector>
#include <cmath>

template <typename T>
class MaskedMaxPoolLayer : public Layer<T>
{
public:
    u64 B;      // batch size
    u64 Nmax;   // nodes per sample
    u64 F;      // feature channels

    MaskedMaxPoolLayer(u64 B_, u64 Nmax_, u64 F_)
        : Layer<T>("MaskedMaxPoolLayer"), B(B_), Nmax(Nmax_), F(F_) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) override
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[0] == B * Nmax && shape[1] == F);
    }

    // Tree reduction: pair and fold log₂(Nmax) levels, batching independent
    // pairwise-max ops per level into single backend calls.
    // Crypto call sequence per pair (matching functional pairwise_max exactly):
    //   sub:  scalarmul(even, -1/(1<<scale)) → truncate → add(result, odd)
    //   relu: backend->relu(diff, ...)
    //   max:  add(even, relu_out)
    void _forward(Tensor<T> &masked) override
    {
        u64 planeSize = B * F;  // elements per plane (one node's data for all samples)

        // Transpose from sample-major (B*Nmax, F) to node-major buffer (Nmax planes).
        // Each plane i: [ sample0_node_i[F], sample1_node_i[F], ..., sampleB-1_node_i[F] ]
        T *d_buf = (T *)gpuMalloc(Nmax * planeSize * sizeof(T));
        for (u64 b = 0; b < B; ++b) {
            // Copy sample b's Nmax rows (each F wide) into planes, scattering across buffer.
            // src: masked.d_data + b*Nmax*F  (sample b's block)
            // dst: d_buf + b*F               (sample b's slot in each plane, stride Nmax*F)
            checkCudaErrors(cudaMemcpy2D(
                d_buf + b * F,                      // dst base
                planeSize * sizeof(T),              // dst pitch (jump B*F per row)
                masked.d_data + b * Nmax * F,       // src base
                F * sizeof(T),                      // src pitch (contiguous F per row)
                F * sizeof(T),                      // width per row
                Nmax,                               // num rows
                cudaMemcpyDeviceToDevice));
        }

        // Tree fold: reduce Nmax planes → 1 plane via log₂ levels.
        u64 curCount = Nmax;
        T *d_cur = d_buf;

        while (curCount > 1) {
            u64 pairs = curCount / 2;
            u64 remainder = curCount % 2;

            if (pairs > 0) {
                // Gather even-indexed planes into d_even, odd into d_odd.
                T *d_even = (T *)gpuMalloc(pairs * planeSize * sizeof(T));
                T *d_odd  = (T *)gpuMalloc(pairs * planeSize * sizeof(T));

                for (u64 p = 0; p < pairs; ++p) {
                    checkCudaErrors(cudaMemcpy(
                        d_even + p * planeSize,
                        d_cur  + (2*p) * planeSize,
                        planeSize * sizeof(T),
                        cudaMemcpyDeviceToDevice));
                    checkCudaErrors(cudaMemcpy(
                        d_odd + p * planeSize,
                        d_cur + (2*p+1) * planeSize,
                        planeSize * sizeof(T),
                        cudaMemcpyDeviceToDevice));
                }

                // pairwise_max(even, odd) = even + ReLU(odd - even)
                // Replicate functional sub/relu/add sequence with backend calls.
                u64 batchSize = pairs * planeSize;

                // sub(odd, even): scalarmul(even, -1/(1<<scale)) + odd.
                // scalarFix = -1 is an EXACT ring element (negation), NOT a
                // fixed-point scalar, so it must NOT be followed by truncateForward.
                // The functional path (masked_maxpool.h MaskedGlobalMaxPool::sub)
                // does ring negation with no truncation; adding a truncate here
                // injected a spurious floor(x/2^scale) rounding per tree level,
                // which compounded across log2(Nmax) levels and corrupted the
                // discriminative (high-affinity) outputs. Keep it truncation-free.
                Tensor<T> t_even(nullptr, d_even, {batchSize});
                Tensor<T> t_scalarmul_out({batchSize});  // Layer::resize allocates host data
                T scalarFix = (T)(-1.0 / (1LL << this->scale) * (1LL << this->scale)); // = -1
                this->backend->scalarmul(t_even, scalarFix, t_scalarmul_out);

                // add(t_scalarmul_out, odd) → diff
                Tensor<T> t_odd(nullptr, d_odd, {batchSize});
                std::vector<Tensor<T>*> add_in = {&t_scalarmul_out, &t_odd};
                Tensor<T> t_diff({batchSize});
                this->backend->add(add_in, t_diff);

                // relu(diff) → drelu (unused), relu_out
                Tensor<T> t_drelu({batchSize});  // shape-only placeholder for keygen assert
                Tensor<T> t_relu_out({batchSize});
                this->backend->relu(t_diff, t_relu_out, t_drelu, this->scale, this->mode);

                // add(even, relu_out) → result
                std::vector<Tensor<T>*> add_in2 = {&t_even, &t_relu_out};
                Tensor<T> t_result({batchSize});
                this->backend->add(add_in2, t_result);

                // Free intermediate GPU buffers (d_even/d_odd freed via scoped tensors' dtors? No — non-owner)
                gpuFree(d_even);
                gpuFree(d_odd);
                gpuFree(t_scalarmul_out.d_data);
                gpuFree(t_diff.d_data);
                gpuFree(t_drelu.d_data);  // likely null from backend
                gpuFree(t_relu_out.d_data);

                // Write result back to first `pairs` planes of d_cur, handling remainder.
                T *d_next = (T *)gpuMalloc((pairs + remainder) * planeSize * sizeof(T));
                checkCudaErrors(cudaMemcpy(d_next, t_result.d_data, pairs * planeSize * sizeof(T),
                                           cudaMemcpyDeviceToDevice));
                gpuFree(t_result.d_data);

                if (remainder) {
                    // Copy unpaired last plane forward.
                    checkCudaErrors(cudaMemcpy(
                        d_next + pairs * planeSize,
                        d_cur  + (curCount - 1) * planeSize,
                        planeSize * sizeof(T),
                        cudaMemcpyDeviceToDevice));
                }

                gpuFree(d_cur);
                d_cur = d_next;
                curCount = pairs + remainder;
            } else {
                // Only remainder=1 plane, done.
                break;
            }
        }

        // Final plane is (B, F) — set as activation's d_data.
        this->activation.d_data = d_cur;

        // Copy to host for activation.data (sytorch contract).
        checkCudaErrors(cudaMemcpy(this->activation.data, d_cur,
                                   B * F * sizeof(T), cudaMemcpyDeviceToHost));
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) override
    {
        always_assert(inShapes.size() == 1);
        return {B, F};
    }
};
