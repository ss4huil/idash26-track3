// lss_gpu_bridge.h — LSS ↔ GPU 管线胶合层（P3，设计文档 §5.2 最小侵入桥接）
//
// ⚠️ 只由 nvcc 编译的 GPU 管线包含（依赖 GPU-MPC utils 的 moveToGPU /
// moveIntoCPUMem 与 CUDA 类型）；纯 CPU 的 lss_test 不包含本文件，
// 保持 lss 模块本身零 CUDA 依赖。
//
// 桥接语义（bw=64，InfType=u64 与 LSS 的 uint64_t 直接对应）：
//   eval 侧：masked-public m（GPU）→ D2H → 转 share（读 MASK_IN）→
//            LSS relu（CPU）→ 加 MASK_OUT 重构（+1 轮）→ masked-public
//            m' → H2D 回 GPU；
//   keygen 侧：in.d_data 即输入 mask r（dealer 全程跟踪 mask，两个
//            dealer 进程随机流确定性一致）→ 写 LSS 记录流 → 返回新输出
//            mask r' 的 GPU buffer（作为输出张量的 mask 继续跟踪）。
#pragma once

#include "lss_online.h"
#include "lss_keygen.h"
#include "utils/gpu_mem.h"

#include <vector>

// eval：d_in = masked-public（GPU），返回 masked-public relu 输出（GPU）。
template <typename T>
T *lssReluEvalGpu(lss::LssParty *lss, T *d_in, u64 n, int bw, Stats *s)
{
    static_assert(sizeof(T) == sizeof(uint64_t), "LSS 桥接仅支持 bw=64");
    std::vector<uint64_t> h_m(n), h_out(n);
    moveIntoCPUMem((u8 *)h_m.data(), (u8 *)d_in, n * sizeof(T), s);
    lss->relu_bridged(h_m.data(), h_out.data(), (size_t)n, bw);
    return (T *)moveToGPU((u8 *)h_out.data(), n * sizeof(T), s);
}

// keygen：d_inMask = 输入 mask（GPU），追加 LSS 记录流，
// 返回输出 mask r'（GPU）——与 gpuGenReluKey 返回 d_reluMask 同款约定。
template <typename T>
T *lssReluKeygenGpu(lss::LssKeygen *kg, T *d_inMask, u64 n, int bw)
{
    static_assert(sizeof(T) == sizeof(uint64_t), "LSS 桥接仅支持 bw=64");
    std::vector<uint64_t> h_r(n), h_ro(n);
    moveIntoCPUMem((u8 *)h_r.data(), (u8 *)d_inMask, n * sizeof(T),
                   (Stats *)NULL);
    kg->gen_relu_bridged(n, bw, h_r.data(), h_ro.data());
    return (T *)moveToGPU((u8 *)h_ro.data(), n * sizeof(T), (Stats *)NULL);
}
