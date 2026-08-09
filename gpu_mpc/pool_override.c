/*
 * pool_override.c — Linker wraps for GPU-MPC memory functions.
 *
 * 1. initGPUMemPool: replaces the upstream 40GiB probe (OOMs on 8GiB GPU).
 * 2. gpuMalloc: wraps cudaMallocAsync to zero-initialize GPU memory, eliminating
 *    nondeterministic keygen from uninitialized buffer residue.
 *
 * Linked via -Wl,--wrap=initGPUMemPool,--wrap=gpuMalloc in Makefile.
 */
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

void __wrap_initGPUMemPool(void)
{
    int device = 0;
    cudaMemPool_t mempool;

    cudaDeviceGetDefaultMemPool(&mempool, device);

    /* Release threshold controls how much freed pool memory is retained rather
     * than returned to the OS. threshold=0 returns every freed block to the OS
     * immediately — correct for avoiding fragmentation, but with the pool's ~1100
     * malloc/free cycles per forward (137 iterations × several ops), each free is
     * an OS release syscall and each subsequent malloc re-acquires from the OS,
     * costing several seconds of pure syscall/sync overhead.
     *
     * A moderate cap (default 384 MiB, override via DDG_POOL_RETAIN_MB) keeps the
     * small pooling working set cached in-pool (fast reuse, no OS round-trip) while
     * still releasing large blocks. The 512 MiB comm buffer is allocated ONCE at
     * startup (before the fold), so caching small blocks during the fold cannot
     * starve it. Set DDG_POOL_RETAIN_MB=0 to restore the old immediate-release. */
    uint64_t retain_mb = 384;
    const char *env = getenv("DDG_POOL_RETAIN_MB");
    if (env != NULL) retain_mb = (uint64_t)strtoull(env, NULL, 10);
    uint64_t threshold = retain_mb * 1024ULL * 1024ULL;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

    /* Skip the 40 GiB warm-up probe — it OOMs on 8 GiB GPUs.
     * Pool memory is committed on first actual allocation instead. */
    printf("[pool_override] initGPUMemPool: retain=%llu MiB, probe skipped (GPU<40GiB)\n",
           (unsigned long long)retain_mb);
}

/* __real_gpuMalloc is the original gpuMalloc from utils/gpu_mem.cu, provided
 * by the linker's --wrap mechanism. */
extern uint8_t *__real_gpuMalloc(size_t size_in_bytes);

/*
 * __wrap_gpuMalloc — zero-initialize every GPU allocation.
 *
 * WHY: gpuMalloc uses cudaMallocAsync, which returns pool memory WITHOUT
 * zeroing. DPF keygen (doDpfTreeKeyGen in fss/gpu_dpf.cu) allocates key buffers
 * (d_k0/d_l0/d_l1/d_tR) and writes them bit-by-bit via packed writes (writeVCW),
 * leaving unwritten bytes as pool residue. moveIntoCPUMem then copies the whole
 * buffer — residue included — into the key file. Since the residue differs
 * across runs/processes, the two dealer processes (party 0, party 1) produce
 * MISMATCHED correction words, breaking the FSS invariant that CWs must be
 * identical across parties. Zeroing on allocation makes keygen deterministic
 * so both dealers generate matching keys. This preserves Sigma's per-party
 * independent-dealer design (each party still runs its own keygen process).
 */
uint8_t *__wrap_gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a = __real_gpuMalloc(size_in_bytes);
    if (d_a != NULL && size_in_bytes > 0)
        cudaMemset(d_a, 0, size_in_bytes);
    return d_a;
}
