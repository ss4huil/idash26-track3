/*
 * pool_override.c — LD_PRELOAD stub that replaces initGPUMemPool() from
 * /home/jiang/EzPC/GPU-MPC/utils/gpu_mem.cu.
 *
 * The upstream implementation does:
 *   1. Set cudaMemPoolAttrReleaseThreshold = UINT64_MAX  (keep-all)
 *   2. cudaMallocAsync(&p, 40GiB, 0) as a warmup probe
 *   3. cudaFreeAsync(p, 0)
 *
 * Step 2 fails with OOM on an 8 GiB GPU (RTX 4060 Laptop).  The probe is
 * purely an eager pool-reservation; skipping it is safe — actual allocations
 * from gpuMalloc() / moveToGPU() still work on demand via the async pool.
 *
 * This stub sets the threshold (preserving the intended pool behaviour) and
 * skips the 40 GiB probe.  Link order: LD_PRELOAD this .so before the binary
 * so the dynamic linker resolves initGPUMemPool to this version.
 */
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

void __wrap_initGPUMemPool(void)
{
    int device = 0;
    cudaMemPool_t mempool;

    cudaDeviceGetDefaultMemPool(&mempool, device);

    /* Keep pool memory: don't return it to the OS until the pool is destroyed */
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

    /* Skip the 40 GiB warm-up probe — it OOMs on 8 GiB GPUs.
     * Pool memory is committed on first actual allocation instead. */
    printf("[pool_override] initGPUMemPool: threshold=UINT64_MAX, probe skipped (GPU<40GiB)\n");
}
