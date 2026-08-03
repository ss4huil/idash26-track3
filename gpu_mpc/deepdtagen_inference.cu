//
// DeepDTAGen affinity path — 2PC inference driver (iDASH 2024 Track 3).
//
// Packaged after experiments/sigma/sigma_offline_online.cu: a single binary
// with a `role` switch — role 0 is the dealer (FSS key generation written to
// disk, the offline phase), role 1 is the evaluator (online protocol). The
// crypto BACKEND is Orca (backend/orca.h), the only GPU-MPC backend providing
// the relu / select / maxPool2D gates this graph needs; the SIGMA backend
// lacks them, so we borrow only Sigma's offline/online *packaging*, not its
// backend. The model graph is DeepDTAGenAffinity<T> (see deepdtagen.h).
//
// Inputs (produced by the Python offline side, fixed-point scale = 12, bw = 32
// — the Q20.12 ring proven to clear the accuracy gate; build BW=64 for a
// 64-bit production ring). Ring element width = InfType (u32 for BW=32, u64
// for BW=64), selected by the Makefile's -DInfType flag:
//   * X, A_hat, mask : P0/P1 drug-graph secret shares  (reference/share_data.py)
//       files {x,adj,mask}_share{0,1}.dat, headerless little-endian InfType,
//       emitted by reference/offline_prepare.py
//   * proteinEmb     : GatedCNN(protein) fixed-point constant, protein_emb.dat
//       — public sequence, loaded on party 1 only (party 0 holds zero)
//   * model weights  : public weights blob weights.bin (+ weights.bin.json
//       sidecar), reference/export_weights.py. NOTE: the weight blob is int64
//       REGARDLESS of the ring bw (export_weights hardcodes BITWIDTH=64), so
//       the weight loader must read int64 even when shares are u32.
//
// BUILD/RUN (CUDA 12.1 toolkit installed at /usr/local/cuda-12.1; nvcc is not
// on the login PATH — prepend it). From this dir, for the local RTX 4060
// (SM 8.9), 32-bit ring:
//     export PATH=/usr/local/cuda-12.1/bin:$PATH
//     make GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC BW=32 GPU_ARCH=89 deepdtagen_inference
// The EzPC/GPU-MPC checkout is used strictly read-only (headers + util TUs).
//
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <omp.h>

#include "backend/orca.h"
#include "deepdtagen.h"

#ifndef InfType
#define InfType u64
#endif

// Load a length-n fixed-point share file (headerless little-endian InfType —
// u32 for BW=32, u64 for BW=64 — matching reference/share_data.py's bw mode)
// into a host Tensor<InfType>.
static void loadShare(const std::string &path, Tensor<InfType> &t)
{
    std::ifstream f(path, std::ios::binary);
    assert(f.good() && "missing share file");
    f.read((char *)t.data, t.size() * sizeof(InfType));
    assert(f.gcount() == (std::streamsize)(t.size() * sizeof(InfType)));
}

int main(int argc, char *argv[])
{
    // argv: bw scale role party keyDir shareDir [ip]
    sytorch_init();
    int bw       = atoi(argv[1]);
    u64 scale    = strtoul(argv[2], 0, 10);
    int role     = atoi(argv[3]);   // 0 = dealer, 1 = evaluator
    int party    = atoi(argv[4]);
    auto keyDir  = std::string(argv[5]);
    auto shareDir = std::string(argv[6]);
    assert(bw <= 8 * (int)sizeof(InfType));
    assert(scale < (u64)bw);

    const u64 Nmax = DDG_NMAX, FEAT = DDG_FEAT;

    auto model = new DeepDTAGenAffinity<InfType>();

    // Primary input X (Nmax x FEAT). Secret side-inputs:
    //   A_hat      (Nmax x Nmax)  P1 graph
    //   maskTiled  (Nmax x 376)   P1 node mask, tiled across pooled channels
    //   proteinEmb (1 x 128)      P2 GatedCNN output (public seq) as a share
    // GCN biases fold into P2's FC weights (see gcn_layer.h) — no bias leaf here.
    Tensor<InfType> X({Nmax, FEAT});
    Tensor<InfType> A_hat({Nmax, Nmax});
    Tensor<InfType> maskTiled({Nmax, 376});
    Tensor<InfType> proteinEmb({1, 128});
    X.zero(); A_hat.zero(); maskTiled.zero(); proteinEmb.zero();
    model->setSample(&A_hat, &maskTiled, &proteinEmb);

    model->init(scale, X);
    model->zero();

    auto expName = std::string("DeepDTAGen_") + std::to_string(bw) + "_" + std::to_string(scale);
    auto keyFileName = keyDir + expName;

    if (role == 0)
    {
        // Dealer: generate FSS keys for the whole graph. Inputs stay zeroed;
        // only shapes matter for keygen.
        auto fss = new OrcaKeygen<InfType>(party, bw, scale, keyFileName);
        model->setBackend(fss);
        model->optimize();
        X.d_data = (InfType *)moveToGPU((u8 *)X.data, X.size() * sizeof(InfType), (Stats *)NULL);
        auto &out = model->forward(X);
        fss->output(out);
        fss->close();
        printf("[dealer] keys written to %s\n", keyFileName.c_str());
    }
    else
    {
        // Evaluator: load this party's real secret shares, then run online.
        //  P1 secrets (drug graph): X, A_hat, maskTiled — both parties hold a share.
        loadShare(shareDir + "/x_share"    + std::to_string(party) + ".dat", X);
        loadShare(shareDir + "/adj_share"  + std::to_string(party) + ".dat", A_hat);
        loadShare(shareDir + "/mask_share" + std::to_string(party) + ".dat", maskTiled);
        //  P2 public constant (GatedCNN output): P2 holds the value, P1 holds
        //  zero. Load only on party 1; party 0's tensor stays zeroed.
        if (party == 1)
            loadShare(shareDir + "/protein_emb.dat", proteinEmb);

        auto ip = argv[7];
        auto fss = new Orca<InfType>(party, ip, bw, (int)scale, keyFileName);
        model->setBackend(fss);
        model->optimize();

        std::vector<u64> times;
        u64 commBytes = 0;
        lseek(fss->fd, 0, SEEK_SET);
        readKey(fss->fd, fss->keySize, fss->startPtr, NULL);
        for (int i = 0; i < 11; i++)
        {
            fss->keyBuf = fss->startPtr;
            fss->s.reset();
            fss->peer->sync();
            auto commStart = fss->peer->bytesSent() + fss->peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            X.d_data = (InfType *)moveToGPU((u8 *)X.data, X.size() * sizeof(InfType), &(fss->s));
            auto &out = model->forward(X);
            fss->output(out);
            auto end = std::chrono::high_resolution_clock::now();
            if (i > 0)
                times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            auto commEnd = fss->peer->bytesSent() + fss->peer->bytesReceived();
            if (i == 0)
                commBytes = commEnd - commStart;
        }
        fss->close();
        auto avgTime = std::reduce(times.begin(), times.end()) / (float)times.size();
        printf("Average time taken (microseconds)=%f\n", avgTime);
        printf("Comm (B)=%lu\n", commBytes);
    }
    return 0;
}
