//
// DeepDTAGen affinity path — 2PC inference driver (iDASH 2024 Track 3).
//
// Follows the structure of experiments/orca/orca_inference.cu: role 0 is the
// dealer (FSS key generation), role 1 is the evaluator (online protocol). The
// model graph is DeepDTAGenAffinity<T> (see deepdtagen.h).
//
// Inputs (produced by the Python side, all in fixed-point scale = 24, bw = 64):
//   * X, A_hat, mask  : P1's drug-graph secret shares  (reference/share_data.py)
//   * model weights   : P2's private weights blob       (reference/export_weights.py)
//   * proteinEmb      : GatedCNN(protein) as a trivial share (public sequence)
//
// BUILD/RUN BLOCKER (this environment): an RTX 4060 (CC 8.9) + driver are
// present but there is NO nvcc/CUDA toolkit installed, so this file is
// write-only here. Compile once the toolkit is available (from this dir):
//     make GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC GPU_ARCH=89 CUDA_VERSION=<ver> \
//          deepdtagen_inference
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

// Load a length-n fixed-point share file (headerless little-endian int64,
// matching reference/share_data.py) into a host Tensor<InfType>.
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
