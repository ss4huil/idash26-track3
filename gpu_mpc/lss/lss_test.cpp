// lss_test.cpp — LSS 原语双进程 loopback 单元测试（gpu-mpc-track v2 Phase 1）
//
// 用法：
//   lss_test                      — driver：生成 key 到临时目录，fork 两个
//                                   子进程（party0/party1，127.0.0.1 真 TCP），
//                                   收集退出码。
//   lss_test party <0|1> <port> <keydir>   — 子进程入口（内部使用）
//
// 每个原语对随机输入 + 边界值（全 0 / 全 1 / UINT64_MAX / 2^63 /
// digit 边界 0..15 全组合）验证：双方输出份额重构后 == 明文结果。
// OT16 的 δ=(r−d) 下标方向是重点验证对象（lss_protocol.md 原语 2）。
#include "lss_keygen.h"
#include "lss_online.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace lss;

// 测试规模
constexpr size_t N_AND  = 1024;
constexpr size_t N_OT   = 512; // 含 16×16 digit 边界全组合
constexpr size_t N_MUX  = 512;
constexpr size_t N_B2A  = 256;
constexpr size_t N_CMP  = 1024; // compare_gt 直接测试（64/63 bit 各一轮）
constexpr size_t N_RELU = 1024; // msb + relu 测试

constexpr uint64_t KEYGEN_SEED = 0xDEADBEEF42;
constexpr uint64_t INPUT_SEED  = 0x1234567890AB; // 双方共享，推明文输入

static int failures = 0;
#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "[party %d] FAIL: %s (%s:%d)\n", party,   \
                    msg, __FILE__, __LINE__);                         \
            failures++;                                               \
        }                                                             \
    } while (0)

// ── keygen ──────────────────────────────────────────────────────────
static bool do_keygen(const std::string &dir) {
    LssKeygen kg(KEYGEN_SEED);
    kg.gen_bit_triples(N_AND);
    kg.gen_ot16(N_OT, /*sender_party=*/0); // Millionaires 约定：party0=sender
    kg.gen_mux(N_MUX);
    kg.gen_b2a(N_B2A);
    // P2 算子测试记录（顺序 = 在线消费顺序）
    kg.gen_compare(N_CMP, 64);      // 测试 5：64-bit compare
    kg.gen_compare(N_CMP, 63);      // 测试 6：63-bit compare（顶 digit 3 bit）
    kg.gen_compare(N_RELU, 63);     // 测试 7：msb
    kg.gen_relu(N_RELU, 64);        // 测试 8：relu
    std::string p0 = dir + "/lss_keys_party0.bin";
    std::string p1 = dir + "/lss_keys_party1.bin";
    kg.write_files(p0, p1);

    // 打印 key 字节核算
    struct stat st0, st1;
    stat(p0.c_str(), &st0);
    stat(p1.c_str(), &st1);
    printf("[keygen] records=%llu  party0=%lld B  party1=%lld B\n",
           (unsigned long long)kg.num_records(), (long long)st0.st_size,
           (long long)st1.st_size);
    // 文件大小必须与 payload_bits 严格一致（格式核算）
    for (int p = 0; p < 2; p++) {
        long long expect =
            (long long)((kg.payload_bits(p) + 7) / 8 + LSS_HEADER_SZ +
                        LSS_TRAILER_SZ);
        long long got = (p == 0 ? st0.st_size : st1.st_size);
        if (got != expect) {
            fprintf(stderr,
                    "[keygen] FAIL: party%d 文件大小 %lld 与理论 %lld 不符\n",
                    p, got, expect);
            return false;
        }
    }
    // 每元素（bw=64 ReLU）key 核算：compare63 = 16 OT16 + 26 triple，+1 MUX
    {
        double bits0 = 16.0 * 35 + 26.0 * 6 + 196; // party0 (OT sender)
        double bits1 = 16.0 * 9 + 26.0 * 6 + 196;  // party1 (OT receiver)
        printf("[keygen] 每 ReLU 元素 key: party0=%.1f B  party1=%.1f B\n",
               bits0 / 8, bits1 / 8);
    }
    return true;
}

// ── party 逻辑 ──────────────────────────────────────────────────────
// 双方用同一 INPUT_SEED 推明文输入，再各自取自己的份额：
// 布尔份额 share_p = (p==0 ? rnd : rnd ^ plain)
// 算术份额 share_p = (p==0 ? rnd : plain - rnd)
static int run_party(int party, int port, const std::string &dir) {
    Channel chan = (party == 0)
                       ? Channel::listen_and_accept(port)
                       : Channel::connect("127.0.0.1", port);
    LssParty P(party, dir + "/lss_keys_party" + std::to_string(party) + ".bin",
               chan, /*local_seed=*/0xC0FFEE); // 本地掩码 PRNG，确定性可复现
    std::mt19937_64 rng(INPUT_SEED);

    // ── 测试 1：AND ──
    {
        std::vector<uint8_t> x(N_AND), y(N_AND), xs(N_AND), ys(N_AND), zs(N_AND);
        for (size_t i = 0; i < N_AND; i++) {
            if (i < 256) { // 边界：全 0 / 全 1 组合
                x[i] = (i / 128) & 1;
                y[i] = (i / 64) & 1;
            } else {
                x[i] = rng() & 1;
                y[i] = rng() & 1;
            }
            uint8_t rx = rng() & 1, ry = rng() & 1;
            xs[i] = (party == 0) ? rx : (rx ^ x[i]);
            ys[i] = (party == 0) ? ry : (ry ^ y[i]);
        }
        P.and_open(xs.data(), ys.data(), zs.data(), N_AND);
        std::vector<uint8_t> z(N_AND);
        P.open_bits(zs.data(), z.data(), N_AND);
        for (size_t i = 0; i < N_AND; i++)
            CHECK(z[i] == (x[i] & y[i]), "and_open 结果与明文不符");
        printf("[party %d] AND: %zu 次通过（含全0/全1边界）\n", party, N_AND);
    }

    // ── 测试 2：OT16（party0=sender）──
    {
        std::vector<uint8_t> g(N_OT * 16), s(N_OT), d(N_OT), out(N_OT, 0);
        for (size_t e = 0; e < N_OT; e++) {
            if (e < 256) {
                // digit 边界全组合：t=e/16, choice=e%16，
                // g(i) = ((t>i)<<1)|(t==i) —— Millionaires 叶子用法
                uint8_t t = e / 16;
                d[e] = e % 16;
                for (int i = 0; i < 16; i++)
                    g[e * 16 + i] = ((t > i) << 1) | (t == i);
                s[e] = rng() & 3;
            } else {
                d[e] = rng() & 15;
                s[e] = rng() & 3;
                for (int i = 0; i < 16; i++) g[e * 16 + i] = rng() & 3;
            }
        }
        if (party == 0)
            P.ot16_send(g.data(), s.data(), N_OT);
        else
            P.ot16_recv(d.data(), out.data(), N_OT);
        // 双方输出份额 XOR == g(d)（sender 份额 = s，receiver 份额 = out）
        // receiver 本地直接校验（测试进程中明文 d,g,s 均已知）
        if (party == 1)
            for (size_t e = 0; e < N_OT; e++)
                CHECK((uint8_t)(s[e] ^ out[e]) == g[e * 16 + d[e]],
                      "ot16 输出份额重构 != g(d)（下标/方向错误?）");
        // 交叉校验：receiver 把 out 发给 sender，sender 验证 s⊕out==g(d)
        if (party == 1) {
            chan.send_all(out.data(), N_OT);
        } else {
            std::vector<uint8_t> ro(N_OT);
            chan.recv_all(ro.data(), N_OT);
            for (size_t e = 0; e < N_OT; e++)
                CHECK((uint8_t)(s[e] ^ ro[e]) == g[e * 16 + d[e]],
                      "ot16 sender 侧重构校验失败");
        }
        printf("[party %d] OT16: %zu 次通过（含 digit 16×16 边界全组合）\n",
               party, N_OT);
    }

    // ── 测试 3：MUX ──
    {
        std::vector<uint8_t> b(N_MUX), bs(N_MUX);
        std::vector<uint64_t> x(N_MUX), xs(N_MUX), zs(N_MUX);
        for (size_t i = 0; i < N_MUX; i++) {
            b[i] = rng() & 1;
            switch (i % 5) { // 边界值
                case 0: x[i] = 0; break;
                case 1: x[i] = 1; break;
                case 2: x[i] = UINT64_MAX; break;
                case 3: x[i] = 1ULL << 63; break;
                default: x[i] = rng(); break;
            }
            uint8_t rb = rng() & 1;
            uint64_t rx = rng();
            bs[i] = (party == 0) ? rb : (rb ^ b[i]);
            xs[i] = (party == 0) ? rx : (x[i] - rx);
        }
        P.mux(bs.data(), xs.data(), zs.data(), N_MUX);
        std::vector<uint64_t> z(N_MUX);
        P.open_u64(zs.data(), z.data(), N_MUX);
        for (size_t i = 0; i < N_MUX; i++)
            CHECK(z[i] == (b[i] ? x[i] : 0), "mux 结果与明文不符");
        printf("[party %d] MUX: %zu 次通过（含 0/1/UINT64_MAX/2^63 边界）\n",
               party, N_MUX);
    }

    // ── 测试 4：B2A ──
    {
        std::vector<uint8_t> b(N_B2A), bs(N_B2A);
        std::vector<uint64_t> zs(N_B2A);
        for (size_t i = 0; i < N_B2A; i++) {
            b[i] = (i < 2) ? (uint8_t)i : (rng() & 1); // 边界 0,1 + 随机
            uint8_t rb = rng() & 1;
            bs[i] = (party == 0) ? rb : (rb ^ b[i]);
        }
        P.b2a(bs.data(), zs.data(), N_B2A);
        std::vector<uint64_t> z(N_B2A);
        P.open_u64(zs.data(), z.data(), N_B2A);
        for (size_t i = 0; i < N_B2A; i++)
            CHECK(z[i] == (uint64_t)b[i], "b2a 结果与明文不符");
        printf("[party %d] B2A: %zu 次通过\n", party, N_B2A);
    }

    // ── 测试 5/6：compare_gt（64-bit 与 63-bit）──
    for (int bitlength : {64, 63}) {
        std::vector<uint64_t> a(N_CMP), b(N_CMP);
        std::vector<uint8_t> out(N_CMP);
        uint64_t mask = (bitlength == 64) ? ~0ULL : ((1ULL << bitlength) - 1);
        for (size_t i = 0; i < N_CMP; i++) {
            // 边界组合 + 等值 + 随机
            switch (i % 10) {
                case 0: a[i] = 0;            b[i] = 0;            break;
                case 1: a[i] = 0;            b[i] = mask;         break;
                case 2: a[i] = mask;         b[i] = 0;            break;
                case 3: a[i] = mask;         b[i] = mask;         break;
                case 4: a[i] = mask >> 1;    b[i] = (mask >> 1) + 1; break;
                case 5: a[i] = (mask >> 1) + 1; b[i] = mask >> 1; break;
                case 6: a[i] = 1;            b[i] = 0;            break;
                case 7: a[i] = 0;            b[i] = 1;            break;
                case 8: a[i] = rng() & mask; b[i] = a[i];         break; // 等值
                default: a[i] = rng() & mask; b[i] = rng() & mask; break;
            }
        }
        uint64_t sent0 = P.chan.bytes_sent, recv0 = P.chan.bytes_recv;
        int f0 = failures;
        P.compare_gt(out.data(), (party == 0 ? a.data() : b.data()), N_CMP,
                     bitlength);
        std::vector<uint8_t> o(N_CMP);
        P.open_bits(out.data(), o.data(), N_CMP);
        for (size_t i = 0; i < N_CMP; i++)
            CHECK(o[i] == (a[i] > b[i] ? 1 : 0), "compare_gt 结果与明文不符");
        printf("[party %d] compare%d: %s（%d 失败；含等值/0/最大值/2^(k-1) 边界）"
               "  comm sent=%llu B recv=%llu B（%.1f B/元素 sent）\n",
               party, bitlength,
               failures == f0 ? "全部通过" : "存在失败", failures - f0,
               (unsigned long long)(P.chan.bytes_sent - sent0),
               (unsigned long long)(P.chan.bytes_recv - recv0),
               (double)(P.chan.bytes_sent - sent0) / N_CMP);
    }

    // ── 测试 7：MSB ──
    {
        std::vector<uint64_t> x(N_RELU), xs(N_RELU);
        std::vector<uint8_t> out(N_RELU);
        for (size_t i = 0; i < N_RELU; i++) {
            switch (i % 8) { // 边界：0/1/2^63 邻域/最大值/负数边界
                case 0: x[i] = 0; break;
                case 1: x[i] = 1; break;
                case 2: x[i] = (1ULL << 63) - 1; break;
                case 3: x[i] = 1ULL << 63; break;           // 最小负数
                case 4: x[i] = (1ULL << 63) + 1; break;
                case 5: x[i] = UINT64_MAX; break;           // -1
                case 6: x[i] = UINT64_MAX - 1; break;       // -2
                default: x[i] = rng(); break;
            }
            uint64_t rx = rng();
            xs[i] = (party == 0) ? rx : (x[i] - rx);
        }
        int f0 = failures;
        P.msb(xs.data(), out.data(), N_RELU, 64);
        std::vector<uint8_t> o(N_RELU);
        P.open_bits(out.data(), o.data(), N_RELU);
        for (size_t i = 0; i < N_RELU; i++)
            CHECK(o[i] == (uint8_t)(x[i] >> 63), "msb 结果与明文不符");
        printf("[party %d] MSB: %s（%d 失败；含 0/2^63±1/-1/-2 边界）\n",
               party, failures == f0 ? "全部通过" : "存在失败", failures - f0);
    }

    // ── 测试 8：ReLU ──
    {
        std::vector<uint64_t> x(N_RELU), xs(N_RELU), ys(N_RELU);
        for (size_t i = 0; i < N_RELU; i++) {
            switch (i % 8) {
                case 0: x[i] = 0; break;
                case 1: x[i] = 1; break;
                case 2: x[i] = (1ULL << 63) - 1; break;
                case 3: x[i] = 1ULL << 63; break;
                case 4: x[i] = (1ULL << 63) + 1; break;
                case 5: x[i] = UINT64_MAX; break;
                case 6: x[i] = UINT64_MAX - 4096; break;    // 小负数
                default: x[i] = rng(); break;
            }
            uint64_t rx = rng();
            xs[i] = (party == 0) ? rx : (x[i] - rx);
        }
        uint64_t sent0 = P.chan.bytes_sent;
        int f0 = failures;
        P.relu(xs.data(), ys.data(), N_RELU, 64);
        std::vector<uint64_t> y(N_RELU);
        P.open_u64(ys.data(), y.data(), N_RELU);
        for (size_t i = 0; i < N_RELU; i++) {
            uint64_t want = (x[i] < (1ULL << 63)) ? x[i] : 0; // 有符号 ReLU
            CHECK(y[i] == want, "relu 结果与明文不符");
        }
        printf("[party %d] ReLU: %s（%d 失败；含 0/2^63±1/负数边界）"
               "  comm %.1f B/元素 sent\n",
               party, failures == f0 ? "全部通过" : "存在失败", failures - f0,
               (double)(P.chan.bytes_sent - sent0) / N_RELU);
    }

    // key 必须恰好耗尽（图序对齐）
    CHECK(P.keys.reader.exhausted(), "key 流未耗尽或超读（图序错位）");
    printf("[party %d] 通信量: sent=%llu B recv=%llu B\n", party,
           (unsigned long long)P.chan.bytes_sent,
           (unsigned long long)P.chan.bytes_recv);
    return failures == 0 ? 0 : 1;
}

// ── driver ──────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "party") == 0) {
        int party = atoi(argv[2]);
        int port = atoi(argv[3]);
        return run_party(party, port, argv[4]);
    }

    char dir[] = "/tmp/lss_test_XXXXXX";
    if (!mkdtemp(dir)) {
        perror("mkdtemp");
        return 1;
    }
    if (!do_keygen(dir)) {
        std::string cmd = std::string("rm -rf ") + dir;
        if (system(cmd.c_str()) != 0) perror("cleanup");
        return 1;
    }

    int port = 43200 + (int)(getpid() % 2000);
    pid_t kids[2];
    for (int p = 0; p < 2; p++) {
        kids[p] = fork();
        if (kids[p] == 0) {
            char port_str[16], party_str[8];
            snprintf(port_str, sizeof(port_str), "%d", port);
            snprintf(party_str, sizeof(party_str), "%d", p);
            execl("/proc/self/exe", "lss_test", "party", party_str, port_str,
                  dir, (char *)nullptr);
            perror("execl");
            _exit(127);
        }
    }
    int rc = 0;
    for (int p = 0; p < 2; p++) {
        int st;
        waitpid(kids[p], &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            printf("[driver] party%d 失败 (status=%d)\n", p, st);
            rc = 1;
        }
    }
    std::string cmd = std::string("rm -rf ") + dir;
    if (system(cmd.c_str()) != 0) perror("cleanup");
    printf(rc == 0 ? "[driver] 全部通过 ✅\n" : "[driver] 存在失败 ❌\n");
    return rc;
}
