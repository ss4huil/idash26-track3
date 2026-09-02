// lss_keys.h — LSS dealer key 文件格式 v2（"LSS2"，种子压缩；gpu-mpc-track v2）
//
// 协议公式与格式的权威定义见同目录 lss_protocol.md；本文件只实现
// 字节级布局。要点：
//   - 文件 = 88B header（含 7 条 PRF 流的种子）+ 比特打包记录流 + 24B trailer；
//   - 记录流为单一连续比特流，LSB-first；每条记录以 3-bit 类型标签开头；
//   - 两方文件记录严格同序（OT16 一方为 SEND 记录、另一方为 RECV 记录）；
//   - LSS2：每方每记录类型一条计数器模式 PRF 流（种子在 header），记录中
//     纯随机的份额全部由种子现场再生，只显式存储相关性修正——见
//     lss_protocol.md §9（Matchmaker ePrint 2025/424 §3.4 + App. H 的思路）。
//
// 安全假设：dealer 半诚实且不与任一方合谋（比赛 Q3 允许）；在线无 dealer。
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lss {

// ── 记录类型标签（3 bit）────────────────────────────────────────────
enum RecordType : uint8_t {
    REC_BIT_TRIPLE = 0,  // 布尔 Beaver 三元组份额：a_i,b_i 种子再生，
                         //   P1 的 c1 种子再生，P0 显式存 c0=(a∧b)⊕c1（1 bit）
    REC_OT16_SEND  = 1,  // 1oo16 OT sender：16 个 2-bit pad 全种子再生
    REC_OT16_RECV  = 2,  // 1oo16 OT receiver：r 种子再生，k_r 显式（2 bit）
    REC_MUX_TRIPLE = 3,  // 选择三元组：ab/aa0/bb 双方种子再生；
                         //   P0 显式存 cc0（64 bit），P1 显式存 aa1∈{0,1,−1}（2 bit）
    REC_B2A_CORR   = 4,  // B2A：rb 双方种子再生；ra0 P0 种子再生，ra1 显式（64 bit）
    REC_MASK_IN    = 5,  // 桥接入口 mask 份额 r_p：P0 份额种子再生，
                         //   P1 显式存 r_in−r0（64 bit；r_in 由管线给定不可压缩）：
                         //   eval 各方从 masked-public m = x+r 恢复 x 的份额
                         //   （P0: x0=m−r0, P1: x1=−r1）
    REC_MASK_OUT   = 6,  // 桥接出口 mask 份额 r'_p：双方种子再生（dealer 定义
                         //   r' = s0+s1），零显式字节；LSS 输出份额加 r' 后重构
};

inline const char *record_type_name(uint8_t t) {
    switch (t) {
        case REC_BIT_TRIPLE: return "BIT_TRIPLE";
        case REC_OT16_SEND:  return "OT16_SEND";
        case REC_OT16_RECV:  return "OT16_RECV";
        case REC_MUX_TRIPLE: return "MUX_TRIPLE";
        case REC_B2A_CORR:   return "B2A_CORR";
        case REC_MASK_IN:    return "MASK_IN";
        case REC_MASK_OUT:   return "MASK_OUT";
        default:             return "<invalid>";
    }
}

// ── 文件常量 ────────────────────────────────────────────────────────
constexpr uint8_t  LSS_MAGIC[4]   = {'L', 'S', 'S', '2'};
constexpr uint32_t LSS_VERSION    = 2;
constexpr size_t   LSS_NUM_SEEDS  = 7;   // 每记录类型一条 PRF 流（含 OT16 两角色）
constexpr size_t   LSS_HEADER_SZ  = 32 + LSS_NUM_SEEDS * 8; // 88 B
constexpr size_t   LSS_TRAILER_SZ = 24;

// ── LSS2 计数器模式 PRF：AES-128-CTR（AES-NI 硬件指令）──────────────────
// 设计（lss_protocol.md §9.3）：每条流（party × 记录类型，共 2×7 条）持一个
// 独立 AES-128 key，由 header 里的 64-bit 流种子经 splitmix64 一次性扩展
// （KDF，仅构造期一次）；记录 i 的第 word 个输出字 =
//   AES-128_K(ctr ‖ 0) 的低 64 bit，ctr = i·STRIDE + word。
// 不同流 ⇒ 不同 key ⇒ 计数器空间天然隔离。key schedule 在 keygen/eval 构造期
// 预展开（LssAesKey，只读共享），在线热路径每输出字只做 10 轮 AESENC，
// 无共享可变状态 ⇒ OpenMP 线程安全。
//
// 实现：AES-NI 内建函数（与本仓库 SCI/src/utils/aes-ni.h 同属公开领域的
// AESNI.c 习语），用 __attribute__((target("aes"))) 限定，无需给编译链加
// -maes（nvcc host 端同样可编译）；lss 模块保持零外部依赖。运行时在
// keygen/eval 构造期用 __builtin_cpu_supports("aes") 检查一次性失败退出。
#include <wmmintrin.h>

constexpr uint64_t LSS_PRF_STRIDE = 4; // 每条记录最多用 4 个 PRF 输出字

// AES-128 展开后的 11 轮密钥（构造期算好，之后只读）
struct LssAesKey {
    __m128i rk[11];
};

inline void lss_require_aesni() {
    if (!__builtin_cpu_supports("aes"))
        throw std::runtime_error("lss: LSS2 PRF (AES-CTR) 需要 AES-NI，"
                                 "当前 CPU 不支持");
}

// splitmix64 混合器：仅用于 64-bit 流种子 → 128-bit AES key 的一次性 KDF
// （不作为在线 PRF）。
inline uint64_t lss_kdf_mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__attribute__((target("aes"))) inline __m128i
lss_aes128_expand_step(__m128i k, const int rcon) {
    __m128i a = _mm_aeskeygenassist_si128(k, rcon);
    a = _mm_shuffle_epi32(a, 0xff);
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    return _mm_xor_si128(k, a);
}

// 由 128-bit key（lo/hi 两个 u64，小端装入）展开轮密钥
__attribute__((target("aes"))) inline void
lss_aes128_expand(LssAesKey &out, uint64_t lo, uint64_t hi) {
    __m128i k = _mm_set_epi64x((long long)hi, (long long)lo);
    out.rk[0] = k;
    out.rk[1] = lss_aes128_expand_step(k, 0x01);
    out.rk[2] = lss_aes128_expand_step(out.rk[1], 0x02);
    out.rk[3] = lss_aes128_expand_step(out.rk[2], 0x04);
    out.rk[4] = lss_aes128_expand_step(out.rk[3], 0x08);
    out.rk[5] = lss_aes128_expand_step(out.rk[4], 0x10);
    out.rk[6] = lss_aes128_expand_step(out.rk[5], 0x20);
    out.rk[7] = lss_aes128_expand_step(out.rk[6], 0x40);
    out.rk[8] = lss_aes128_expand_step(out.rk[7], 0x80);
    out.rk[9] = lss_aes128_expand_step(out.rk[8], 0x1b);
    out.rk[10] = lss_aes128_expand_step(out.rk[9], 0x36);
}

// 流种子 → 本条流的 AES-128 key（KDF：splitmix64 两次，域分隔常数）
inline LssAesKey lss_stream_key(uint64_t seed) {
    LssAesKey k;
    uint64_t lo = lss_kdf_mix(seed);
    uint64_t hi = lss_kdf_mix(seed ^ 0x9E3779B97F4A7C15ULL);
    lss_aes128_expand(k, lo, hi);
    return k;
}

// AES-128 单分组加密（自测/通用入口）
__attribute__((target("aes"))) inline __m128i
lss_aes128_encrypt(const LssAesKey &k, __m128i m) {
    m = _mm_xor_si128(m, k.rk[0]);
    for (int i = 1; i < 10; i++) m = _mm_aesenc_si128(m, k.rk[i]);
    return _mm_aesenclast_si128(m, k.rk[10]);
}

// 计数器模式 PRF：ctr → 伪随机 u64。纯函数、无状态、线程安全。
inline uint64_t lss_prf(const LssAesKey &k, uint64_t ctr) {
    __m128i m = _mm_set_epi64x(0, (long long)ctr);
    return (uint64_t)_mm_cvtsi128_si64(lss_aes128_encrypt(k, m));
}

// 每条记录的总 bit 数（含 3-bit 标签）按 (类型, party) 定长——随机访问
// 定位与标签校验都依赖定长。party 只接受 0/1。
inline unsigned record_total_bits(uint8_t t, int party) {
    switch (t) {
        case REC_BIT_TRIPLE: return party == 0 ? 4 : 3;  // P0: tag+c0
        case REC_OT16_SEND:  return 3;                   // tag only
        case REC_OT16_RECV:  return 5;                   // tag+k_r(2)
        case REC_MUX_TRIPLE: return party == 0 ? 67 : 5; // P0: tag+cc0; P1: tag+aa1(2)
        case REC_B2A_CORR:   return party == 0 ? 3 : 67; // P1: tag+ra1
        case REC_MASK_IN:    return party == 0 ? 3 : 67; // P1: tag+(r_in−r0)
        case REC_MASK_OUT:   return 3;                   // tag only
        default:             return 0;
    }
}

// ── Millionaires 结构计数（keygen 与在线必须一致，见 lss_protocol.md §5）──
// radix-2^4 分块：digit 数 = ceil(bitlength/4)，顶层 digit 可能不足 4 bit
// （仍用 OT16，只用前 2^r 个入口）。
inline int lss_num_digits(int bitlength) { return (bitlength + 3) / 4; }

// 每次比较的 AND 门数（AND 树，语义同 SCI traverse_and_compute_ANDs）：
// 每层 stride i：j==0 组 1 门（cmp∧eq），其余组 2 门（+eq∧eq）。
// num_digits=16 时 = 15+7+3+1 = 26。
inline uint64_t lss_and_gates_per_compare(int num_digits) {
    uint64_t g = 0;
    for (int i = 1; i < num_digits; i <<= 1)
        for (int j = 0; j + i < num_digits; j += 2 * i)
            g += (j == 0) ? 1 : 2;
    return g;
}

// ── 比特流（LSB-first：bit i → byte[i/8] 的第 i%8 位）────────────────
class BitWriter {
public:
    std::vector<uint8_t> buf;
    uint64_t nbits = 0;  // 已写 bit 总数

    // 写 value 的低 n 位（n ≤ 64），LSB-first
    void put(uint64_t value, unsigned n) {
        for (unsigned k = 0; k < n; k++) {
            if (nbits % 8 == 0) buf.push_back(0);
            if ((value >> k) & 1) buf.back() |= uint8_t(1u << (nbits % 8));
            nbits++;
        }
    }
    uint64_t payload_bytes() const { return (nbits + 7) / 8; }
};

class BitReader {
public:
    const uint8_t *buf;
    uint64_t nbits;   // 有效 bit 总数
    uint64_t pos = 0; // 当前游标（bit）

    BitReader(const uint8_t *buf, uint64_t nbits) : buf(buf), nbits(nbits) {}

    // 读 n 位（n ≤ 64），LSB-first
    uint64_t get(unsigned n) {
        if (pos + n > nbits)
            throw std::runtime_error("lss::BitReader: 越界读取（key 文件耗尽或图序错位）");
        uint64_t v = 0;
        for (unsigned k = 0; k < n; k++) {
            if ((buf[pos / 8] >> (pos % 8)) & 1) v |= (1ULL << k);
            pos++;
        }
        return v;
    }

    // 随机只读访问（const、线程安全，供并行原语使用）：读 bitpos 起的 n 位
    // （n ≤ 64），LSB-first。小端平台假定（x86/EPYC）。
    // 记录定长 ⇒ 第 e 条记录的偏移 = base + e×record_bits，可并行定位。
    uint64_t get_at(uint64_t bitpos, unsigned n) const {
        if (bitpos + n > nbits)
            throw std::runtime_error("lss::BitReader: get_at 越界（key 文件耗尽或图序错位）");
        uint64_t byte = bitpos >> 3;
        unsigned off = (unsigned)(bitpos & 7);
        unsigned __int128 v = 0;
        uint64_t total_bytes = (nbits + 7) / 8;
        uint64_t avail = total_bytes - byte;
        size_t take = avail < 16 ? (size_t)avail : 16;
        memcpy(&v, buf + byte, take);   // 末尾不足 16B 时高位为零，不影响结果
        v >>= off;
        if (n == 64) return (uint64_t)v;
        return (uint64_t)v & ((1ULL << n) - 1);
    }
    bool exhausted() const { return pos >= nbits; }
};

// ── 并行安全的小位宽打包/解包（bpe ∈ {1,2,4}：每 8/bpe 个元素恰好 1 字节，
// 各线程写整字节、互不重叠）──────────────────────────────────────────
// out 需预先分配 (n*bpe+7)/8 字节。LSB-first，与 BitWriter 布局一致。
inline void pack_small(uint8_t *out, const uint8_t *vals, size_t n,
                       unsigned bpe) {
    const unsigned per_byte = 8 / bpe;
    const uint8_t mask = (uint8_t)((1u << bpe) - 1);
    size_t nbytes = (n * bpe + 7) / 8;
#ifdef _OPENMP
#pragma omp parallel for if(nbytes > 4096)
#endif
    for (size_t by = 0; by < nbytes; by++) {
        uint8_t byte = 0;
        size_t base = by * per_byte;
        for (unsigned k = 0; k < per_byte && base + k < n; k++)
            byte |= (vals[base + k] & mask) << (k * bpe);
        out[by] = byte;
    }
}

inline void unpack_small(const uint8_t *in, uint8_t *vals, size_t n,
                         unsigned bpe) {
    const unsigned per_byte = 8 / bpe;
    const uint8_t mask = (uint8_t)((1u << bpe) - 1);
    size_t nbytes = (n * bpe + 7) / 8;
#ifdef _OPENMP
#pragma omp parallel for if(n > 32768)
#endif
    for (size_t by = 0; by < nbytes; by++) {
        uint8_t byte = in[by];
        size_t base = by * per_byte;
        for (unsigned k = 0; k < per_byte && base + k < n; k++)
            vals[base + k] = (byte >> (k * bpe)) & mask;
    }
}

// 并行校验 n 条定长记录的类型标签（图序错位的第一时间暴露）；
// 供各在线原语在 get_at 随机读之前调用。
inline void check_record_tags(const BitReader &reader, uint64_t base,
                              unsigned rec_bits, RecordType want, size_t n) {
    bool ok = true;
#ifdef _OPENMP
#pragma omp parallel for if(n > 8192) reduction(& : ok)
#endif
    for (size_t i = 0; i < n; i++)
        ok &= (reader.get_at(base + i * rec_bits, 3) == (uint8_t)want);
    if (!ok)
        throw std::runtime_error(std::string("lss: key 图序错位，期望 ") +
                                 record_type_name(want));
}

// ── header / trailer ────────────────────────────────────────────────
struct LssHeader {
    uint32_t party;
    uint64_t num_records;
    uint64_t payload_bits;
    uint64_t seeds[LSS_NUM_SEEDS]; // 本方各记录类型的 PRF 流种子（LSS2）
};

inline void write_header(FILE *f, const LssHeader &h) {
    uint8_t hdr[LSS_HEADER_SZ];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, LSS_MAGIC, 4);
    uint32_t ver = LSS_VERSION;
    memcpy(hdr + 4, &ver, 4);
    memcpy(hdr + 8, &h.party, 4);
    memcpy(hdr + 16, &h.num_records, 8);
    memcpy(hdr + 24, &h.payload_bits, 8);
    memcpy(hdr + 32, h.seeds, LSS_NUM_SEEDS * 8);
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr))
        throw std::runtime_error("lss: 写 header 失败");
}

inline LssHeader read_header(FILE *f) {
    uint8_t hdr[LSS_HEADER_SZ];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr))
        throw std::runtime_error("lss: 读 header 失败");
    if (memcmp(hdr, LSS_MAGIC, 4) != 0)
        throw std::runtime_error(
            "lss: magic 不匹配，不是 LSS2 key 文件（LSS1 与 LSS2 不兼容，"
            "keygen 与 eval 必须用同一格式版本重新生成）");
    uint32_t ver;
    memcpy(&ver, hdr + 4, 4);
    if (ver != LSS_VERSION)
        throw std::runtime_error("lss: 不支持的 key 版本 " + std::to_string(ver));
    LssHeader h;
    memcpy(&h.party, hdr + 8, 4);
    memcpy(&h.num_records, hdr + 16, 8);
    memcpy(&h.payload_bits, hdr + 24, 8);
    memcpy(h.seeds, hdr + 32, LSS_NUM_SEEDS * 8);
    return h;
}

// FNV-1a 64（简单尾校验；防截断/错位，非密码学校验）
inline uint64_t fnv1a64(const uint8_t *data, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

inline void write_trailer(FILE *f, uint64_t num_records, uint64_t payload_bytes,
                          uint64_t checksum) {
    uint8_t tr[LSS_TRAILER_SZ];
    memcpy(tr, &num_records, 8);
    memcpy(tr + 8, &payload_bytes, 8);
    memcpy(tr + 16, &checksum, 8);
    if (fwrite(tr, 1, sizeof(tr), f) != sizeof(tr))
        throw std::runtime_error("lss: 写 trailer 失败");
}

// ── KeyFileReader：在线方的 key 文件游标 ─────────────────────────────
// 一次性读入内存（P3 再做 SSD 流式/双路径，见设计文档 §5.4），
// 消费时校验标签 == 期望类型，第一时间暴露图序错位。
class KeyFileReader {
public:
    LssHeader header;
    std::vector<uint8_t> payload;
    BitReader reader;

    explicit KeyFileReader(const std::string &path)
        : header{}, payload(), reader(nullptr, 0) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("lss: 打不开 key 文件 " + path);
        header = read_header(f);
        uint64_t pbytes = (header.payload_bits + 7) / 8;
        payload.resize(pbytes);
        if (pbytes && fread(payload.data(), 1, pbytes, f) != pbytes)
            throw std::runtime_error("lss: payload 截断 " + path);
        // trailer 校验：记录数 + 字节数 + checksum
        uint8_t tr[LSS_TRAILER_SZ];
        if (fread(tr, 1, sizeof(tr), f) != sizeof(tr))
            throw std::runtime_error("lss: 读 trailer 失败 " + path);
        uint64_t t_recs, t_bytes, t_sum;
        memcpy(&t_recs, tr, 8);
        memcpy(&t_bytes, tr + 8, 8);
        memcpy(&t_sum, tr + 16, 8);
        fclose(f);
        if (t_recs != header.num_records || t_bytes != pbytes)
            throw std::runtime_error("lss: trailer 计数与 header 不符 " + path);
        if (t_sum != fnv1a64(payload.data(), pbytes))
            throw std::runtime_error("lss: payload checksum 不符 " + path);
        // LSS2：预展开 7 条流的 AES-128 轮密钥（之后只读，线程安全）
        lss_require_aesni();
        for (size_t t = 0; t < LSS_NUM_SEEDS; t++)
            stream_keys_[t] = lss_stream_key(header.seeds[t]);
        reader = BitReader(payload.data(), header.payload_bits);
    }

    // 读取下一条记录的类型标签并校验
    uint8_t expect(RecordType want) {
        uint8_t got = (uint8_t)reader.get(3);
        if (got != (uint8_t)want)
            throw std::runtime_error(
                std::string("lss: key 图序错位，期望 ") + record_type_name(want) +
                " 实为 " + record_type_name(got));
        return got;
    }

    // 游标回卷：eval 基准流程每次 forward 复用同一 key 流（与 FSS 侧
    // keyBuf = startPtr 复位同款纪律；注意这意味着 benchmark 的 11 次迭代
    // 复用同一批相关性——仅用于计时/精度验证，生产必须每批生成新 key）。
    void rewind() {
        reader.pos = 0;
        memset(ctr_, 0, sizeof(ctr_));
    }
    bool exhausted() const { return reader.exhausted(); }

    // ── LSS2 PRF 流访问（eval 侧）───────────────────────────────────
    // OT16 的 SEND/RECV 是同一逻辑相关性的两面，共用一条流计数器。
    static unsigned stream_slot(uint8_t t) {
        return t == REC_OT16_RECV ? (unsigned)REC_OT16_SEND : (unsigned)t;
    }
    // 当前该类型下一条记录的流内序号（PRF 索引基准）
    uint64_t stream_pos(uint8_t t) const { return ctr_[stream_slot(t)]; }
    void stream_advance(uint8_t t, uint64_t n) { ctr_[stream_slot(t)] += n; }
    // 本方记录 rec_idx（流内序号）的第 word 个 PRF 输出字（AES-128-CTR）
    uint64_t prf(uint8_t seed_slot, uint64_t rec_idx, unsigned word) const {
        return lss_prf(stream_keys_[seed_slot],
                       rec_idx * LSS_PRF_STRIDE + word);
    }

private:
    uint64_t ctr_[LSS_NUM_SEEDS] = {};   // 每类型的已消费记录数（PRF 索引）
    LssAesKey stream_keys_[LSS_NUM_SEEDS]; // 预展开的流轮密钥（只读）
};

} // namespace lss
