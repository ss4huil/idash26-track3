// lss_keys.h — LSS dealer key 文件格式 v1（gpu-mpc-track v2 Phase 1）
//
// 协议公式与格式的权威定义见同目录 lss_protocol.md；本文件只实现
// 字节级布局。要点：
//   - 文件 = 32B header + 比特打包记录流 + 24B trailer；
//   - 记录流为单一连续比特流，LSB-first；每条记录以 3-bit 类型标签开头；
//   - 两方文件记录严格同序（OT16 一方为 SEND 记录、另一方为 RECV 记录）。
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
    REC_BIT_TRIPLE = 0,  // 布尔 Beaver 三元组份额 (a,b,c=a∧b)，3 bit 负载
    REC_OT16_SEND  = 1,  // 1oo16 OT sender：16 个 2-bit pad
    REC_OT16_RECV  = 2,  // 1oo16 OT receiver：r(4bit) + pad_r(2bit)
    REC_MUX_TRIPLE = 3,  // 选择三元组：a^b(1) + a^a(64) + b(64) + c(64)
    REC_B2A_CORR   = 4,  // B2A 相关性：rb(1) + ra(64)
    REC_MASK_IN    = 5,  // 桥接入口 mask 份额 r_p（u64）：eval 各方从 masked-public
                         // m = x+r 恢复 x 的份额（P0: x0=m−r0, P1: x1=−r1）
    REC_MASK_OUT   = 6,  // 桥接出口 mask 份额 r'_p（u64）：LSS 输出份额加 r'
                         // 后重构，恢复 masked-public
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
constexpr uint8_t  LSS_MAGIC[4]   = {'L', 'S', 'S', '1'};
constexpr uint32_t LSS_VERSION    = 1;
constexpr size_t   LSS_HEADER_SZ  = 32;
constexpr size_t   LSS_TRAILER_SZ = 24;

// 每条记录的负载 bit 数（不含 3-bit 标签）
constexpr unsigned BITS_BIT_TRIPLE = 3;
constexpr unsigned BITS_OT16_SEND  = 32;
constexpr unsigned BITS_OT16_RECV  = 6;
constexpr unsigned BITS_MUX_TRIPLE = 193;
constexpr unsigned BITS_B2A_CORR   = 65;
constexpr unsigned BITS_MASK       = 64; // MASK_IN / MASK_OUT

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
    bool exhausted() const { return pos >= nbits; }
};

// ── header / trailer ────────────────────────────────────────────────
struct LssHeader {
    uint32_t party;
    uint64_t num_records;
    uint64_t payload_bits;
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
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr))
        throw std::runtime_error("lss: 写 header 失败");
}

inline LssHeader read_header(FILE *f) {
    uint8_t hdr[LSS_HEADER_SZ];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr))
        throw std::runtime_error("lss: 读 header 失败");
    if (memcmp(hdr, LSS_MAGIC, 4) != 0)
        throw std::runtime_error("lss: magic 不匹配，不是 LSS key 文件");
    uint32_t ver;
    memcpy(&ver, hdr + 4, 4);
    if (ver != LSS_VERSION)
        throw std::runtime_error("lss: 不支持的 key 版本 " + std::to_string(ver));
    LssHeader h;
    memcpy(&h.party, hdr + 8, 4);
    memcpy(&h.num_records, hdr + 16, 8);
    memcpy(&h.payload_bits, hdr + 24, 8);
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
    void rewind() { reader.pos = 0; }
    bool exhausted() const { return reader.exhausted(); }
};

} // namespace lss
