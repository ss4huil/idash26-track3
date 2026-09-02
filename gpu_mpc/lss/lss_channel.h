// lss_channel.h — 最小 TCP 双工通道（gpu-mpc-track v2 Phase 1）
//
// 一对 BSD socket，party 0 listen、party 1 connect（与现有 GpuPeer/SigmaPeer
// 并行使用不同端口即可，见设计文档 §5.3）。header-only，无任何第三方依赖。
// exchange() 双方等量互换：先发后收；大于 TCP 缓冲的批量数据按 1 MiB 块
// 交错收发避免死锁。
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/socket.h>

namespace lss {

class Channel {
public:
    int fd = -1;
    uint64_t bytes_sent = 0, bytes_recv = 0;

    Channel() = default;
    explicit Channel(int fd) : fd(fd) { set_nodelay(); }
    ~Channel() { if (fd >= 0) ::close(fd); }
    Channel(const Channel &) = delete;
    Channel &operator=(const Channel &) = delete;

    // party 0：监听并接受一个连接
    static Channel listen_and_accept(int port) {
        int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (lfd < 0) throw std::runtime_error("lss: socket() 失败");
        int one = 1;
        setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons((uint16_t)port);
        if (::bind(lfd, (sockaddr *)&addr, sizeof(addr)) < 0)
            throw std::runtime_error("lss: bind 端口 " + std::to_string(port) +
                                     " 失败: " + strerror(errno));
        if (::listen(lfd, 1) < 0) throw std::runtime_error("lss: listen 失败");
        int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) throw std::runtime_error("lss: accept 失败");
        ::close(lfd);
        return Channel(cfd);
    }

    // party 1：连接对端（带简单重试，等对端 listen 起来）
    static Channel connect(const std::string &ip, int port) {
        for (int attempt = 0; attempt < 100; attempt++) {
            int cfd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (cfd < 0) throw std::runtime_error("lss: socket() 失败");
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons((uint16_t)port);
            if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1)
                throw std::runtime_error("lss: 非法 IP " + ip);
            if (::connect(cfd, (sockaddr *)&addr, sizeof(addr)) == 0)
                return Channel(cfd);
            ::close(cfd);
            usleep(50 * 1000); // 50ms
        }
        throw std::runtime_error("lss: connect " + ip + ":" +
                                 std::to_string(port) + " 失败");
    }

    void send_all(const void *data, size_t n) {
        const uint8_t *p = (const uint8_t *)data;
        while (n > 0) {
            ssize_t s = ::send(fd, p, n, MSG_NOSIGNAL);
            if (s <= 0) throw std::runtime_error("lss: send 失败");
            p += s;
            n -= (size_t)s;
            bytes_sent += (uint64_t)s;
        }
    }

    void recv_all(void *data, size_t n) {
        uint8_t *p = (uint8_t *)data;
        while (n > 0) {
            ssize_t s = ::recv(fd, p, n, MSG_WAITALL);
            if (s <= 0) throw std::runtime_error("lss: recv 失败（对端关闭?）");
            p += s;
            n -= (size_t)s;
            bytes_recv += (uint64_t)s;
        }
    }

    // 双方等量互换：按 1 MiB 块交错收发，避免大批量时双方同时阻塞在 send。
    void exchange(const void *out, void *in, size_t n) {
        const size_t CHUNK = 1 << 20;
        const uint8_t *po = (const uint8_t *)out;
        uint8_t *pi = (uint8_t *)in;
        size_t done = 0;
        while (done < n) {
            size_t c = std::min(CHUNK, n - done);
            send_all(po + done, c);
            recv_all(pi + done, c);
            done += c;
        }
    }

private:
    void set_nodelay() {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
};

} // namespace lss
