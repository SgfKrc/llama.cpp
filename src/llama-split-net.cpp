// [QLH-SPLIT] See llama-split-net.h.
#include "llama-split-net.h"

#ifndef _WIN32
// timeval for SO_RCVTIMEO/SO_SNDTIMEO: <sys/socket.h> is not required to pull it in.
#  include <sys/time.h>
// iovec/writev for send_two (scatter/gather).
#  include <sys/uio.h>
// fcntl / errno for the bounded connect below.
#  include <fcntl.h>
#  include <cerrno>
#endif

namespace qlh_split::net {

bool global_init() {
#ifdef _WIN32
    static bool ready = false;
    if (!ready) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            return false;
        }
        ready = true;
    }
#endif
    return true;
}

void close_sock(sock_t s) {
    if (!valid(s)) {
        return;
    }
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
}

bool send_all(sock_t s, const void * src, size_t n) {
    const auto * p = (const uint8_t *) src;
    while (n > 0) {
        const int put = (int) send(s, (const char *) p, (int) n, 0);
        if (put <= 0) {
            return false;
        }
        p += put;
        n -= (size_t) put;
    }
    return true;
}

bool send_two(sock_t s, const void * a, size_t na, const void * b, size_t nb) {
    // Scatter/gather, not memcpy: see the header. The loop only matters when the kernel accepts
    // part of the request (a full send buffer, or a signal on POSIX); a partial transfer has to
    // be resumed from the first buffer that still has bytes left, so the split is tracked here
    // instead of in the caller.
    const auto * pa = (const uint8_t *) a;
    const auto * pb = (const uint8_t *) b;
    while (na > 0 || nb > 0) {
#ifdef _WIN32
        WSABUF bufs[2];
        DWORD nbufs = 0;
        if (na > 0) {
            bufs[nbufs].buf = (char *) pa;
            bufs[nbufs].len = (ULONG) na;
            ++nbufs;
        }
        if (nb > 0) {
            bufs[nbufs].buf = (char *) pb;
            bufs[nbufs].len = (ULONG) nb;
            ++nbufs;
        }
        DWORD sent = 0;
        if (WSASend(s, bufs, nbufs, &sent, 0, nullptr, nullptr) == SOCKET_ERROR || sent == 0) {
            return false;
        }
        size_t left = (size_t) sent;
#else
        iovec iov[2];
        int nbufs = 0;
        if (na > 0) {
            iov[nbufs].iov_base = (void *) pa;
            iov[nbufs].iov_len  = na;
            ++nbufs;
        }
        if (nb > 0) {
            iov[nbufs].iov_base = (void *) pb;
            iov[nbufs].iov_len  = nb;
            ++nbufs;
        }
        const ssize_t sent = writev(s, iov, nbufs);
        if (sent <= 0) {
            return false;
        }
        size_t left = (size_t) sent;
#endif
        if (left < na) {
            pa += left;
            na -= left;
        } else {
            left -= na;
            na = 0;
            pb += left;
            nb -= left;
        }
    }
    return true;
}

bool set_nodelay(sock_t s) {
    if (!valid(s)) {
        return false;
    }
    int one = 1;
#ifdef _WIN32
    return setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one)) == 0;
#else
    return setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0;
#endif
}

bool recv_all(sock_t s, void * dst, size_t n) {
    auto * p = (uint8_t *) dst;
    while (n > 0) {
        const int got = (int) recv(s, (char *) p, (int) n, 0);
        if (got <= 0) {
            return false;
        }
        p += got;
        n -= (size_t) got;
    }
    return true;
}

bool set_timeout(sock_t s, int ms) {
    if (!valid(s) || ms <= 0) {
        return false;
    }
#ifdef _WIN32
    // Winsock wants a DWORD of milliseconds for both options, not a timeval.
    const DWORD dw = (DWORD) ms;
    const bool ok_rcv = setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *) &dw, sizeof(dw)) == 0;
    const bool ok_snd = setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *) &dw, sizeof(dw)) == 0;
#else
    timeval tv{};
    tv.tv_sec  = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    const bool ok_rcv = setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
    const bool ok_snd = setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) == 0;
#endif
    return ok_rcv && ok_snd;
}

bool connect_timeout(sock_t s, const struct sockaddr * addr, int addrlen, int ms) {
    if (!valid(s) || addr == nullptr) {
        return false;
    }
    if (ms <= 0) {
        return ::connect(s, addr, addrlen) == 0;
    }

    // Non-blocking for the duration of the connect only. The caller's send/recv helpers assume a
    // blocking socket, and this file is the only place that is allowed to flip that.
#ifdef _WIN32
    u_long nb = 1;
    if (ioctlsocket(s, FIONBIO, &nb) != 0) {
        return false;
    }
#else
    const int flags = fcntl(s, F_GETFL, 0);
    if (flags < 0 || fcntl(s, F_SETFL, flags | O_NONBLOCK) < 0) {
        return false;
    }
#endif

    bool ok = ::connect(s, addr, addrlen) == 0;
    if (!ok) {
#ifdef _WIN32
        const int err = WSAGetLastError();
        const bool in_progress = (err == WSAEWOULDBLOCK);
#else
        const bool in_progress = (errno == EINPROGRESS);
#endif
        if (in_progress) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(s, &wfds);
            timeval tv{};
            tv.tv_sec  = ms / 1000;
            tv.tv_usec = (ms % 1000) * 1000;
#ifdef _WIN32
            const int sel = select(0, nullptr, &wfds, nullptr, &tv); // nfds is ignored on Windows
#else
            const int sel = select(s + 1, nullptr, &wfds, nullptr, &tv);
#endif
            if (sel > 0) {
                // Readable-for-write only means the attempt finished; SO_ERROR says how.
                int so_err = 0;
#ifdef _WIN32
                int so_len = (int) sizeof(so_err);
#else
                socklen_t so_len = sizeof(so_err);
#endif
                if (getsockopt(s, SOL_SOCKET, SO_ERROR, (char *) &so_err, &so_len) == 0) {
                    ok = (so_err == 0);
                }
            }
        }
    }

    // Back to blocking on every path, including the failures: a socket that stays non-blocking
    // would turn every later send/recv into an EWOULDBLOCK the rest of this layer reads as a
    // dead peer.
#ifdef _WIN32
    nb = 0;
    ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, flags);
#endif
    return ok;
}

} // namespace qlh_split::net
