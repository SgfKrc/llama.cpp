// [QLH-SPLIT] One place for the socket plumbing shared by both ends of a stage connection.
//
// The client (src/llama-split-remote.cpp) and the server (tools/split-stage/main.cpp) were
// each carrying their own send_all/recv_all/close_sock/WSAStartup. Two copies of the same
// four functions is exactly the kind of drift that ends with one end handling a short write
// and the other not.
//
// Deliberately not reusing ggml-rpc's socket helpers: that code is a backend transport with
// its own framing and lifecycle, it is not exported (ggml/include/ggml-rpc.h exposes only
// ggml_backend_rpc_add_server), and reaching into it would tie our protocol to an internal
// boundary that upstream is free to move.
//
// Known limits of this layer (deliberate, documented rather than engineered around):
//
//  - The timeout is per read/write call, not per request and not per chain. A stage that keeps
//    trickling bytes resets it every time, and a stage that answers slowly but steadily is not
//    a timeout at all. It only catches "stopped answering altogether".
//
//  - It is set by the client (src/llama-split-remote.cpp, llama_split_remote::connect). The
//    stage process does not bound its own blocking: if the driving server stops reading, the
//    stage blocks in send() until the server's socket closes.
//
//  - A stalled chain is NOT visible through the *shallow* llama-server /health. That endpoint
//    reports whether a model is loaded, not whether the chain is moving, and it must stay that
//    way: it is what an orchestrator polls to decide whether a machine is alive, and answering
//    "down" because one network stage had a bad second is how a healthy machine gets taken out
//    of rotation. A caller that wants the chain itself must ask for it explicitly with
//    /health?deep=1 (tools/server/server-context.cpp), which drives one real token through
//    every stage and answers 503 when that cannot be done. A chain that already failed an
//    operation is remembered (llama_memory_split::remote_failure) and refused without touching
//    the network, so a deep probe cannot hang on it.
//
//    Do not poll deep=1. The probe is a real request: it runs with cache_prompt = false (so the
//    slot cannot answer it from logits it already has), which makes the server clear that slot's
//    KV and re-process the probe prompt. On a remote chain, whose head carries one sequence by
//    construction, the next genuine request then has to re-process its prompt instead of reusing
//    the cache. Poll the shallow /health; use deep=1 when a chain is already suspected, or once
//    after a config change.
#pragma once

// LLAMA_API: the stage tool links against the library instead of carrying its own copy, so
// these entry points have to be visible from the DLL. Same pattern as src/llama-ext.h.
#include "llama.h"

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX   // winsock2.h pulls in <windows.h>, whose min/max macros break std::min
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
namespace qlh_split::net {
using sock_t = SOCKET;
constexpr sock_t invalid = INVALID_SOCKET;
}
#else
#  include <arpa/inet.h>
#  include <netdb.h>          // addrinfo / getaddrinfo; Windows gets these from ws2tcpip.h
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
namespace qlh_split::net {
using sock_t = int;
constexpr sock_t invalid = -1;
}
#endif

namespace qlh_split::net {

// One-time process-wide socket setup. Safe to call more than once; on POSIX it is a no-op.
LLAMA_API bool global_init();

inline bool valid(sock_t s) { return s != invalid; }

LLAMA_API void close_sock(sock_t s);

// Both return false as soon as the peer stops cooperating. Partial transfers are retried;
// a read/write that blocks longer than the socket's timeout (see set_timeout) counts as the
// peer having stopped cooperating.
LLAMA_API bool send_all(sock_t s, const void * src, size_t n);
LLAMA_API bool recv_all(sock_t s, void * dst, size_t n);

// [QLH-SPLIT] Send two buffers as ONE write (WSASend scatter/gather on Windows, writev on POSIX).
//
// The frame format is "fixed header + payload", and the two used to be two send() calls. Over
// loopback that costs nothing, which is why it survived so long; over a real link it is the
// Nagle/delayed-ACK trap. The kernel sees a 32 B write followed immediately by a second small
// write: the first segment goes out alone, and Nagle holds the second one until the peer's
// delayed ACK arrives. Measured on a cross-host chain: 48-52 ms between the header and the body
// of the candidate reply (build-split/cand/events_9333.txt), i.e. half a token of latency added
// to every token for a 320 B frame that should have cost microseconds.
//
// Scatter/gather instead of memcpy on purpose: the biggest payload on this wire is the 248k-entry
// logits row (993 KB), and copying that once per token to make the write "contiguous" would cost
// more than the syscall it saves. WSASend takes both buffers in one call and the kernel still
// emits them as MSS-sized segments, so no user-space copy is needed.
//
// Retries partial sends, like send_all. Returns false when the peer stops cooperating.
LLAMA_API bool send_two(sock_t s, const void * a, size_t na, const void * b, size_t nb);

// [QLH-SPLIT] Turn off Nagle on one connection (TCP_NODELAY).
//
// Merging header+body (send_two) fixes the frame shapes this protocol actually sends, but it is
// a per-call cure: any future small write that follows unacknowledged data pays the same 50 ms
// again. TCP_NODELAY removes the wait wholesale, at the cost of letting the kernel send each
// write as soon as it can instead of batching small ones. That trade is right here because
// latency, not packet count, is what a chain of two machines spends its time on: every token
// crosses the link twice, so the ACK delay is charged per token.
//
// Call it on both ends after connect/accept: Nagle delays the *sender*, so a one-sided setting
// only removes half the problem. Returns false if the option could not be set.
LLAMA_API bool set_nodelay(sock_t s);

// [QLH-SPLIT] Bound how long one send/recv may block, in milliseconds (SO_RCVTIMEO/SO_SNDTIMEO).
//
// Without this a stage that stops answering -- suspended, killed without a FIN, a NIC that
// disappeared -- pins the driving slot forever: there is no other timeout on this path, and
// llama-server's /health only answers "is the model loaded", so the chain stops producing
// tokens while every health check keeps returning 200. A finite timeout turns that into the
// ordinary request failure it already knows how to report.
//
// Returns false if either option could not be set, which the caller should treat as "this
// connection has no failure bound" rather than as a usable socket.
LLAMA_API bool set_timeout(sock_t s, int ms);

// [QLH-SPLIT] Connect with a bound, in milliseconds (0 keeps the ordinary blocking connect).
//
// A blocking connect() has no timeout of its own: on Windows it waits out the kernel's SYN
// retries (~21 s) before answering WSAETIMEDOUT, and on a chain that is being rebuilt the caller
// may have to try this on every request. That is the difference between "the stage is down, the
// chain says so immediately" and "every request spends 21 s in connect() first", which is
// exactly the behaviour the rebuild throttle exists to prevent.
//
// Non-blocking connect + select is the portable way to get a deadline. The socket is left in
// blocking mode again on every path, so the caller can hand it straight to send_all/recv_all.
LLAMA_API bool connect_timeout(sock_t s, const struct sockaddr * addr, int addrlen, int ms);

} // namespace qlh_split::net
