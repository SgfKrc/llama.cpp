// See llama-split-remote.h for the error policy this implements.
#include "llama-split-remote.h"
#include "llama-impl.h"   // LLAMA_LOG_*: the library logger, so --log-file and verbosity apply

#include "llama-split-net.h"
#include "llama-split-proto.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace qlh_split;

namespace {

// The data-path read timeout: far above any honest stage latency (one -ubatch of prefill on CPU),
// finite only so a stage that stops answering becomes a failed request instead of a pinned slot.
constexpr int kStageTimeoutMs = 120 * 1000;

// Budget for one rebuild handshake (connect, then the clear that follows it).
//
// Deliberately not the data-path timeout. The rebuild runs on a request thread, often for a chain
// that is still down, and its whole job is to answer "is this stage back?" -- a question a healthy
// stage answers in microseconds and a suspended one never answers at all. A stage that has been
// suspended still completes the TCP handshake out of its listen backlog, so the connect succeeds
// and it is the clear() that hangs; bounding it here is what keeps one request from paying the
// full 120 s while the throttle keeps the next five from even trying.
constexpr int kRebuildTimeoutMs = 1500;

// "host:port", and nothing more elaborate: the endpoint syntax has to stay obvious because it
// appears in command lines.
bool parse_endpoint(const std::string & ep, std::string & host, uint16_t & port) {
    const size_t colon = ep.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= ep.size()) {
        return false;
    }
    host = ep.substr(0, colon);
    try {
        const int p = std::stoi(ep.substr(colon + 1));
        if (p <= 0 || p > 65535) {
            return false;
        }
        port = (uint16_t) p;
    } catch (...) {
        return false;
    }
    return true;
}

// The reply header is checked on every operation; keeping the two failure messages in
// one place is what keeps them from drifting. The version check in particular used to say
// "bad reply header", which an operator reads as "the network is flaky": the actual cause is
// always that one end was built from a different source tree, and saying so is the whole point
// of refusing to serve in that state.
bool check_reply_header(const rsp_header & rsp, std::string & err) {
    if (rsp.magic != MAGIC) {
        err = "the peer is not speaking the split-stage protocol (reply magic " +
              std::to_string((unsigned) rsp.magic) + ", expected " + std::to_string((unsigned) MAGIC) +
              "); check the endpoint and port";
        return false;
    }
    if (rsp.version != PROTO_VER) {
        err = "protocol version mismatch: this build speaks v" + std::to_string((unsigned) PROTO_VER) +
              ", the peer answered v" + std::to_string((unsigned) rsp.version) +
              ". Both ends must be built from the same source tree (rebuild llama-server and "
              "split-stage together); this is not a network problem";
        return false;
    }
    return true;
}

const char * role_name(uint8_t r) {
    switch (r) {
        case ROLE_RELAY: return "relay";
        case ROLE_TAIL:  return "tail";
        default:         return "unknown";
    }
}

// What the stage says it is. Printed next to the expectation in a mismatch message, so an
// operator can compare the two ranges without looking up anything else.
//
// The four chain-identity numbers are part of the description. They are not
// optional decoration: when they disagree the two ends produce a hard failure mid-request (a KV
// slice too small, a rollback refused, an output buffer that aborts), and the operator's only chance
// to see the pair of numbers side by side is this message.
std::string describe_identity(const stage_id & id) {
    return "layers [" + std::to_string(id.layer_lo) + ", " + std::to_string(id.layer_hi) + ") of a " +
           std::to_string(id.n_layer_src) + "-layer model (n_embd_inp " +
           std::to_string(id.n_embd_inp) + ", n_embd_out " + std::to_string(id.n_embd_out) +
           ", n_vocab " + std::to_string(id.n_vocab) + ", role " + role_name(id.role) + ")" +
           ", n_seq_max " + std::to_string((unsigned) id.n_seq_max) +
           ", n_ctx_seq " + std::to_string(id.n_ctx_seq) +
           ", n_rs_seq " + std::to_string(id.n_rs_seq) +
           ", n_verify_rows " + std::to_string(id.n_verify_rows);
}

// What this position in the chain must be. `layer_hi` may be unknown for a stage that is not the
// chain's last one (see stage_id); that is shown as '?' rather than as a number, so the message
// does not claim to know something it does not.
std::string describe_expect(const stage_id & id) {
    const std::string hi = id.layer_hi == ID_ANY_I32 ? "?" : std::to_string(id.layer_hi);
    return "layers [" + std::to_string(id.layer_lo) + ", " + hi + ") of a " +
           std::to_string(id.n_layer_src) + "-layer model (n_embd_inp " +
           std::to_string(id.n_embd_inp) + ", n_embd_out " + std::to_string(id.n_embd_out) +
           ", n_vocab " + std::to_string(id.n_vocab) + ", role " + role_name(id.role) + ")" +
           ", n_seq_max " + std::to_string((unsigned) id.n_seq_max) +
           ", n_ctx_seq " + std::to_string(id.n_ctx_seq) +
           ", n_rs_seq " + std::to_string(id.n_rs_seq) +
           ", n_verify_rows " + std::to_string(id.n_verify_rows);
}

// Every field is compared; the list is built rather than short-circuited so one failure message
// names all the numbers that disagree. An empty answer means the identity matches.
std::string identity_mismatch(const stage_id & expect, const stage_id & got) {
    std::string out;
    const auto add = [&out](const char * field, const std::string & want, const std::string & have) {
        if (!out.empty()) {
            out += "; ";
        }
        out += std::string(field) + " expected " + want + ", the stage reports " + have;
    };
    if (expect.layer_lo != got.layer_lo) {
        add("layer_lo", std::to_string(expect.layer_lo), std::to_string(got.layer_lo));
    }
    if (expect.layer_hi != ID_ANY_I32 && expect.layer_hi != got.layer_hi) {
        add("layer_hi", std::to_string(expect.layer_hi), std::to_string(got.layer_hi));
    }
    if (expect.n_layer_src != got.n_layer_src) {
        add("n_layer_src", std::to_string(expect.n_layer_src), std::to_string(got.n_layer_src));
    }
    if (expect.n_embd_inp != got.n_embd_inp) {
        add("n_embd_inp", std::to_string(expect.n_embd_inp), std::to_string(got.n_embd_inp));
    }
    if (expect.n_embd_out != got.n_embd_out) {
        add("n_embd_out", std::to_string(expect.n_embd_out), std::to_string(got.n_embd_out));
    }
    if (expect.n_vocab != got.n_vocab) {
        add("n_vocab", std::to_string(expect.n_vocab), std::to_string(got.n_vocab));
    }
    if (expect.role != ROLE_ANY && expect.role != got.role) {
        add("role", role_name(expect.role), role_name(got.role));
    }
    // The four numbers a multi-sequence chain has to agree on.
    //
    // Three of them are compared as "the stage may be larger, never smaller", and that is not a
    // loosened check -- it is the direction the failure actually has:
    //  - n_ctx_seq: too small on the stage is a mid-request ST_NO_SPACE (the head hands it a
    //    position it has no room for); too large is a few unused cells.
    //  - n_rs_seq: too small on the stage is a refused rollback (the request fails); too large is a
    //    few unused snapshots.
    //  - n_verify_rows: too small on the stage is output_reserve() aborting the stage process; too
    //    large is a bigger logits buffer than needed.
    // n_seq_max is the one that must be equal: it *divides* the pool, so a stage with more sequences
    // gives every one of them less room than the head thinks it has, and a stage with fewer has
    // nowhere to put the extra ones.
    //
    // The message names the flags rather than leaving the operator to work out which pair of numbers
    // belongs to which option.
    if (expect.n_seq_max != got.n_seq_max) {
        add("n_seq_max (-np / --n-seq-max)", std::to_string((unsigned) expect.n_seq_max),
            std::to_string((unsigned) got.n_seq_max));
    }
    if (got.n_ctx_seq < expect.n_ctx_seq) {
        add("n_ctx_seq (-c / -np, the stage must be at least this large)",
            std::to_string(expect.n_ctx_seq), std::to_string(got.n_ctx_seq));
    }
    if (got.n_rs_seq < expect.n_rs_seq) {
        add("n_rs_seq (--split-rs-seq / --rs-seq, the stage must be at least this large)",
            std::to_string(expect.n_rs_seq), std::to_string(got.n_rs_seq));
    }
    if (got.n_verify_rows < expect.n_verify_rows) {
        add("n_verify_rows (--split-verify-rows, the stage must be at least this large)",
            std::to_string(expect.n_verify_rows), std::to_string(got.n_verify_rows));
    }
    return out;
}

} // namespace

llama_split_remote::~llama_split_remote() {
    close();
}

bool llama_split_remote::state_size(int32_t seq_id, uint32_t flags, uint32_t & n_bytes, int32_t & pos_max, std::string & err) {
    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_STATE_SAVE;
    // The size the head has to make room for is the *handle record*, not the
    // stage's state. Asking with the handle flag is what makes the stage answer with what
    // state_take() would actually give back (STATE_HANDLE_BYTES), and it keeps the size query
    // and the save in the same mode -- they are two round trips and the head checks that the
    // second agrees with the first.
    req.flags   = (uint8_t) ((flags & 0xFF) | STATE_FLAG_SIZE_ONLY | STATE_FLAG_HANDLE);
    req.seq_id  = seq_id;

    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }
    if (!send_all(&req, sizeof(req))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return false;
    }
    if (!check_reply_header(rsp, err)) {
        return false;
    }
    if (rsp.status != ST_OK) {
        err = "stage refused the state size query";
        return false;
    }
    if (rsp.n_bytes > MAX_STATE) {
        err = "stage reported an implausible state size";
        return false;
    }
    n_bytes = rsp.n_bytes;
    pos_max = rsp.pos_max;
    return true;
}

bool llama_split_remote::state_take(int32_t seq_id, uint32_t flags, uint64_t & handle, int32_t & pos_max, std::string & err) {
    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_STATE_SAVE;
    req.flags   = (uint8_t) ((flags & 0xFF) | STATE_FLAG_HANDLE);
    req.seq_id  = seq_id;

    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }
    if (!send_all(&req, sizeof(req))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return false;
    }
    if (!check_reply_header(rsp, err)) {
        return false;
    }
    if (rsp.status != ST_OK) {
        err = "stage refused the state save";
        return false;
    }
    if (rsp.n_bytes != STATE_HANDLE_BYTES) {
        // A stage that answered with its whole blob here would be an older one, but the version
        // in the header already ruled that out -- so this is a corrupt frame, not a slow path.
        err = "stage answered " + std::to_string(rsp.n_bytes) + " bytes for a snapshot handle, "
              "expected " + std::to_string(STATE_HANDLE_BYTES);
        return false;
    }

    uint64_t h = 0;
    if (!recv_all(&h, sizeof(h))) {
        err = "recv snapshot handle failed";
        return false;
    }
    if (h == 0) {
        err = "stage answered a zero snapshot handle";
        return false;
    }
    handle  = h;
    pos_max = rsp.pos_max;
    return true;
}

bool llama_split_remote::state_give(int32_t seq_id, uint32_t flags, uint64_t handle, int32_t pos_expected,
                                    int32_t & pos_max_before, bool & stale, std::string & err) {
    stale = false;

    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_STATE_LOAD;
    req.flags   = (uint8_t) ((flags & 0xFF) | STATE_FLAG_HANDLE);
    req.seq_id  = seq_id;
    req.pos0    = pos_expected;
    req.n_bytes = STATE_HANDLE_BYTES;

    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }
    if (!send_all(&req, sizeof(req)) || !send_all(&handle, sizeof(handle))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return false;
    }
    if (!check_reply_header(rsp, err)) {
        return false;
    }
    pos_max_before = rsp.pos_max;

    if (rsp.status == ST_STALE) {
        // Not a failure: the stage dropped the snapshot (cache eviction, or a restart) and did
        // not touch its memory. The caller must recover by re-processing, not by killing the
        // chain -- see the contract in llama-split-remote.h and the server branch that uses it.
        stale = true;
        return true;
    }
    if (rsp.status != ST_OK) {
        err = "stage refused the state load (its position is " + std::to_string(rsp.pos_max) +
              ", the snapshot was taken at " + std::to_string(pos_expected) + ")";
        return false;
    }
    return true;
}

bool llama_split_remote::state_load(int32_t seq_id, uint32_t flags, const std::vector<uint8_t> & blob,
                                    int32_t pos_expected, int32_t & pos_max_before, std::string & err) {
    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_STATE_LOAD;
    req.flags   = (uint8_t) (flags & 0xFF);
    req.seq_id  = seq_id;
    req.pos0    = pos_expected;
    req.n_bytes = (uint32_t) blob.size();

    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }
    if (blob.size() > MAX_STATE) {
        err = "state blob too large to send";
        return false;
    }
    if (!send_all(&req, sizeof(req)) || (!blob.empty() && !send_all(blob.data(), blob.size()))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return false;
    }
    if (!check_reply_header(rsp, err)) {
        return false;
    }
    if (rsp.status != ST_OK) {
        pos_max_before = rsp.pos_max;
        err = "stage refused the state load (its position is " + std::to_string(rsp.pos_max) +
              ", the snapshot was taken at " + std::to_string(pos_expected) + ")";
        return false;
    }
    pos_max_before = rsp.pos_max;
    return true;
}

void llama_split_remote::close() {
    close(true);
}

void llama_split_remote::close(bool send_bye) {
    if (net::valid(sock)) {
        if (send_bye && greeted) {
            // Best effort BYE so the stage returns to accept() instead of logging a broken pipe.
            // Only after a completed HELLO: a stage refuses any first frame that is not one,
            // so a BYE on a half-open attach would be reported as a protocol error there.
            req_header h{};
            h.magic   = MAGIC;
            h.version = PROTO_VER;
            h.op      = OP_BYE;
            send_all(&h, sizeof(h));
        }
        net::close_sock(sock);
        sock = net::invalid;
    }
    greeted = false;
}

bool llama_split_remote::connect_endpoint(const std::string & endpoint, std::string & err, int connect_ms, int io_ms) {
    ep = endpoint;

    std::string host;
    uint16_t    port = 0;
    if (!parse_endpoint(endpoint, host, port)) {
        err = "bad endpoint '" + endpoint + "', expected host:port";
        return false;
    }

    // Winsock setup lives in llama-split-net.cpp: the stage and this client must not carry
    // separate WSAStartup paths, or one of them ends up initializing after its first socket call.
    if (!net::global_init()) {
        err = "WSAStartup failed";
        return false;
    }

    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * res = nullptr;
    const std::string port_s = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) != 0 || res == nullptr) {
        err = "cannot resolve " + host;
        return false;
    }

    net::sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (!net::valid(s)) {
        freeaddrinfo(res);
        err = "socket() failed";
        return false;
    }

    // A rebuild may have to run on a request thread while the stage is still down, so the connect
    // itself has to be bounded too (a blocking connect to a host that is not answering waits out
    // the kernel's SYN retries, ~21 s on Windows). The initial attach keeps the ordinary blocking
    // connect: a chain that cannot reach its stage at startup should say so with the OS's own
    // error, not with a deadline nobody can distinguish from a slow network.
    if (!net::connect_timeout(s, res->ai_addr, (int) res->ai_addrlen, connect_ms)) {
        freeaddrinfo(res);
        net::close_sock(s);
        err = connect_ms > 0
            ? "cannot connect to " + endpoint + " within " + std::to_string(connect_ms) + " ms"
            : "cannot connect to " + endpoint;
        return false;
    }
    freeaddrinfo(res);

    // Nagle off for the life of the connection (see llama-split-net.h).
    //
    // This side had it from the start, which is exactly why the *up* direction never showed the
    // 48-52 ms gap the proxy trace caught: every request frame (header + 8 KB hidden rows) left
    // immediately, 0.1-0.2 ms apart. The stage side did not, so its 32 B header + 320 B
    // candidate body paid the delayed-ACK wait instead. Setting it here through the shared
    // helper keeps both ends from drifting apart again.
    net::set_nodelay(s);

    // The stage protocol has no other timing bound, so a stage that stops answering
    // mid-flight -- suspended, killed in a way that leaves the connection half-open, a NIC that
    // went away -- blocks the driving slot forever. Nothing in llama-server notices by default:
    // /health reports whether the model is loaded, not whether the chain is moving, so an
    // orchestrator keeps routing traffic at a server that can no longer emit a token. A read
    // timeout makes that an ordinary failed request (recv_all returns false -> decode returns
    // -3, or seq_rm fails and marks the chain dead) which the server already reports.
    //
    // 120 s is chosen to be far above any honest stage latency: the slowest answer this chain
    // asks for is one -ubatch of prefill on CPU, which is seconds at the batch sizes and models
    // it is used with, not minutes. It only has to be finite; making it smaller starts cutting
    // working requests on a busy machine, and making it configurable would turn a safety net
    // into something an operator can misconfigure.
    if (!net::set_timeout(s, io_ms)) {
        net::close_sock(s);
        err = "cannot set a read timeout on the connection to " + endpoint;
        return false;
    }

    sock = s;
    greeted = false; // a fresh socket has not been through a HELLO yet
    return true;
}

bool llama_split_remote::connect(const std::string & endpoint, std::string & err) {
    close();
    // A new connection has not confirmed anything yet. reconnect() must keep `confirmed` (that is
    // what it demands back), so this reset lives here and not in connect_endpoint().
    confirmed      = stage_id{};
    have_confirmed = false;
    return connect_endpoint(endpoint, err, 0, kStageTimeoutMs);
}

bool llama_split_remote::hello(const stage_id & expect, stage_id * got, std::string & err) {
    if (got != nullptr) {
        *got = stage_id{};
    }
    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }

    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_HELLO;
    req.n_bytes = (uint32_t) sizeof(stage_id);
    if (!send_all(&req, sizeof(req)) || !send_all(&expect, sizeof(expect))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        // A stage that predates the handshake checks the version on the first frame and answers BAD_REQ, so
        // this really is "no answer", not "wrong answer": name the build skew as the likely cause
        // instead of leaving the operator to blame the network.
        err = "the stage did not answer the identity handshake (it may be a split-stage from an older build, "
              "or not a split-stage at all); this build speaks the v" +
              std::to_string((unsigned) PROTO_VER) + " protocol";
        return false;
    }
    if (!check_reply_header(rsp, err)) {
        return false;
    }

    // From here on the peer is in the protocol state: a close may send BYE even if the identity
    // check below fails.
    greeted = true;

    stage_id self{};
    if (rsp.n_bytes != 0) {
        if (rsp.n_bytes != sizeof(stage_id)) {
            err = "bad HELLO reply: the stage sent " + std::to_string(rsp.n_bytes) +
                  " bytes of identity, this build expects " + std::to_string(sizeof(stage_id)) +
                  " (the two ends were not built from the same source tree)";
            return false;
        }
        if (!recv_all(&self, sizeof(self))) {
            err = "recv failed";
            return false;
        }
    }
    if (got != nullptr) {
        *got = self;
    }
    if (rsp.status != ST_OK) {
        err = "the stage refused the identity handshake: the stage is " + describe_identity(self) +
              ", the chain expects " + describe_expect(expect);
        return false;
    }

    const std::string diff = identity_mismatch(expect, self);
    // QLH_SPLIT_SKIP_IDENTITY=1 turns the comparison off, and it exists for exactly
    // one purpose: exercising the *degraded* path a mismatched chain would take if the handshake did
    // not catch it (the first frame then answers ST_BAD_REQ from the stage's own per-row check_seq_id
    // instead of a clean attach-time refusal). The degraded path needs both halves of that
    // sentence, and the second one cannot be reached while the first one works.
    //
    // Test hook: kept on purpose. The multi-sequence verification sets this variable and greps the
    // server log for the warning below; deleting the switch would delete the only way to reach the
    // degraded path.
    //
    // It is read here rather than plumbed through a parameter because it must not be reachable from a
    // normal command line: a chain that skips its identity check is a chain whose tokens nobody can
    // trust, and an operator should have to know an env var name to get there. Same style as the
    // existing QLH_FORCE_SEQ_RM_FAIL test hook.
    static const bool skip_identity = []() {
        const char * v = getenv("QLH_SPLIT_SKIP_IDENTITY");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (!diff.empty() && !skip_identity) {
        err = "stage identity mismatch: this stage is " + describe_identity(self) +
              ", but the chain expects " + describe_expect(expect) + " (" + diff + "). " +
              "It looks like the wrong stage is listening on " + ep + ". Both ends of a chain must "
              "agree on -np / --n-seq-max, -c (which is what makes n_ctx_seq equal), "
              "--split-rs-seq / --rs-seq and --split-verify-rows; the stage may be configured larger "
              "than the head, never smaller. Check those four on both sides.";
        return false;
    }
    if (!diff.empty()) {
        // Loud on purpose: this is a chain that has been told to ignore the one check that makes a
        // wrong stage visible before it computes anything.
        LLAMA_LOG_WARN("[split-remote] QLH_SPLIT_SKIP_IDENTITY=1: ignoring an identity mismatch "
                       "with %s (%s)\n", ep.c_str(), diff.c_str());
    }

    // Only an identity that matched is remembered: reconnect() must demand exactly this stage
    // back, not "a stage that answered the same expectations" (which for a middle stage would
    // leave its layer_hi unpinned).
    confirmed      = self;
    have_confirmed = true;
    return true;
}

bool llama_split_remote::reconnect(std::string & err) {
    // No BYE: see the header. The old socket is closed unconditionally -- even if the stage has
    // already answered nothing for a while, this side is the only one that can drop it.
    close(false);

    if (ep.empty()) {
        err = "this stage has no endpoint recorded; it was never connected";
        return false;
    }
    if (!have_confirmed) {
        // No successful attach ever happened on this client, so there is no identity to demand.
        // Refusing is the safe answer: a rebuild exists to restore a chain that was built.
        err = "this stage has no confirmed identity; it was never attached";
        return false;
    }
    if (!connect_endpoint(ep, err, kRebuildTimeoutMs, kRebuildTimeoutMs)) {
        return false;
    }

    // The whole point of the rebuild: a TCP connection is not "the stage is back". Give up the
    // socket on a mismatch rather than leaving a half-verified connection in place, so the chain
    // stays dead and the next attempt starts clean.
    stage_id seen{};
    std::string why;
    if (!hello(confirmed, &seen, why)) {
        close(false);
        err = "reconnected, but the stage now listening there is not the one this chain was "
              "built from: " + why;
        return false;
    }
    return true;
}

bool llama_split_remote::restore_data_timeout() {
    return net::valid(sock) && net::set_timeout(sock, kStageTimeoutMs);
}

bool llama_split_remote::send_all(const void * src, size_t n) {
    return net::send_all(sock, src, n);
}

bool llama_split_remote::recv_all(void * dst, size_t n) {
    return net::recv_all(sock, dst, n);
}

// ---------------------------------------------------------------------------------------------
// Reading one reply's body.
//
// The sections come in the protocol's wire order -- row table, per-row ids, h_nextn rows, the
// output section, taps (see body_layout in llama-split-proto.h) -- and each reader below owns
// exactly one of them: it consumes its section and leaves the stream at the next one, or it fails.
// decode() calls them in that order exactly once and returns from one place.
//
// That is the point of the split. The reply's shape decides which sections exist, and the body used
// to answer that with an early `return 0` per shape -- five of them, and every one had to consume
// the tap block first, because the tap section is last on the wire. A rule like that is enforced by
// nothing: a shape added later forgets it and the next frame is read out of the previous reply's
// tail. Here the tap read is the last statement before the single return.
// ---------------------------------------------------------------------------------------------

namespace {

// What the request knows about the reply it is about to read, packed once so that the section
// readers do not each take ten arguments. Immutable while a reply is being read.
struct reply_frame {
    const rsp_header &  rsp;
    const body_layout & lay;
    uint32_t            n_tok;      // frame rows: the tap section's stride
    uint32_t            n_embd;     // frame row width
    uint32_t            tap_wide;   // width of one tap row
    uint64_t            tap_bytes;  // the whole tap section, 64-bit checked
    const std::vector<const llama_split_cand_req *> & cand_rows;
    size_t   n_cand_rows;
    bool     verify_rows;
    bool     nextn_rows;
    bool     reply_is_hidden;
    bool     tap_on_hidden;
};

// The row table, immediately after the header and before every other body section. It says which
// rows of the frame this answer covers, ascending, and every per-row section below it is in that
// order. It is read for every shape -- it is part of the body, not of any one channel -- which is
// what keeps the stream aligned.
int read_rows(net::sock_t sock, const reply_frame & f, split_reply & reply, std::string & err) {
    reply.rows.resize(f.rsp.n_tokens);
    if (!net::recv_all(sock, reply.rows.data(), reply.rows.size() * sizeof(int32_t))) {
        err = "recv row table failed";
        return -3;
    }
    for (uint32_t i = 0; i < f.rsp.n_tokens; ++i) {
        if (reply.rows[i] < 0 || reply.rows[i] >= (int32_t) f.n_tok ||
            (i > 0 && reply.rows[i] <= reply.rows[i - 1])) {
            // A row number outside the frame (or out of order) would have this side hand back
            // another row's data as if it were the row it asked for -- a wrong token with no error
            // anywhere, which is the one failure mode a row table exists to make impossible.
            err = "stage answered a row table that is not ascending or is outside this " +
                  std::to_string(f.n_tok) + "-row frame (entry " + std::to_string(i) + " = " +
                  std::to_string(reply.rows[i]) + ")";
            return -3;
        }
    }
    return 0;
}

// The per-row ids of a verify answer: one int32 per row of the reply's *row table*, in table order
// -- not one per row of the frame. A frame whose verify rows are rows 4..11 answers eight ids and
// names them in the table; a table-wide answer with -1 holes is what lets the verify section and
// the h_nextn section share one table when the frame also carries a sequence that did not ask for
// either.
int read_verify(net::sock_t sock, const reply_frame & f, split_reply & reply, std::string & err) {
    if ((f.rsp.flags & RSP_FLAG_VERIFY_ROWS) == 0) {
        return 0;
    }
    if (!f.verify_rows) {
        err = "stage answered per-row results that were not requested";
        return -3;
    }
    if (!f.lay.matches(f.rsp.n_bytes)) {
        err = "stage answered " + std::to_string(f.rsp.n_bytes) + " bytes for a " +
              std::to_string(f.lay.total) + "-byte verify answer (" +
              std::to_string(f.rsp.n_tokens) + " rows)";
        return -3;
    }
    reply.verify.resize(f.rsp.n_tokens);
    if (!net::recv_all(sock, reply.verify.data(), reply.verify.size() * sizeof(int32_t))) {
        err = "recv per-row tokens failed";
        return -3;
    }
    return 0;
}

// The h_nextn rows, one per table row. A tail without a NextN block ignores the flag, so "no rows"
// is an answer and not an error -- see the nextn note on decode().
int read_nextn(net::sock_t sock, const reply_frame & f, split_reply & reply, std::string & err) {
    // nextn and layer-input taps are mutually exclusive on the wire (one reply has one n_embd
    // field), so a stage that set both is answering a frame no build of this tree sends. Checked
    // for every shape, and before the section it guards.
    if ((f.rsp.flags & RSP_FLAG_NEXTN_ROWS) && (f.rsp.flags & RSP_FLAG_LAYER_INP)) {
        err = "stage answered both h_nextn rows and layer-input taps in one reply";
        return -3;
    }
    if ((f.rsp.flags & RSP_FLAG_NEXTN_ROWS) == 0) {
        return 0;
    }
    if (!f.nextn_rows) {
        err = "stage answered h_nextn rows that were not requested";
        return -3;
    }
    // The stage says how many rows it owes by naming them in the row table, and rsp.n_tokens is
    // that table's length. The width is checked against it, so a stage that answered a different
    // shape is caught here instead of shifting every later reply.
    const uint32_t n_rows = f.rsp.n_tokens;
    if (f.rsp.n_embd != f.n_embd || n_rows == 0 ||
        (size_t) f.lay.nextn_bytes != (size_t) n_rows * f.n_embd * sizeof(float)) {
        err = "stage answered implausible h_nextn rows (n_tokens " + std::to_string(f.rsp.n_tokens) +
              ", n_embd " + std::to_string(f.rsp.n_embd) + ", n_bytes " +
              std::to_string(f.rsp.n_bytes) + ")";
        return -3;
    }
    reply.nextn.resize((size_t) n_rows * f.rsp.n_embd);
    if (!net::recv_all(sock, reply.nextn.data(), reply.nextn.size() * sizeof(float))) {
        err = "recv h_nextn rows failed";
        return -3;
    }
    reply.nextn_embd = f.rsp.n_embd;
    return 0;
}

// The output section: exactly one of the three channels a reply answers with, in the precedence
// body_layout states -- candidates, else logits rows when the reply states a vocabulary, else the
// relay's hidden rows. A verify answer whose vocabulary is 0 (a pure verify reply) has no output
// section at all, and this reader reads nothing for it.
int read_output(net::sock_t sock, const reply_frame & f, split_reply & reply, std::string & err) {
    // A relay answer. When it also carries taps, both widths are in its header, so the block is
    // checked strictly here (the layout table is authoritative) instead of being inferred.
    if (f.reply_is_hidden) {
        if (f.tap_on_hidden) {
            if (f.rsp.n_tokens != f.n_tok || f.rsp.n_embd != f.n_embd) {
                err = "stage answered " + std::to_string(f.rsp.n_tokens) + " hidden rows of " +
                      std::to_string(f.rsp.n_embd) + " for a " + std::to_string(f.n_tok) +
                      "-row frame of " + std::to_string(f.n_embd);
                return -3;
            }
            const uint64_t hidden_bytes = (uint64_t) f.n_tok * f.n_embd * sizeof(float);
            if (!f.lay.matches(f.rsp.n_bytes)) {
                err = "stage answered " + std::to_string(f.rsp.n_bytes) + " bytes for " +
                      std::to_string(hidden_bytes) + " hidden bytes plus " +
                      std::to_string(f.tap_bytes) + " of taps";
                return -3;
            }
        }
        reply.hidden.resize((size_t) f.rsp.n_tokens * f.rsp.n_embd);
        if (!net::recv_all(sock, reply.hidden.data(), reply.hidden.size() * sizeof(float))) {
            err = "recv hidden failed";
            return -3;
        }
        return 0;
    }

    if (f.rsp.flags & RSP_FLAG_CANDIDATES) {
        // The candidate section is self-describing --
        //   uint32 n_cand_rows, n_cand_rows * {int32 row, uint32 n_bytes}, then the payloads in
        // that order -- because one reply can carry several rows' arrays and the width of each is
        // the chain's top_k, which no header field carries (see RSP_FLAG_CANDIDATES).
        if ((uint64_t) f.rsp.n_bytes < f.lay.total) {
            err = "stage answered " + std::to_string(f.rsp.n_bytes) + " bytes for " +
                  std::to_string(f.lay.nextn_bytes) + " of h rows and " +
                  std::to_string(f.tap_bytes) + " of taps";
            return -3;
        }
        if (f.n_cand_rows == 0) {
            // The stage must not volunteer candidates: every later row would be sampled from a
            // set nobody validated.
            err = "stage answered candidates that were not requested";
            return -3;
        }
        // A1: the candidates are the one section whose width the layout table cannot know.
        // Everything else comes from the table, so the count is the remainder, not a re-derivation
        // of the body.
        const uint32_t cand_bytes = f.rsp.n_bytes - f.lay.total;
        if (cand_bytes < sizeof(uint32_t)) {
            err = "stage answered an implausible candidate section (" +
                  std::to_string(cand_bytes) + " bytes)";
            return -3;
        }
        uint32_t n_rows_got = 0;
        if (!net::recv_all(sock, &n_rows_got, sizeof(n_rows_got))) {
            err = "recv candidate row count failed";
            return -3;
        }
        const uint64_t refs_bytes = (uint64_t) n_rows_got * sizeof(cand_row_ref);
        if (n_rows_got == 0 || n_rows_got > (uint32_t) f.n_tok ||
            (uint64_t) sizeof(uint32_t) + refs_bytes > cand_bytes) {
            err = "stage answered " + std::to_string(n_rows_got) + " candidate rows for a " +
                  std::to_string(cand_bytes) + "-byte section";
            return -3;
        }
        std::vector<cand_row_ref> refs(n_rows_got);
        if (!net::recv_all(sock, refs.data(), refs.size() * sizeof(cand_row_ref))) {
            err = "recv candidate index failed";
            return -3;
        }
        uint64_t payload_sum = 0;
        for (uint32_t i = 0; i < n_rows_got; ++i) {
            if (refs[i].row < 0 || refs[i].row >= (int32_t) f.n_tok ||
                (i > 0 && refs[i].row <= refs[i - 1].row) ||
                refs[i].n_bytes == 0 || refs[i].n_bytes % sizeof(token_cand) != 0) {
                // A row number outside the frame, out of order, or a payload that is not a whole
                // number of pairs would have this side hand one row's array to another row's
                // sampler -- a wrong token with no error anywhere.
                err = "stage answered a candidate index that is not ascending, outside this " +
                      std::to_string(f.n_tok) + "-row frame, or not pair-aligned (entry " +
                      std::to_string(i) + " = row " + std::to_string(refs[i].row) + ", " +
                      std::to_string(refs[i].n_bytes) + " bytes)";
                return -3;
            }
            payload_sum += refs[i].n_bytes;
        }
        if ((uint64_t) sizeof(uint32_t) + refs_bytes + payload_sum != cand_bytes) {
            err = "stage answered a candidate section of " + std::to_string(cand_bytes) +
                  " bytes whose index and payloads add up to " +
                  std::to_string(sizeof(uint32_t) + refs_bytes + payload_sum);
            return -3;
        }
        reply.cands.resize(n_rows_got);
        for (uint32_t i = 0; i < n_rows_got; ++i) {
            reply.cands[i].row = refs[i].row;
            reply.cands[i].pairs.resize(refs[i].n_bytes / sizeof(token_cand));
            if (!net::recv_all(sock, reply.cands[i].pairs.data(), refs[i].n_bytes)) {
                err = "recv candidates failed";
                return -3;
            }
        }
        return 0;
    }

    // The logits rows. A reply that states a vocabulary has to state how many rows it carries, and
    // that section is n_logits * n_vocab floats; n_logits == 0 with a vocabulary set is a corrupt
    // frame (it would have this side read the next reply's header as a row).
    if (f.rsp.n_vocab > 0 && f.rsp.n_logits == 0) {
        err = "stage answered a logits reply with no rows";
        return -3;
    }
    const uint64_t logits_bytes = (uint64_t) f.rsp.n_logits * f.rsp.n_vocab * sizeof(float);
    // A logits reply has never carried n_bytes (the op predates the field being used for replies):
    // the rows are exactly n_logits * n_vocab floats and the old parse read them without looking.
    // So the arithmetic is only checked when the stage did set n_bytes -- a stage that sets it has
    // to cover the rows plus the taps, and one that leaves it 0 answers the single-row shape, which
    // the head parses without the tap section.
    if (f.rsp.n_vocab > 0 && f.rsp.n_bytes > 0 && (uint64_t) f.rsp.n_bytes < f.lay.total) {
        err = "stage answered " + std::to_string(f.rsp.n_bytes) + " bytes for " +
              std::to_string(logits_bytes) + " bytes of logits rows and " +
              std::to_string(f.tap_bytes) + " of taps";
        return -3;
    }
    reply.logits.resize((size_t) f.rsp.n_logits * f.rsp.n_vocab);
    if (!reply.logits.empty() &&
        !net::recv_all(sock, reply.logits.data(), reply.logits.size() * sizeof(float))) {
        err = "recv logits failed";
        return -3;
    }
    reply.n_vocab = f.rsp.n_vocab;
    return 0;
}

// The layer-input taps. One lambda in decode() used to do this, and every early exit had to call it
// first because the section is last on the wire; here it is simply the last section read.
int read_tap(net::sock_t sock, const reply_frame & f, split_reply & reply, std::string & err) {
    if ((f.rsp.flags & RSP_FLAG_LAYER_INP) == 0) {
        return 0;
    }
    reply.tap.ids.resize(f.rsp.n_taps);
    if (!net::recv_all(sock, reply.tap.ids.data(), reply.tap.ids.size() * sizeof(uint32_t))) {
        err = "recv layer-input tap ids failed";
        return -3;
    }
    reply.tap.rows.resize((size_t) f.n_tok * f.rsp.n_taps * f.tap_wide);
    if (!reply.tap.rows.empty() &&
        !net::recv_all(sock, reply.tap.rows.data(), reply.tap.rows.size() * sizeof(float))) {
        err = "recv layer-input tap rows failed";
        return -3;
    }
    reply.tap.n_embd = f.tap_wide;
    return 0;
}

} // namespace

int llama_split_remote::decode(const std::vector<float> & hidden,
                               const std::vector<int32_t> & positions,
                               uint32_t n_pos,
                               uint32_t n_embd,
                               const std::vector<row_req> & rows,
                               const std::vector<const llama_split_cand_req *> & cand_rows,
                               bool verify_rows,
                               bool nextn_rows,
                               const llama_split_tap_req & tap,
                               split_reply & reply,
                               std::string & err) {
    if (!net::valid(sock)) {
        err = "not connected";
        return -3;
    }
    if (n_pos == 0 || positions.size() != (hidden.size() / (n_embd == 0 ? 1 : n_embd)) * n_pos) {
        err = "positions/hidden size mismatch";
        return -1;
    }

    const uint32_t n_tok = (uint32_t) (hidden.size() / (n_embd == 0 ? 1 : n_embd));

    // The frame's per-row records are what tells the stage which sequence owns each
    // row, so a frame that does not describe exactly its own rows cannot be sent at all. This is the
    // sender's check, not the stage's: a mismatch would have the stage read n_tokens records out of
    // whatever follows the rows section.
    if (rows.size() != n_tok) {
        err = "per-row records do not describe this frame (" + std::to_string(rows.size()) +
              " rows for a " + std::to_string(n_tok) + "-row frame)";
        return -1;
    }
    // One optional candidate request per row. A vector that is not
    // the frame's row count cannot be aligned with the records above, so it is a caller bug.
    if (!cand_rows.empty() && cand_rows.size() != n_tok) {
        err = "candidate requests do not describe this frame (" + std::to_string(cand_rows.size()) +
              " entries for a " + std::to_string(n_tok) + "-row frame)";
        return -1;
    }
    size_t n_cand_rows = 0;
    for (const auto * c : cand_rows) {
        n_cand_rows += c != nullptr ? 1 : 0;
    }

    // The frame-size bounds are the *sender's* to enforce, not the stage's: a stage
    // refuses an oversized frame only after these bytes are already on the wire, and its
    // ST_BAD_REQ then leaves the stream desynchronized -- the next reply is read as this frame's
    // body. Same two constants the stage checks with (llama-split-proto.h), so the two ends cannot
    // drift apart on what "too large" means.
    if (n_tok > qlh_split::MAX_TOKENS || n_embd > qlh_split::MAX_EMBD) {
        err = "decode frame too large to send (n_tokens " + std::to_string(n_tok) +
              ", n_embd " + std::to_string(n_embd) + "; MAX_TOKENS " +
              std::to_string(qlh_split::MAX_TOKENS) + ", MAX_EMBD " +
              std::to_string(qlh_split::MAX_EMBD) + ")";
        return -1;
    }

    // Ask for the layer-input taps of this frame. Empty/null ids means "send no tap
    // request", which is what a whole-model and a local chain always have.
    const size_t n_tap_ask = tap.ids != nullptr ? tap.ids->size() : 0;

    req_header req{};
    req.magic    = MAGIC;
    req.version  = PROTO_VER;
    req.op       = OP_DECODE;
    req.flags    = (uint8_t) ((n_cand_rows > 0    ? DECODE_FLAG_CANDIDATES : 0) |
                              (verify_rows       ? DECODE_FLAG_VERIFY_ROWS : 0) |
                              (nextn_rows        ? DECODE_FLAG_NEXTN_ROWS : 0) |
                              (n_tap_ask > 0     ? DECODE_FLAG_LAYER_INP   : 0));
    req.n_tokens = n_tok;
    req.n_embd   = n_embd;
    req.n_pos    = n_pos;
    // Only for a readable log line -- the stage takes each row's sequence from the
    // rows section below.
    req.seq_id   = rows.empty() ? 0 : rows[0].seq_id;

    if (!send_all(&req, sizeof(req)) ||
        !send_all(positions.data(), positions.size() * sizeof(int32_t)) ||
        !send_all(hidden.data(), hidden.size() * sizeof(float))) {
        err = "send failed";
        return -3;
    }

    if (n_tap_ask > 0) {
        // The tap request is a self-describing section of its own (see
        // layer_inp_req). The candidate block sits behind the rows section, so
        // the frame order is: layer-input block, rows, candidate block.
        static_assert(sizeof(uint32_t) == sizeof(int32_t), "layer ids travel as int32");
        layer_inp_req treq{};
        treq.n_taps = (uint32_t) n_tap_ask;
        treq.n_embd = tap.n_embd;
        std::vector<int32_t> ids(n_tap_ask);
        for (size_t i = 0; i < n_tap_ask; ++i) {
            ids[i] = (int32_t) (*tap.ids)[i];
        }
        if (!send_all(&treq, sizeof(treq)) ||
            !send_all(ids.data(), ids.size() * sizeof(int32_t))) {
            err = "send failed";
            return -3;
        }
    }

    // The per-row records follow the layer-input block, so the fixed-width blocks
    // stay first and the self-describing ones last (see the frame layout note in
    // llama-split-proto.h).
    if (!rows.empty() && !send_all(rows.data(), rows.size() * sizeof(row_req))) {
        err = "send failed";
        return -3;
    }

    // The candidate block is the frame's *last* section, behind the
    // rows it names -- the index has to be readable against rows the reader has already seen. One
    // block for the frame, one entry per row that asked, each with its own cand_req/bias/history.
    if (n_cand_rows > 0) {
        std::vector<cand_row_ref> refs;
        refs.reserve(n_cand_rows);
        std::vector<std::vector<uint8_t>> payloads;
        payloads.reserve(n_cand_rows);
        for (uint32_t i = 0; i < n_tok; ++i) {
            const llama_split_cand_req * c = cand_rows[i];
            if (c == nullptr) {
                continue;
            }
            // The candidate request is the tail's input for the steps *before* top_k. Its history
            // and biases are sent per frame rather than kept on the stage: see cand_req.
            cand_req cq = c->req;
            cq.n_bias = (uint32_t) c->bias.size();
            cq.n_hist = (uint32_t) c->hist.size();
            const size_t bytes = sizeof(cand_req) +
                                 c->bias.size() * sizeof(llama_logit_bias) +
                                 c->hist.size() * sizeof(llama_token);
            std::vector<uint8_t> payload(bytes);
            memcpy(payload.data(), &cq, sizeof(cq));
            size_t off = sizeof(cq);
            if (!c->bias.empty()) {
                memcpy(payload.data() + off, c->bias.data(),
                       c->bias.size() * sizeof(llama_logit_bias));
                off += c->bias.size() * sizeof(llama_logit_bias);
            }
            if (!c->hist.empty()) {
                memcpy(payload.data() + off, c->hist.data(),
                       c->hist.size() * sizeof(llama_token));
            }
            refs.push_back(cand_row_ref{ (int32_t) i, (uint32_t) bytes });
            payloads.push_back(std::move(payload));
        }
        const uint32_t n_refs = (uint32_t) refs.size();
        if (!send_all(&n_refs, sizeof(n_refs)) ||
            !send_all(refs.data(), refs.size() * sizeof(cand_row_ref))) {
            err = "send failed";
            return -3;
        }
        for (const auto & payload : payloads) {
            if (!send_all(payload.data(), payload.size())) {
                err = "send failed";
                return -3;
            }
        }
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return -3;
    }
    if (!check_reply_header(rsp, err)) {
        return -3;
    }

    switch (rsp.status) {
        case ST_OK: break;
        case ST_NO_SPACE: return 1;
        // The stage cannot answer this *frame's capability* (no h_nextn rows, no
        // draft head). The frame was consumed and nothing moved, so this is the same class of
        // answer as "no space": the caller falls back, the chain stays alive.
        case ST_UNSUPPORTED: return 1;
        case ST_BAD_REQ:  err = "stage rejected the request"; return -1;
        default:          err = "stage reported an internal error"; return -3;
    }

    // One answer carries one set of channels: whatever this reply does not fill below stays empty,
    // and a caller that asks a channel this reply did not answer reads an empty vector rather than
    // the previous frame's data. So the reply is cleared here, once, before any section is read.
    reply.clear();

    // A1: *where* each body section starts, and how wide it is, comes from layout_of() -- the same
    // table the stage used to produce rsp_header::n_bytes. This side used to derive the same
    // sections by subtracting them back out of n_bytes, which was the second implementation of one
    // rule: every protocol upgrade had to change both, and a section the writer learned about but
    // the reader did not would be read at the wrong offset (the next frame's header parsed out of
    // hidden rows).
    //
    // Two row counts go in, and they are not the same number (see body_layout): rsp.n_tokens is how
    // many rows this reply *answers* (also the row table's length), while n_tok is the frame's own
    // row count, which is the stride of the tap section -- a tap row exists for every position of the
    // batch even when the reply answers one of them.

    // The tap section this reply owes, and the check that it is exactly the shape this
    // frame asked for. A tap row is *per frame row*, not per answered row: the draft injects every
    // position of the batch, so n_rows is this request's own row count and never something the stage
    // reports. n_taps, by contrast, is the stage's answer -- it owns some of the ids and skips the
    // rest -- so it travels in the header.
    //
    // Two reply shapes carry taps. A tail answers one of the three output channels plus taps, and then
    // rsp.n_embd names the tap width (that is the only width it is *not* already stating elsewhere). A
    // relay answers hidden rows plus taps, and there rsp.n_embd is the hidden width -- the tap width is
    // the one this frame asked for, which is why the request carries it and the reply does not repeat
    // it.
    const bool reply_is_hidden = rsp.n_vocab == 0 && rsp.n_embd > 0 &&
                                 (rsp.flags & (RSP_FLAG_VERIFY_ROWS | RSP_FLAG_CANDIDATES |
                                               RSP_FLAG_NEXTN_ROWS)) == 0;
    const bool tap_on_hidden   = reply_is_hidden && (rsp.flags & RSP_FLAG_LAYER_INP) != 0;
    const uint32_t tap_wide    = tap_on_hidden ? tap.n_embd : rsp.n_embd;

    // A reply always answers at least one row, and never more rows than the frame
    // had. A stage that states otherwise is answering a different frame, and the row table below
    // would index a batch this side never sent.
    if (rsp.n_tokens == 0 || rsp.n_tokens > n_tok) {
        err = "stage answered " + std::to_string(rsp.n_tokens) + " rows for a " +
              std::to_string(n_tok) + "-row frame";
        return -3;
    }
    // The logits section answers a leading subset of the row table, so it cannot be wider than it.
    if (rsp.n_logits > rsp.n_tokens) {
        err = "stage answered " + std::to_string(rsp.n_logits) + " rows of logits for a " +
              std::to_string(rsp.n_tokens) + "-row answer";
        return -3;
    }

    // One table, both ends (llama-split-proto.h). The row width is the tap width: on a relay answer
    // that is the width the frame asked for, on every other shape it is rsp.n_embd.
    const body_layout lay = layout_of(rsp.flags, rsp.n_tokens, n_tok, tap_wide, rsp.n_vocab,
                                      rsp.n_taps, rsp.n_logits);

    uint64_t tap_bytes = 0;
    if (rsp.flags & RSP_FLAG_LAYER_INP) {
        if (n_tap_ask == 0) {
            err = "stage answered layer-input taps that were not requested";
            return -3;
        }
        if (rsp.n_taps == 0 || rsp.n_taps > n_tap_ask || rsp.n_taps > MAX_LAYER_INP_TAPS) {
            err = "stage answered " + std::to_string((unsigned) rsp.n_taps) + " layer-input taps for a " +
                  std::to_string(n_tap_ask) + "-tap request";
            return -3;
        }
        if (tap_wide == 0 || tap_wide != tap.n_embd) {
            err = "stage answered layer-input taps that are " + std::to_string(tap_wide) +
                  " wide, the frame asked for " + std::to_string(tap.n_embd);
            return -3;
        }
        // A tap block wider than one frame's rows * taps * n_embd would overflow n_bytes, so the
        // arithmetic is done in 64 bits and the frame is refused instead of truncated.
        tap_bytes = lay.tap_section_bytes64();
        if (tap_bytes > (uint64_t) rsp.n_bytes) {
            err = "stage answered " + std::to_string(rsp.n_bytes) + " bytes for a tap section of " +
                  std::to_string(tap_bytes) + " (" + std::to_string((unsigned) rsp.n_taps) +
                  " taps x " + std::to_string(n_tok) + " rows x " + std::to_string(tap_wide) + ")";
            return -3;
        }
    } else if (n_tap_ask > 0) {
        // Not an error: a stage that owns none of the named layers ignores the request, exactly like
        // a stage without a NextN block ignores DECODE_FLAG_NEXTN_ROWS. The caller decides what a
        // missing tap means.
    } else if (rsp.n_taps != 0) {
        err = "stage named layer-input taps in a reply that carries none";
        return -3;
    }

    // Self-check, the reading half, and it is an if rather than an assert: libllama is built Release
    // (/DNDEBUG), so an assert would be compiled out of exactly the binary that reads the wire. On a
    // reply that carries no candidates the body is exactly what the table says it is -- a stage built
    // from this tree always sets n_bytes to the table's total -- so this cannot fire on a valid frame.
    // It fires on a frame from anything else, which is the point: reading on would take the next
    // frame's header for this body. A candidate reply is exempt, because its section width is the
    // remainder and is checked below.
    if (!(rsp.flags & RSP_FLAG_CANDIDATES) && rsp.n_bytes != 0 && !lay.matches(rsp.n_bytes)) {
        err = "stage answered " + std::to_string(rsp.n_bytes) + " bytes for a " +
              std::to_string(lay.total) + "-byte body (flags " + std::to_string((unsigned) rsp.flags) +
              ", " + std::to_string(n_tok) + " rows, n_embd " + std::to_string(rsp.n_embd) +
              ", n_vocab " + std::to_string(rsp.n_vocab) + ", n_taps " +
              std::to_string((unsigned) rsp.n_taps) + ")";
        return -3;
    }

    // The body sections are read in wire order (see body_layout in llama-split-proto.h): the row
    // table, the per-row ids, the h_nextn rows, then candidates or the logits rows, and the
    // layer-input taps after all of them. That order is the protocol's, not a choice made here, and
    // the readers above implement it one section each.
    //
    // One exit from here on: the tap read is the last statement before the single `return 0`, so no
    // reply shape can skip the section that is last on the wire. The shapes used to leave through
    // five different `return 0`s, each of which had to remember to consume the taps first.
    const reply_frame f = {
        rsp, lay, n_tok, n_embd, tap_wide, tap_bytes, cand_rows, n_cand_rows,
        verify_rows, nextn_rows, reply_is_hidden, tap_on_hidden,
    };

    if (const int rc = read_rows(sock, f, reply, err))   { return rc; }
    if (const int rc = read_verify(sock, f, reply, err)) { return rc; }
    if (const int rc = read_nextn(sock, f, reply, err))  { return rc; }
    if (const int rc = read_output(sock, f, reply, err)) { return rc; }
    if (const int rc = read_tap(sock, f, reply, err))    { return rc; }
    return 0;
}

// One MTP draft round over the wire. See llama-split-remote.h for the contract;
// what is worth repeating here is the error split, because it is the same one the whole class
// makes: a *transport* failure poisons the chain (-1, the caller marks it dead), while every
// answer the stage gives that means "I cannot draft" is a 0 -- the request loses its drafts and
// keeps generating. Returning -1 for ST_UNSUPPORTED would turn "the tail has no NextN block" into
// "the chain is broken", which is exactly the failure mode this layer was built to avoid.
int llama_split_remote::mtp_draft(const std::vector<float> & h, int32_t token, int32_t pos, uint32_t k,
                                  int32_t seq_id, std::vector<int32_t> & out, std::string & err) {
    out.clear();

    if (!net::valid(sock)) {
        err = "not connected";
        return -1;
    }
    if (!have_confirmed || confirmed.has_nextn == 0) {
        // Not an error: this stage never said it could draft. The caller asked anyway (a stale
        // plan), and the honest answer is "no drafts".
        return 0;
    }
    // The k cap is not a protocol constant. The head asks for what its own
    // draft block is (common_params_speculative_draft::n_max) and the stage bounds it by the
    // output-row capacity it was built with -- that is the number a verify round has to fit, and it
    // is compared at attach time through stage_id::n_verify_rows. What is left here is a sanity
    // bound on a value that came off a plan, so a corrupt k cannot become a huge allocation.
    if (k == 0 || k > qlh_split::MAX_TOKENS) {
        err = "mtp draft k out of range (" + std::to_string(k) + ")";
        return -1;
    }
    if (h.size() != confirmed.n_embd_out) {
        err = "mtp draft hidden row is " + std::to_string(h.size()) + " wide, the stage is " +
              std::to_string(confirmed.n_embd_out);
        return -1;
    }

    req_header req{};
    req.magic    = MAGIC;
    req.version  = PROTO_VER;
    req.op       = OP_MTP_DRAFT;
    req.n_tokens = 1;
    req.n_embd   = confirmed.n_embd_out;
    req.n_pos    = 1;
    req.seq_id   = seq_id;
    req.n_bytes  = (uint32_t) sizeof(mtp_req);

    mtp_req mreq{};
    mreq.token = token;
    mreq.pos   = pos;
    mreq.k     = k;

    if (!send_all(&req, sizeof(req)) ||
        !send_all(&mreq, sizeof(mreq)) ||
        !send_all(h.data(), h.size() * sizeof(float))) {
        err = "send failed";
        return -1;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return -1;
    }
    if (!check_reply_header(rsp, err)) {
        return -1;
    }
    switch (rsp.status) {
        case ST_OK: break;
        case ST_UNSUPPORTED:
            // The capability is gone (the stage was restarted without its MTP context, or its
            // draft context failed where the handshake said it was up). Not a chain failure: the
            // stage consumed the frame and touched nothing.
            err = "the stage cannot draft (no MTP context)";
            return 0;
        case ST_NO_SPACE:
            err = "the stage has no room for the draft";
            return 0;
        case ST_BAD_REQ:
            err = "stage rejected the draft request";
            return -1;
        default:
            err = "stage reported an internal error while drafting";
            return -1;
    }

    if (rsp.n_tokens == 0 || rsp.n_tokens > k ||
        rsp.n_bytes != rsp.n_tokens * sizeof(int32_t)) {
        err = "stage answered " + std::to_string(rsp.n_tokens) + " drafts / " +
              std::to_string(rsp.n_bytes) + " bytes for a k=" + std::to_string(k) + " round";
        return -1;
    }
    out.resize(rsp.n_tokens);
    if (!recv_all(out.data(), out.size() * sizeof(int32_t))) {
        err = "recv drafts failed";
        return -1;
    }
    return (int) out.size();
}

bool llama_split_remote::seq_rm(int32_t seq_id, int32_t p0, int32_t p1, std::string & err) {
    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }

    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_SEQ_RM;
    req.seq_id  = seq_id;
    req.pos0    = p0;
    req.pos1    = p1;

    if (!send_all(&req, sizeof(req))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return false;
    }
    if (!check_reply_header(rsp, err)) {
        return false;
    }
    if (rsp.status != ST_OK) {
        err = "stage refused seq_rm";
        return false;
    }
    return true;
}

bool llama_split_remote::clear(std::string & err) {
    if (!net::valid(sock)) {
        err = "not connected";
        return false;
    }

    req_header req{};
    req.magic   = MAGIC;
    req.version = PROTO_VER;
    req.op      = OP_CLEAR;

    if (!send_all(&req, sizeof(req))) {
        err = "send failed";
        return false;
    }

    rsp_header rsp{};
    if (!recv_all(&rsp, sizeof(rsp))) {
        err = "recv failed";
        return false;
    }
    // Same header check as decode/seq_rm: without it a desynchronized stream (or a stage from
    // another build) is read as if it were a valid answer.
    if (!check_reply_header(rsp, err)) {
        return false;
    }
    if (rsp.status != ST_OK) {
        err = "stage refused clear";
        return false;
    }
    return true;
}
