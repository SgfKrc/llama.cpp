// [QLH-SPLIT] See llama-split-remote.h for the error policy this implements.
#include "llama-split-remote.h"

#include "llama-split-net.h"
#include "llama-split-proto.h"

#include <cstdio>
#include <cstring>

using namespace qlh_split;

namespace {

// [QLH-SPLIT] The data-path read timeout, unchanged from when it was a local constant inside
// connect(): far above any honest stage latency (one -ubatch of prefill on CPU), finite only so a
// stage that stops answering becomes a failed request instead of a pinned slot.
constexpr int kStageTimeoutMs = 120 * 1000;

// [QLH-SPLIT] Budget for one rebuild handshake (connect, then the clear that follows it).
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

// [QLH-SPLIT] The reply header is checked on every operation; keeping the two failure messages in
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
std::string describe_identity(const stage_id & id) {
    return "layers [" + std::to_string(id.layer_lo) + ", " + std::to_string(id.layer_hi) + ") of a " +
           std::to_string(id.n_layer_src) + "-layer model (n_embd_inp " +
           std::to_string(id.n_embd_inp) + ", n_embd_out " + std::to_string(id.n_embd_out) +
           ", n_vocab " + std::to_string(id.n_vocab) + ", role " + role_name(id.role) + ")";
}

// What this position in the chain must be. `layer_hi` may be unknown for a stage that is not the
// chain's last one (see stage_id); that is shown as '?' rather than as a number, so the message
// does not claim to know something it does not.
std::string describe_expect(const stage_id & id) {
    const std::string hi = id.layer_hi == ID_ANY_I32 ? "?" : std::to_string(id.layer_hi);
    return "layers [" + std::to_string(id.layer_lo) + ", " + hi + ") of a " +
           std::to_string(id.n_layer_src) + "-layer model (n_embd_inp " +
           std::to_string(id.n_embd_inp) + ", n_embd_out " + std::to_string(id.n_embd_out) +
           ", n_vocab " + std::to_string(id.n_vocab) + ", role " + role_name(id.role) + ")";
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
    // [QLH-SPLIT] v6: the size the head has to make room for is the *handle record*, not the
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
        // A stage that answered with its whole blob here would be a pre-v6 one, but the version
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
            // Only after a completed HELLO: a v4 stage refuses any first frame that is not one,
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

    // [QLH-SPLIT] Nagle off for the life of the connection (see llama-split-net.h).
    //
    // This side had it from the start, which is exactly why the *up* direction never showed the
    // 48-52 ms gap the proxy trace caught: every request frame (header + 8 KB hidden rows) left
    // immediately, 0.1-0.2 ms apart. The stage side did not, so its 32 B header + 320 B
    // candidate body paid the delayed-ACK wait instead. Setting it here through the shared
    // helper keeps both ends from drifting apart again.
    net::set_nodelay(s);

    // [QLH-SPLIT] The stage protocol has no other timing bound, so a stage that stops answering
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
        // A stage that predates v4 checks the version on the first frame and answers BAD_REQ, so
        // this really is "no answer", not "wrong answer": name the build skew as the likely cause
        // instead of leaving the operator to blame the network.
        err = "the stage did not answer the identity handshake (it may be a pre-v4 split-stage, "
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
    if (!diff.empty()) {
        err = "stage identity mismatch: this stage is " + describe_identity(self) +
              ", but the chain expects " + describe_expect(expect) + " (" + diff + "). " +
              "It looks like the wrong stage is listening on " + ep + ".";
        return false;
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

int llama_split_remote::decode(const std::vector<float> & hidden,
                               const std::vector<int32_t> & positions,
                               uint32_t n_pos,
                               uint32_t n_embd,
                               int32_t seq_id,
                               const cand_req * cand,
                               const std::vector<llama_logit_bias> & cand_bias,
                               const std::vector<llama_token> & cand_hist,
                               bool verify_rows,
                               bool nextn_rows,
                               const llama_split_tap_req & tap,
                               std::vector<float> & out_hidden,
                               std::vector<float> & out_logits,
                               uint32_t & out_n_vocab,
                               std::vector<token_cand> & out_cands,
                               std::vector<int32_t> & out_verify,
                               std::vector<float> & out_nextn,
                               uint32_t & out_nextn_embd,
                               llama_split_tap_out & out_tap,
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

    // [QLH-SPLIT] The frame-size bounds are the *sender's* to enforce, not the stage's: a stage
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

    // [QLH-SPLIT] v9: ask for the layer-input taps of this frame. Empty/null ids means "send no tap
    // request", which is what a whole-model and a local chain always have.
    const size_t n_tap_ask = tap.ids != nullptr ? tap.ids->size() : 0;

    req_header req{};
    req.magic    = MAGIC;
    req.version  = PROTO_VER;
    req.op       = OP_DECODE;
    req.flags    = (uint8_t) ((cand != nullptr ? DECODE_FLAG_CANDIDATES : 0) |
                              (verify_rows       ? DECODE_FLAG_VERIFY_ROWS : 0) |
                              (nextn_rows        ? DECODE_FLAG_NEXTN_ROWS : 0) |
                              (n_tap_ask > 0     ? DECODE_FLAG_LAYER_INP   : 0));
    req.n_tokens = n_tok;
    req.n_embd   = n_embd;
    req.n_pos    = n_pos;
    req.seq_id   = seq_id;

    if (!send_all(&req, sizeof(req)) ||
        !send_all(positions.data(), positions.size() * sizeof(int32_t)) ||
        !send_all(hidden.data(), hidden.size() * sizeof(float))) {
        err = "send failed";
        return -3;
    }

    if (cand != nullptr) {
        // The candidate request is the tail's input for the steps *before* top_k. Its history
        // and biases are sent per frame rather than kept on the stage: see cand_req.
        cand_req c = *cand;
        c.n_bias = (uint32_t) cand_bias.size();
        c.n_hist = (uint32_t) cand_hist.size();
        if (!send_all(&c, sizeof(c))) {
            err = "send failed";
            return -3;
        }
        if (!cand_bias.empty() &&
            !send_all(cand_bias.data(), cand_bias.size() * sizeof(llama_logit_bias))) {
            err = "send failed";
            return -3;
        }
        if (!cand_hist.empty() &&
            !send_all(cand_hist.data(), cand_hist.size() * sizeof(llama_token))) {
            err = "send failed";
            return -3;
        }
    }

    if (n_tap_ask > 0) {
        // [QLH-SPLIT] v9: the tap request goes last in the frame, so a stage reads the candidate
        // block before it and both sections stay self-describing (see layer_inp_req).
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
        // [QLH-SPLIT] v8: the stage cannot answer this *frame's capability* (no h_nextn rows, no
        // draft head). The frame was consumed and nothing moved, so this is the same class of
        // answer as "no space": the caller falls back, the chain stays alive.
        case ST_UNSUPPORTED: return 1;
        case ST_BAD_REQ:  err = "stage rejected the request"; return -1;
        default:          err = "stage reported an internal error"; return -3;
    }

    out_hidden.clear();
    out_logits.clear();
    out_cands.clear();
    out_verify.clear();
    out_nextn.clear();
    out_tap.ids.clear();
    out_tap.rows.clear();
    out_tap.n_embd = 0;
    out_n_vocab    = 0;
    out_nextn_embd = 0;

    // [QLH-SPLIT] v8: the body sections are read in the order the protocol documents --
    // per-row ids, then h_nextn rows, then candidates or the logits row, and (v9) the layer-input
    // taps after all of them. So a reply can carry several channels without either side guessing.
    //
    // [QLH-SPLIT] A1: *where* each of those sections starts, and how wide it is, comes from
    // layout_of() -- the same table the stage used to produce rsp_header::n_bytes. This side used to
    // derive the same sections by subtracting them back out of n_bytes, which was the second
    // implementation of one rule: every protocol upgrade had to change both, and a section the
    // writer learned about but the reader did not would be read at the wrong offset (the next
    // frame's header parsed out of hidden rows).
    //
    // The layout is built from the *frame's* row count, not from rsp.n_tokens: a tap row exists for
    // every position of the batch, so a reply that answers one row still carries n_tok of them. The
    // h_nextn section is the one that follows the answered rows instead -- see body_layout.

    // [QLH-SPLIT] v9: the tap section this reply owes, and the check that it is exactly the shape this
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

    // One table, both ends (llama-split-proto.h). The row width is the tap width: on a relay answer
    // that is the width the frame asked for, on every other shape it is rsp.n_embd.
    const body_layout lay = layout_of(rsp.flags, n_tok, tap_wide, rsp.n_vocab, rsp.n_taps);

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

    // [QLH-SPLIT] v9: the tap section is last, so every early exit below has to consume it first.
    // One lambda instead of four copies of the same read.
    const auto read_tap = [&]() -> int {
        if (!(rsp.flags & RSP_FLAG_LAYER_INP)) {
            return 0;
        }
        out_tap.ids.resize(rsp.n_taps);
        if (!recv_all(out_tap.ids.data(), out_tap.ids.size() * sizeof(uint32_t))) {
            err = "recv layer-input tap ids failed";
            return -3;
        }
        out_tap.rows.resize((size_t) n_tok * rsp.n_taps * tap_wide);
        if (!out_tap.rows.empty() &&
            !recv_all(out_tap.rows.data(), out_tap.rows.size() * sizeof(float))) {
            err = "recv layer-input tap rows failed";
            return -3;
        }
        out_tap.n_embd = tap_wide;
        return 0;
    };

    if (reply_is_hidden && !tap_on_hidden) {
        out_hidden.resize((size_t) rsp.n_tokens * rsp.n_embd);
        if (!recv_all(out_hidden.data(), out_hidden.size() * sizeof(float))) {
            err = "recv hidden failed";
            return -3;
        }
        return 0;
    }

    // [QLH-SPLIT] v9: a relay that owns taps answers its hidden rows first and the tap block last.
    if (tap_on_hidden) {
        if (rsp.n_tokens != n_tok || rsp.n_embd != n_embd) {
            err = "stage answered " + std::to_string(rsp.n_tokens) + " hidden rows of " +
                  std::to_string(rsp.n_embd) + " for a " + std::to_string(n_tok) + "-row frame of " +
                  std::to_string(n_embd);
            return -3;
        }
        const uint64_t hidden_bytes = (uint64_t) n_tok * n_embd * sizeof(float);
        // [QLH-SPLIT] A1: the body is exactly what the layout table says it is -- hidden rows plus
        // the tap section -- and this is the strict form of that check (a relay answer states both
        // widths in its header, so nothing here is inferred).
        if (!lay.matches(rsp.n_bytes)) {
            err = "stage answered " + std::to_string(rsp.n_bytes) + " bytes for " +
                  std::to_string(hidden_bytes) + " hidden bytes plus " + std::to_string(tap_bytes) +
                  " of taps";
            return -3;
        }
        out_hidden.resize((size_t) n_tok * n_embd);
        if (!recv_all(out_hidden.data(), out_hidden.size() * sizeof(float))) {
            err = "recv hidden failed";
            return -3;
        }
        if (read_tap() != 0) {
            return -3;
        }
        return 0;
    }

    if (rsp.flags & RSP_FLAG_VERIFY_ROWS) {
        // The tail answered the per-row request: one int32 token id per row of the frame it was
        // handed, in row order. The frame carried no extra bytes, so the count is the request's
        // own n_tokens -- checking it catches a stage that answered a different frame number
        // rather than letting the caller index an array that does not describe its batch.
        if (!verify_rows) {
            err = "stage answered per-row results that were not requested";
            return -3;
        }
        if (rsp.n_tokens != n_tok || rsp.n_bytes < lay.total) {
            err = "stage answered " + std::to_string(rsp.n_tokens) + " rows / " +
                  std::to_string(rsp.n_bytes) + " bytes for a " + std::to_string(n_tok) +
                  "-row verify frame, expected at least " +
                  std::to_string(n_tok * sizeof(int32_t)) + " bytes";
            return -3;
        }
        out_verify.resize(n_tok);
        if (!recv_all(out_verify.data(), out_verify.size() * sizeof(int32_t))) {
            err = "recv per-row tokens failed";
            return -3;
        }
    }

    // nextn and layer-input taps are mutually exclusive on the wire (one reply has one n_embd field),
    // so a stage that set both is answering a frame no build of this tree sends.
    if ((rsp.flags & RSP_FLAG_NEXTN_ROWS) && (rsp.flags & RSP_FLAG_LAYER_INP)) {
        err = "stage answered both h_nextn rows and layer-input taps in one reply";
        return -3;
    }

    if (rsp.flags & RSP_FLAG_NEXTN_ROWS) {
        if (!nextn_rows) {
            err = "stage answered h_nextn rows that were not requested";
            return -3;
        }
        // The stage decides how many rows it owes (all rows of a verify frame, the last one
        // otherwise) and says so through n_tokens, which is the frame's row count for a verify
        // reply and 1 for a plain one. Both are checked against the width, so a stage that
        // answered a different shape is caught here instead of shifting every later reply.
        const uint32_t n_rows = rsp.n_tokens;
        if (rsp.n_embd != n_embd || n_rows == 0 ||
            (size_t) lay.nextn_bytes != (size_t) n_rows * n_embd * sizeof(float) ||
            (!(rsp.flags & RSP_FLAG_VERIFY_ROWS) && n_rows != 1)) {
            err = "stage answered implausible h_nextn rows (n_tokens " + std::to_string(rsp.n_tokens) +
                  ", n_embd " + std::to_string(rsp.n_embd) + ", n_bytes " +
                  std::to_string(rsp.n_bytes) + ")";
            return -3;
        }
        out_nextn.resize((size_t) n_rows * rsp.n_embd);
        if (!recv_all(out_nextn.data(), out_nextn.size() * sizeof(float))) {
            err = "recv h_nextn rows failed";
            return -3;
        }
        out_nextn_embd = rsp.n_embd;
    }

    if (rsp.flags & RSP_FLAG_VERIFY_ROWS) {
        // Ids, the h rows (when asked) and the taps (when asked) were consumed above; nothing else
        // follows a verify reply.
        if (read_tap() != 0) {
            return -3;
        }
        return 0;
    }

    if (rsp.flags & RSP_FLAG_CANDIDATES) {
        // The tail answered the candidate request: n_bytes worth of (id, logit) pairs, already
        // sorted by descending logit. A tail that refuses (a v4 build) fails the version check
        // above; one that answers the row instead would leave this branch empty and out_logits
        // filled, which decode_split() accepts as the slow-but-correct path -- the one thing it
        // must not do is invent candidates.
        if ((uint64_t) rsp.n_bytes < lay.total) {
            err = "stage answered " + std::to_string(rsp.n_bytes) + " bytes for " +
                  std::to_string(lay.nextn_bytes) + " of h rows and " +
                  std::to_string(tap_bytes) + " of taps";
            return -3;
        }
        // [QLH-SPLIT] A1: the candidates are the one section whose width the layout table cannot
        // know -- it is the chain's top_k, and no header field carries it. Everything else comes
        // from the table, so the count is the remainder, not a re-derivation of the body.
        const uint32_t cand_bytes = rsp.n_bytes - lay.total;
        if (cand_bytes % sizeof(token_cand) != 0 || cand_bytes == 0) {
            err = "stage answered an implausible candidate payload (" +
                  std::to_string(cand_bytes) + " bytes)";
            return -3;
        }
        out_cands.resize(cand_bytes / sizeof(token_cand));
        if (!recv_all(out_cands.data(), out_cands.size() * sizeof(token_cand))) {
            err = "recv candidates failed";
            return -3;
        }
        if (cand == nullptr) {
            // The stage must not volunteer candidates: every later row would be sampled from a
            // set nobody validated.
            err = "stage answered candidates that were not requested";
            return -3;
        }
        if (read_tap() != 0) {
            return -3;
        }
        return 0;
    }

    const uint64_t logits_bytes = (uint64_t) rsp.n_vocab * sizeof(float);
    // A logits reply has never carried n_bytes (the op predates the field being used for replies):
    // the row is exactly n_vocab floats and the old parse read it without looking. So the tap
    // arithmetic is only checked when the stage did set n_bytes -- a stage that sets it has to cover
    // the row plus the taps, and one that leaves it 0 is answering the v8 shape, which is what the
    // head parsed before this channel existed.
    if (rsp.n_vocab > 0 && rsp.n_bytes > 0 &&
        (uint64_t) rsp.n_bytes < lay.total) {
        err = "stage answered " + std::to_string(rsp.n_bytes) + " bytes for a " +
              std::to_string(logits_bytes) + "-byte logits row and " +
              std::to_string(tap_bytes) + " of taps";
        return -3;
    }
    out_logits.resize(rsp.n_vocab);
    if (rsp.n_vocab > 0 && !recv_all(out_logits.data(), out_logits.size() * sizeof(float))) {
        err = "recv logits failed";
        return -3;
    }
    out_n_vocab = rsp.n_vocab;
    if (read_tap() != 0) {
        return -3;
    }
    return 0;
}

// [QLH-SPLIT] v8: one MTP draft round over the wire. See llama-split-remote.h for the contract;
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
    if (k == 0 || k > MAX_MTP_DRAFT) {
        err = "mtp draft k out of range";
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
