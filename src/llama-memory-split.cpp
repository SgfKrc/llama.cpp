// [QLH-SPLIT] See llama-memory-split.h for the contract each method keeps.
#include "llama-memory-split.h"

#include "llama-context.h"
#include "llama-io.h"
#include "llama-split-proto.h"
#include "llama-split-remote.h"

#include <chrono>
#include <stdexcept>

using qlh_split::MAX_STATE;
using qlh_split::STATE_HANDLE_BYTES;

namespace {

// [QLH-SPLIT] Minimum spacing between two rebuild attempts. A rebuild on a chain whose stage is
// still down costs one bounded connect plus one bounded clear (about 1.5-3 s); without this
// window every request in a retry storm pays that again, and a client polling a broken chain
// turns into a connect storm against a machine that may be coming back. Five seconds is short
// enough that a stage which returned is picked up by the next request, and long enough that a
// burst of requests during the outage makes exactly one attempt (verified:
// verify/verify_chain_death.py, group F).
constexpr int64_t kRebuildThrottleMs = 5000;

int64_t steady_now_ms() {
    return (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

// [QLH-SPLIT] Rebuild a dead chain in place. See the header for why this can be safe at all: the
// batch has to start at position 0, because "all stages empty" is the only state that can be
// established over the wire, and an empty chain can only be written from the beginning.
bool llama_memory_split::try_rebuild(bool batch_starts_at_zero, std::string & err) {
    if (remote_failure.empty()) {
        return true;
    }
    if (remotes.empty()) {
        // A local chain cannot get here (remote_failure is only written from the paths that talk
        // to a stage), but answering instead of crashing keeps this a pure decision.
        err = remote_failure;
        return false;
    }

    // The interlock. Checked before the throttle and before any network call: an unsafe batch is
    // not a reason to touch a stage, and it must not consume the attempt budget either.
    if (!batch_starts_at_zero) {
        err = remote_failure + " (not rebuilt: this batch continues from positions the chain no "
                               "longer holds anywhere, so clearing it would decode into an empty "
                               "KV -- the request has to restart from position 0)";
        LLAMA_LOG_WARN("%s: rebuild not attempted: the batch does not start at position 0\n", __func__);
        return false;
    }

    const int64_t now = steady_now_ms();
    if (rebuild_last_ms >= 0 && now - rebuild_last_ms < kRebuildThrottleMs) {
        // Logged every time, with the numbers, so an operator (and the acceptance script) can
        // tell "the chain is down and we are waiting" from "the chain is down and we are
        // hammering it". `remote_failure` still names the original cause.
        LLAMA_LOG_WARN("%s: rebuild suppressed: %.1f s of %d s since attempt #%llu (chain still "
                       "dead: %s)\n", __func__,
                       (double) (now - rebuild_last_ms) / 1000.0, (int) (kRebuildThrottleMs / 1000),
                       (unsigned long long) rebuild_attempts, remote_failure.c_str());
        err = remote_failure + " (the reconnect is throttled; the next attempt is allowed " +
              std::to_string((int) ((kRebuildThrottleMs - (now - rebuild_last_ms)) / 1000.0 + 1)) +
              " s after the last one)";
        return false;
    }

    rebuild_last_ms = now;
    ++rebuild_attempts;

    LLAMA_LOG_WARN("%s: rebuild attempt #%llu: the chain was dead (%s); reconnecting %zu remote "
                   "stage(s) and clearing every stage\n", __func__,
                   (unsigned long long) rebuild_attempts, remote_failure.c_str(), remotes.size());

    // 1. Every remote has to come back *as itself*: reconnect() reconnects and then repeats the
    // v4 identity handshake against the identity confirmed at attach time. A stage that accepts
    // the TCP connection but reports a different layer range / model / depth is not the stage
    // this chain was built from, and the rebuild must fail rather than clear and serve from it.
    // The order matters only for the message: a stage that will not connect is named, and the
    // chain stays dead.
    for (size_t i = 0; i < remotes.size(); ++i) {
        std::string why;
        if (!remotes[i]->reconnect(why)) {
            err = "remote stage " + std::to_string(i + 1) + " (" + remotes[i]->endpoint() +
                  ") could not be reconnected: " + why;
            LLAMA_LOG_WARN("%s: rebuild attempt #%llu failed: %s\n", __func__,
                           (unsigned long long) rebuild_attempts, err.c_str());
            return false;
        }
    }

    // 2. All stages empty is the one state that can be re-established without moving any KV
    // between machines. clear() is the existing path for it -- it broadcasts to the head, the
    // local tails and every remote stage, and it is the operation that is allowed to drop
    // remote_failure (see its definition), so a clear that only half-succeeded leaves the chain
    // dead with the failing stage named.
    //
    // The chain keeps the reason it was dead for, not the reason the clear failed: a rebuild runs
    // on every request that finds the chain dead, so overwriting the flag with each attempt's
    // message would make the state say "clear failed" forever and lose the original cause (in the
    // measured case: the stage never answered a seq_rm). The attempt's own failure is reported
    // through `err`, which the caller logs, and through the log line below.
    const std::string was = remote_failure;
    clear(true);

    if (!remote_failure.empty()) {
        const std::string clear_why = remote_failure;
        remote_failure = was;
        err = "the chain is still dead (" + was + "); every stage was reconnected but the clear "
              "did not succeed: " + clear_why;
        LLAMA_LOG_WARN("%s: rebuild attempt #%llu failed: %s\n", __func__,
                       (unsigned long long) rebuild_attempts, err.c_str());
        return false;
    }

    // 3. A connection with no failure bound is not usable, and the rebuild socket was created
    // with the short one. Back to the data-path timeout before anything is decoded on it.
    for (size_t i = 0; i < remotes.size(); ++i) {
        if (!remotes[i]->restore_data_timeout()) {
            err = "remote stage " + std::to_string(i + 1) + " (" + remotes[i]->endpoint() +
                  ") is connected but could not be put back on the data-path timeout";
            LLAMA_LOG_WARN("%s: rebuild attempt #%llu failed: %s\n", __func__,
                           (unsigned long long) rebuild_attempts, err.c_str());
            remote_failure = err;
            return false;
        }
    }

    // 4. Every stage is empty and every stage is reachable again. The batch starts at position 0
    // (checked above), so decoding it now writes exactly the positions it claims.
    LLAMA_LOG_WARN("%s: rebuild attempt #%llu succeeded: every stage is empty again, decoding this "
                   "batch from position 0\n", __func__, (unsigned long long) rebuild_attempts);
    return true;
}

llama_memory_context_ptr llama_memory_split::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    // Only the head splits the batch; the tail stages receive the rows it produced.
    return head->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_memory_split::init_full() {
    return head->init_full();
}

llama_memory_context_ptr llama_memory_split::init_update(llama_context * lctx, bool optimize) {
    return head->init_update(lctx, optimize);
}

bool llama_memory_split::get_can_shift() const {
    // Shifting rewrites positions inside one stage's KV. Positions are the only thing tying
    // the stages together (there is no n_past to resync), so a shift on one stage and not
    // another would silently misalign the chain. Report false and let the caller degrade.
    return false;
}

void llama_memory_split::clear(bool data) {
    head->clear(data);
    for (auto * m : tails) {
        m->clear(data);
    }

    // [QLH-SPLIT] A failed clear used to poison the chain for the rest of the process: decode_split
    // refuses to run while remote_failure is set, and only re-attaching (which builds a fresh
    // view) ever cleared it, so one dropped frame meant "restart the server". That is too strong
    // for what the flag reports: clear() is the one operation that re-establishes a known state,
    // because when every stage answers it they are all empty and whatever a previous failed clear
    // left inconsistent is gone. So forget the previous failure here, and record one only if
    // *this* attempt fails again. A stage that comes back, or a transient send/recv failure,
    // therefore recovers on the next clear; until one succeeds the chain stays dead (there is
    // nothing to repair from here), and re-attaching rebuilds the view and starts trusted again.
    remote_failure.clear();

    // clear() has no error channel to report through, so a stage that cannot be reached leaves
    // the chain dead: decode_split checks remote_failure before driving it. Recorded once, and
    // with the endpoint, because the caller of llama_memory_clear() (warmup, or the server's
    // capability probe) has no way to tell which stage it was either.
    for (size_t i = 0; i < remotes.size(); ++i) {
        std::string err;
        if (!remotes[i]->clear(err) && remote_failure.empty()) {
            remote_failure = "remote stage " + std::to_string(i + 1) + " (" + remotes[i]->endpoint() +
                             ") refused clear: " + err;
        }
    }
}

bool llama_memory_split::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // [QLH-SPLIT] Once the chain is untrusted, no further trimming can make it trustworthy
    // again: the stages are position-split by construction, so the only way back is a clear
    // (every stage empty) or a re-attach. Answering without touching the network is what keeps
    // a dead chain from costing one socket timeout (measured: 120 s) per call -- exactly the
    // "every later request hangs" behaviour this flag exists to stop. Returning false is safe
    // here because common_context_seq_rm no longer aborts for a failed split chain and
    // decode_split refuses every decode while this is set, so no token is served from a KV the
    // chain no longer agrees on.
    if (!remote_failure.empty()) {
        return false;
    }

    // Every stage must drop the same range. A partial success is the one outcome that cannot
    // be allowed through: the head and a stage that kept the range would then compute the next
    // token from different positions. For a local chain common/common.cpp still aborts on false
    // (a local stage has no failure the process can survive: it is in-process state, not a
    // network peer); for a remote chain the false is recorded below and the caller reports a
    // request-level error instead of killing the server.
    bool ok = head->seq_rm(seq_id, p0, p1);
    for (auto * m : tails) {
        // Still inform every local stage even after a failure, so a chain that keeps running
        // (the remote case) has as much of the local state trimmed as possible.
        ok = m->seq_rm(seq_id, p0, p1) && ok;
    }
    for (size_t i = 0; i < remotes.size(); ++i) {
        std::string err;
        // Ask every stage even after a failure: the log has to name every stage that could not
        // be trimmed, not just the first one.
        if (!remotes[i]->seq_rm(seq_id, p0, p1, err)) {
            LLAMA_LOG_ERROR("%s: remote stage %zu (%s) failed to remove seq %d positions [%d, %d): "
                    "%s\n", __func__, i + 1, remotes[i]->endpoint().c_str(),
                    (int) seq_id, (int) p0, (int) p1, err.c_str());
            // [QLH-SPLIT] Mark the chain dead instead of letting the caller abort the process.
            //
            // The abort was right about the consequence -- after this, the head and the stage
            // hold different positions and every following token would be computed from state
            // one of them does not have -- and wrong about who pays for it: one network glitch
            // killed the whole server, every slot with it, while /health kept answering 200
            // (measured on a cross-host chain: /completion failed after 120.2 s and the head
            // process was gone).
            //
            // The consequence is unchanged, the carrier is: decode_split checks this flag
            // before computing anything and returns -3, so from here on every request fails as
            // a request. There is deliberately no path that clears it except a successful
            // clear() (all stages empty again -- the one consistent state this layer can
            // rebuild) or a re-attach, and neither can happen behind the caller's back. That is
            // what keeps "chain has diverged" from ever being served as success.
            if (remote_failure.empty()) {
                remote_failure = "remote stage " + std::to_string(i + 1) + " (" +
                                 remotes[i]->endpoint() + ") failed to remove seq " +
                                 std::to_string((int) seq_id) + " positions [" +
                                 std::to_string((int) p0) + ", " + std::to_string((int) p1) +
                                 "): " + err;
            }
            ok = false;
        }
    }
    return ok;
}

void llama_memory_split::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    head->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    for (auto * m : tails) {
        m->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    if (!remotes.empty()) {
        reject_remote("seq_cp");
    }
}

void llama_memory_split::seq_keep(llama_seq_id seq_id) {
    head->seq_keep(seq_id);
    for (auto * m : tails) {
        m->seq_keep(seq_id);
    }
    if (!remotes.empty()) {
        reject_remote("seq_keep");
    }
}

void llama_memory_split::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    head->seq_add(seq_id, p0, p1, shift);
    for (auto * m : tails) {
        m->seq_add(seq_id, p0, p1, shift);
    }
    if (!remotes.empty()) {
        reject_remote("seq_add");
    }
}

void llama_memory_split::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    head->seq_div(seq_id, p0, p1, d);
    for (auto * m : tails) {
        m->seq_div(seq_id, p0, p1, d);
    }
    if (!remotes.empty()) {
        reject_remote("seq_div");
    }
}

void llama_memory_split::reject_remote(const char * op) {
    if (remote_failure.empty()) {
        remote_failure = std::string(op) + " is not carried by the stage protocol; the remote "
                         "stages would keep a stale copy of the sequence";
    }
    LLAMA_LOG_ERROR("%s: %s\n", __func__, remote_failure.c_str());
}

llama_pos llama_memory_split::seq_pos_min(llama_seq_id seq_id) const {
    // Positions are identical by construction (decode_split copies batch.pos into every hop),
    // so any single stage answers for the chain. Use the head: it is the one that owns the
    // batch decomposition, so its answer is also the one that matches init_batch's view.
    return head->seq_pos_min(seq_id);
}

llama_pos llama_memory_split::seq_pos_max(llama_seq_id seq_id) const {
    return head->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_split::memory_breakdown() const {
    auto total = head->memory_breakdown();
    for (auto * m : tails) {
        for (const auto & it : m->memory_breakdown()) {
            total[it.first] += it.second;
        }
    }
    return total;
}

// The chain's state blob. For a local chain it is exactly what it always was -- a stage count,
// then one self-delimiting blob per stage, head first. A chain with remote stages adds one
// section, and it comes *before* the head:
//
//   uint32_t n_stages      = tails.size() + 1
//   uint32_t n_remotes                                   (only when remotes is non-empty)
//   n_remotes x { int32_t pos_max, uint32_t n_bytes, uint8_t bytes[n_bytes] }
//   head->state_write(...)
//
// [QLH-SPLIT] v6: `bytes` is the 8-byte handle the stage minted for its own snapshot
// (STATE_FLAG_HANDLE), so the whole remote section is 16 bytes per stage instead of 12.56 MiB.
// The n_bytes field is still there and still self-describing: a blob written by a pre-v6 build
// carries the state itself in that field, and state_read() still accepts it (a slot file
// persists checkpoints across builds, and parsing 12.56 MiB of recurrent state as a handle would
// be exactly the kind of silent misread this layer exists to prevent).
//
// Two reasons for that order. A restore must not touch the head until every remote stage has
// accepted its snapshot (the head is what answers pos_min/pos_max, so a head that moved while a
// stage did not is precisely the divergence this layer exists to prevent) -- writing the remote
// section first lets state_read apply it first while still consuming the stream front to back.
// And a save that fails costs nothing local: the head has not been written to the buffer yet.
//
// `pos_max` per stage is what makes a restore safe to trust: all stages are position-split by
// construction, so they must agree on it at save time, and a stage that has meanwhile moved
// backwards (cleared, or never got the tokens) is caught before its state is applied.
static constexpr size_t split_state_remote_hdr = sizeof(int32_t) + sizeof(uint32_t);

// [QLH-SPLIT] The chain's state operations all start with this: a dead chain has no state worth
// asking for, and asking costs one socket timeout per stage per call. Measured on a chain with a
// suspended stage: after the operation that discovered the failure, the server went on to query
// the stage for a checkpoint size and added a second 120 s to the same failing request. Refusing
// here keeps one failed request down to one timeout.
const char * llama_memory_split::state_unavailable() const {
    return remote_failure.empty() ? nullptr : remote_failure.c_str();
}

size_t llama_memory_split::state_size(llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (const char * why = state_unavailable()) {
        throw std::runtime_error(std::string("the chain is not trustworthy: ") + why);
    }

    size_t size = sizeof(uint32_t); // n_stages

    size_query_valid = false;
    size_query_bytes.clear();

    if (!remotes.empty()) {
        size += sizeof(uint32_t); // n_remotes
        size_query_bytes.assign(remotes.size(), 0);
        for (size_t i = 0; i < remotes.size(); ++i) {
            uint32_t n_bytes = 0;
            int32_t  pos_max = 0;
            std::string err;
            if (!remotes[i]->state_size(seq_id, flags, n_bytes, pos_max, err)) {
                throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                        remotes[i]->endpoint() + ") failed the state size query: " + err);
            }
            // [QLH-SPLIT] v6: this is STATE_HANDLE_BYTES (8), not the stage's 12.56 MiB -- the
            // stage is asked *with* STATE_FLAG_HANDLE, so it answers with the record it would
            // mint. The head's whole remote section is 16 bytes per stage; the state itself never
            // leaves the process that already holds it.
            size_query_bytes[i] = n_bytes;
            size += split_state_remote_hdr + n_bytes;
        }
        // Remembered for state_write(), which runs in a second round trip and has to detect that
        // a stage answered a different size than the one the caller's buffer was sized for.
        size_query_seq   = seq_id;
        size_query_flags = flags;
        size_query_valid = true;
    }

    // the head (and the local tails, if any) count themselves through a dummy io, exactly like
    // they did before remote stages existed
    llama_io_write_dummy io(false);
    head->state_write(io, seq_id, flags);
    for (auto * m : tails) {
        m->state_write(io, seq_id, flags);
    }
    size += io.n_bytes();

    return size;
}

void llama_memory_split::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (const char * why = state_unavailable()) {
        throw std::runtime_error(std::string("the chain is not trustworthy: ") + why);
    }

    // The chain's state is the concatenation of its stages'. Each stage writes its own
    // header-less blob; the split context wraps them so a later state_read can hand each
    // slice back to the stage that produced it.
    const uint32_t n = (uint32_t) tails.size() + 1;

    if (remotes.empty()) {
        io.write(&n, sizeof(n));
        head->state_write(io, seq_id, flags);
        for (auto * m : tails) {
            m->state_write(io, seq_id, flags);
        }
        return;
    }

    // Snapshot every stage before any of it is written: the head's position is the yardstick and
    // a stage that disagrees means the chain is already out of step, which is not something to
    // bake into a checkpoint that will be restored later.
    const llama_pos pos_head = head->seq_pos_max(seq_id);

    // [QLH-SPLIT] v6: 8 bytes per stage instead of 12.56 MiB. The stage keeps the snapshot in its
    // own cache and this is its name for it; the head never reads the bytes, it only stores them
    // and hands them back, so moving them was pure cost (measured: 0.8-1.2 s per checkpoint at
    // 16 MB/s, 2.8 s at the 5 MB/s of the field report).
    std::vector<uint64_t> handles(remotes.size());
    std::vector<int32_t>  pos_max(remotes.size());
    for (size_t i = 0; i < remotes.size(); ++i) {
        std::string err;
        if (!remotes[i]->state_take(seq_id, flags, handles[i], pos_max[i], err)) {
            throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                    remotes[i]->endpoint() + ") failed the state save: " + err);
        }
        if (pos_max[i] != pos_head) {
            throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                    remotes[i]->endpoint() + ") is at position " + std::to_string(pos_max[i]) +
                    " while the head is at " + std::to_string(pos_head) +
                    "; the chain has drifted, refusing to checkpoint it");
        }
    }

    // [QLH-SPLIT] The caller's buffer was sized by state_size(), which asked the stages for their
    // record size in an earlier round trip. With v6 that answer is a constant, so a mismatch now
    // means the two ends disagree about the layout (they are built together, so this is a build
    // skew or a corrupt build), not "the state changed in between". Checking it before the first
    // write is what keeps it a clean refusal instead of a buffer underflow in
    // llama_io_write_host::write, which reaches common/common.cpp as a 0-byte save.
    if (size_query_valid && size_query_seq == seq_id && size_query_flags == flags &&
            size_query_bytes.size() == remotes.size()) {
        for (size_t i = 0; i < remotes.size(); ++i) {
            if (size_query_bytes[i] != STATE_HANDLE_BYTES) {
                throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                        remotes[i]->endpoint() + ") answered " +
                        std::to_string(size_query_bytes[i]) + " bytes when the buffer was sized for "
                        "a snapshot handle (" + std::to_string(STATE_HANDLE_BYTES) + "); the two "
                        "ends disagree about the state layout, so this checkpoint cannot be written");
            }
        }
    }

    io.write(&n, sizeof(n));
    const uint32_t n_remotes = (uint32_t) remotes.size();
    io.write(&n_remotes, sizeof(n_remotes));
    for (size_t i = 0; i < remotes.size(); ++i) {
        const uint32_t n_bytes = STATE_HANDLE_BYTES;
        io.write(&pos_max[i], sizeof(pos_max[i]));
        io.write(&n_bytes, sizeof(n_bytes));
        io.write(&handles[i], sizeof(handles[i]));
    }

    head->state_write(io, seq_id, flags);
    for (auto * m : tails) {
        m->state_write(io, seq_id, flags);
    }
}

void llama_memory_split::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (const char * why = state_unavailable()) {
        throw std::runtime_error(std::string("the chain is not trustworthy: ") + why);
    }

    uint32_t n = 0;
    io.read(&n, sizeof(n));

    if (n != tails.size() + 1) {
        throw std::runtime_error("split memory state: stage count mismatch");
    }

    if (!remotes.empty()) {
        uint32_t n_remotes = 0;
        io.read(&n_remotes, sizeof(n_remotes));
        if (n_remotes != remotes.size()) {
            throw std::runtime_error("split memory state: remote stage count mismatch (" +
                    std::to_string(n_remotes) + " in the blob, " + std::to_string(remotes.size()) +
                    " attached)");
        }

        // [QLH-SPLIT] v6: a record is either a snapshot handle (8 bytes) or, from a slot file
        // written by a pre-v6 build, the state blob itself. Both are self-describing, so they are
        // still told apart by their length and the old file keeps working.
        std::vector<uint64_t>              handles(n_remotes, 0);
        std::vector<std::vector<uint8_t>>  blobs(n_remotes);
        std::vector<bool>                  is_handle(n_remotes, false);
        std::vector<int32_t>               pos_max(n_remotes);
        for (uint32_t i = 0; i < n_remotes; ++i) {
            uint32_t n_bytes = 0;
            io.read(&pos_max[i], sizeof(pos_max[i]));
            io.read(&n_bytes, sizeof(n_bytes));
            if (n_bytes > MAX_STATE) {
                throw std::runtime_error("split memory state: implausible remote stage size");
            }
            if (n_bytes == STATE_HANDLE_BYTES) {
                is_handle[i] = true;
                io.read(&handles[i], sizeof(handles[i]));
            } else {
                blobs[i].resize(n_bytes);
                if (n_bytes > 0) {
                    io.read(blobs[i].data(), n_bytes);
                }
            }
        }

        // Cleared here, not after the loop: it describes this attempt, and a caller that asks
        // after a failed restore must not read the answer of the previous one.
        state_stale = false;

        // Restore the remote stages first, head last: the caller reads its positions from the
        // head, so the head must not move unless every stage after it has.
        for (uint32_t i = 0; i < n_remotes; ++i) {
            std::string err;
            int32_t pos_before = -1;
            bool    stale      = false;
            const bool ok = is_handle[i]
                    ? remotes[i]->state_give(seq_id, flags, handles[i], pos_max[i], pos_before, stale, err)
                    : remotes[i]->state_load(seq_id, flags, blobs[i], pos_max[i], pos_before, err);
            if (!ok) {
                // [QLH-SPLIT] A restore that stops here leaves the chain split-brained: the
                // stages before this one were rewound to the snapshot, this one and the ones
                // after it were not, and the head (restored last) was not either. Nothing
                // later repairs that on its own -- a transport failure would show up again as
                // a timeout on the next operation -- so the chain is marked dead exactly like
                // a failed seq_rm. The throw is then a failed request (common/common.cpp /
                // tools/server report it) instead of a dead server; decode_split refuses from
                // here on, so the split brain can never be read from.
                if (remote_failure.empty()) {
                    remote_failure = "remote stage " + std::to_string(i + 1) + " (" +
                                     remotes[i]->endpoint() + ") failed the state restore: " + err;
                }
                throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                        remotes[i]->endpoint() + ") failed the state load: " + err);
            }
            if (stale) {
                // [QLH-SPLIT] v6: the stage no longer holds the snapshot this checkpoint names.
                // Unlike the branch above, *nothing* was applied -- the stage refused before it
                // touched its memory, and the head is read last -- so no stage moved and the
                // chain is not split-brained. The only thing that is gone is one checkpoint, so
                // this is reported separately (state_stale) and the server recovers the way it
                // already does for a server started without checkpoints: re-process the prompt.
                // Treating it as remote_failure would kill every following request for a cache
                // eviction.
                state_stale = true;
                throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                        remotes[i]->endpoint() + ") no longer holds the checkpoint's snapshot "
                        "(it was evicted, or the stage was restarted); the chain was not touched "
                        "and the prompt has to be re-processed");
            }
            // A stage that has moved backwards lost the prefix this snapshot sits on (a clear, a
            // restart, a different sequence), and restoring on top of that would silently
            // compute from positions it never decoded. Moving *forward* is the normal case: the
            // checkpoint is older than the current state by construction.
            if (pos_before < pos_max[i]) {
                if (remote_failure.empty()) {
                    remote_failure = "remote stage " + std::to_string(i + 1) + " (" +
                                     remotes[i]->endpoint() + ") is at position " +
                                     std::to_string(pos_before) + ", behind the checkpoint's " +
                                     std::to_string(pos_max[i]) +
                                     "; the chain has drifted and cannot be restored into it";
                }
                throw std::runtime_error("remote stage " + std::to_string(i + 1) + " (" +
                        remotes[i]->endpoint() + ") is at position " + std::to_string(pos_before) +
                        ", behind the checkpoint's " + std::to_string(pos_max[i]) +
                        "; refusing to restore the chain into it");
            }
        }
    }

    head->state_read(io, seq_id, flags);
    for (auto * m : tails) {
        m->state_read(io, seq_id, flags);
    }
}
