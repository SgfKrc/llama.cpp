// [QLH-SPLIT] Memory view over a chain of stages.
//
// A split context has one KV cache per stage, but llama-server talks to a single
// llama_memory_t. This object presents the union: queries that must be consistent across the
// chain are answered from the head, mutations are broadcast to every stage, and anything that
// would move positions inside a stage (shifting) is refused outright because the stages would
// diverge silently.
//
// Design notes worth keeping:
//  - init_batch / init_full / init_update go to the head only. The ubatch decomposition is a
//    property of the batch and the head's capacity; the tail stages consume the hidden rows
//    that the head already split, so they never get to choose differently.
//  - seq_rm must succeed on EVERY stage, and a stage that dropped a range while another kept it
//    is exactly the divergence this whole layer exists to prevent. What happens to a false
//    return depends on the kind of stage: a local chain still aborts in common/common.cpp (the
//    process is the only thing that can be inconsistent, and it cannot be repaired in place),
//    while a failed REMOTE stage marks the chain dead (remote_failure) and the request fails.
//    The difference is the failure being transient and outside the process (a dropped frame)
//    versus in-process state that is definitely wrong. Neither path ever serves a token from a
//    diverged chain: decode_split checks the flag first.
//  - can_shift is false. Shifting rewrites positions inside one stage's KV; with positions
//    being the only thing tying the stages together, that cannot be done safely, and the
//    server already degrades gracefully when can_shift is false.
//  - state_write / state_read are the whole chain. llama-server's context checkpoints save and
//    restore the sequence state, and a checkpoint that covered only the head would leave the
//    tails at their pre-checkpoint positions: the head would look rewound (pos_min/pos_max come
//    from the head) while the next truncating seq_rm still had to rewind a tail's recurrent
//    state, which no finite n_rs_seq could honour. llama_context therefore routes its state
//    through get_memory(), not its `memory` member -- keep that way.
//    Remote stages join that blob (see llama-memory-split.cpp for the layout). The wire carries
//    the recurrent state only, because for a model without SWA the checkpoint flavour
//    (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) writes no KV at all -- llama-kv-cache.cpp returns
//    immediately. That is why the per-stage blob is a constant 6.4 MiB for an 8-layer qwen35
//    range no matter how long the prompt is.
//
//    [v6] That blob is no longer *moved* through the head. A remote stage is asked to keep the
//    snapshot and answers with a handle, so the record the head stores per stage is 16 bytes
//    instead of 12.56 MiB, and a restore is 8 bytes up instead of the whole blob. The bytes on
//    both sides of the change are the same state -- the stage snapshots exactly what it used to
//    send -- so a checkpoint still restores bit for bit. See STATE_FLAG_HANDLE / ST_STALE in
//    llama-split-proto.h and `state_stale` below for what a missing handle means.
//  - Remote stages are held here too, as borrowed pointers, so that the server's KV bookkeeping
//    (seq_rm in particular) reaches them. A remote stage has no local memory object to call, so
//    the dispatch is a separate loop over `remotes` rather than a second kind of tail.
//
// What a remote stage can and cannot do with positions -- this is the boundary the error policy
// in llama-split-remote.h is built on:
//  - it CAN append: decode_split hands it every token's position verbatim, so its KV is
//    position-identical to the head's by construction.
//  - it CAN drop everything (p0 = 0): a full clear needs no history, on any memory type.
//  - it CANNOT rewind arbitrarily: removing [p0, p1) needs the recurrent state behind p0, and a
//    stage only keeps n_rs_seq tokens of snapshots. In practice llama-server does not even ask:
//    for a hybrid like qwen35 the aggregate memory reports pos_min == pos_max == the last
//    position after any decode (measured on the whole model: 80 prompt tokens -> 79/79, 96 ->
//    95/95), so as soon as the shared prefix ends before the cached one, its reuse path takes
//    the "restore a context checkpoint, otherwise re-process the prompt" branch. A remote chain
//    can take the same branch: its checkpoints carry the stages' state too (state_write /
//    state_read below), so the restore path works over the wire. The refusal in seq_rm is for
//    the general case: a model or server path where the distance is real is exactly the case
//    where carrying on would compute from positions the head no longer has.
//  - it CAN restore a checkpoint, but only the checkpoint flavour: the wire carries the recurrent
//    state (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY), and llama_context::state_seq_get_size reports 0
//    for the full flavour of a remote chain, so prompt_cache (full states) is skipped and the
//    server's context checkpoints (partial) are the ones that work. A restore that fails part-way
//    is recorded as a dead chain: the stages before the failure were rewound and the others were
//    not, and no later operation can repair that. That is what the throw in state_read() becomes
//    -- a failed request with the reason, not a dead server.
//    (The numbers below were measured before remote stages could carry their checkpoint state;
//     they are kept for the design rationale, not as the current cost model -- the partial
//     flavour does round-trip now, see [G2] in tests/split-smoke.cpp.)
//    What this costs, measured with a server on either side of the same model:
//     - a prompt that continues the previous token stream exactly (the client appends what the
//       model produced) reuses everything: 33-token second turn -> prompt_n = 1, 50 ms, same as
//       the same server without remote stages;
//     - a chat request whose history the client re-renders (messages -> template, as every chat
//       API client does) reuses nothing: 111-token third turn -> prompt_n = 111, 1070 ms, where
//       the local server restores a checkpoint at the last user message and decodes 45
//       (274 ms). The gap is the conversation history, so it grows with the session.
//    Closing that gap means letting a stage keep its own checkpoints -- a protocol change that
//    does not move KV over the wire, but a change nonetheless. Until then a remote chain is for
//    append-only prompt reuse, not for long re-rendered chat histories.
#pragma once

#include "llama-memory.h"

#include <cstdint>

// Only pointers are stored: the wire client drags in the socket headers, and this view is
// included far more widely than the stage protocol needs to be.
struct llama_split_remote;

struct llama_memory_split : llama_memory_i {
    // Not owned. `head` is the context this view was installed on.
    llama_memory_t              head;
    std::vector<llama_memory_t> tails;
    // Not owned. llama_context::split_remotes owns these and outlives this view; an empty list
    // means a local chain, exactly like before remote stages existed.
    //
    // mutable because state_write() is a const override but has to ask each stage for its
    // snapshot: the vector is a list of handles, not chain state, and the calls only read the
    // stage. (state_read() is non-const and needs no such treatment.)
    mutable std::vector<llama_split_remote *> remotes;

    llama_memory_split(llama_memory_t head,
                       const std::vector<llama_memory_t> & tails,
                       const std::vector<llama_split_remote *> & remotes)
        : head(head), tails(tails), remotes(remotes) {}

    // Non-empty once a remote stage failed an operation the chain cannot survive: clear() (no
    // error channel at all), or seq_rm(), whose false still aborts a *local* chain in
    // common/common.cpp but must not kill the process for a remote one. The chain cannot be
    // repaired from here, so decode_split refuses to run and reports this string, and seq_rm
    // answers false without touching the network: the alternatives are serving hidden states
    // from a KV that was never cleared, or one socket timeout of latency per request.
    //
    // One failure is not permanent any more, but until it is repaired the chain is dead: the
    // flag is cleared when a later clear() succeeds on every stage (all stages empty again is
    // the consistent state this layer can re-establish, see llama_memory-split.cpp) or when the
    // chain is rebuilt by re-attaching, which constructs a fresh view. What it must not do is
    // clear itself on an unrelated success: a decode that happens to work says nothing about
    // whether the stage that missed the clear still holds the old KV.
    //
    // Read it through llama_context_split_failed()/llama_context_split_failure()
    // (src/llama-ext.h) rather than directly: this string is written by the thread that drives
    // the chain, and those two are the declared way for the rest of the tree to ask.
    std::string remote_failure;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // [QLH-SPLIT] Self-heal: re-establish a chain that was marked dead, in place.
    //
    // A dead chain is one where a stage stopped answering (or answered wrong) while the others
    // kept going: from that moment no stage's KV agrees with another's, and the only state this
    // layer can put *every* stage into without moving data between machines is "all empty".
    //
    // So a rebuild is: reconnect every remote stage (close without BYE, bounded connect), then
    // clear the whole chain, then hand the connection back to the data-path timeout. When this
    // returns true the head, the local tails and every remote stage hold nothing, and the caller
    // may decode a batch into it.
    //
    // `batch_starts_at_zero` is the safety interlock, and it is not advisory: an empty chain can
    // only be written from position 0. A batch that continues from positions the chain no longer
    // holds anywhere would decode into a KV that is empty while the positions keep counting --
    // exactly the silent-wrong-token state this layer exists to prevent. In that case nothing is
    // tried (no network, no reconnect) and the call answers false; the caller fails the request,
    // the server's existing decode-error path drops the slot's cached prompt, and the next
    // request starts at 0 and can be rebuilt.
    //
    // Returns true only when the chain is healthy again *and* usable for this batch. False
    // leaves remote_failure set (with a reason) and the chain dead.
    bool try_rebuild(bool batch_starts_at_zero, std::string & err);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Size in bytes that state_write() would produce, without producing it.
    //
    // llama_context::state_seq_get_size needs it before the buffer exists, and for a chain with
    // remote stages a plain dummy write is not enough: it would have to fetch every stage's
    // snapshot just to count it. This asks the stages for sizes instead (one round trip each).
    size_t state_size(llama_seq_id seq_id, llama_state_seq_flags flags) const;

    // What the last state_size() asked the remote stages for.
    //
    // The size and the blob are two separate round trips: llama_context::state_seq_get_size
    // sizes the caller's buffer, state_seq_get_data fills it. A stage whose state changes in
    // between answers the second call with a different byte count, and the caller's buffer was
    // sized from the first answer. state_write() compares against these and fails *before it
    // writes anything* (every stage's blob is fetched before the first write), so the failure
    // is a clean "this round cannot be checkpointed" instead of a buffer underflow in
    // llama_io_write_host::write, which reaches common/common.cpp as a 0-byte save.
    //
    // mutable because state_write() is a const override and this is a record of the last query,
    // not chain state.
    mutable bool                  size_query_valid = false;
    mutable llama_seq_id          size_query_seq   = -1;
    mutable llama_state_seq_flags size_query_flags = 0;
    mutable std::vector<uint32_t> size_query_bytes;

    // [QLH-SPLIT] v6: the last state_read() found a snapshot the stage no longer holds.
    //
    // This is *not* remote_failure. A stale handle means the stage evicted the snapshot (its
    // cache is bounded) or the stage is not the process that minted it; the stage refuses before
    // applying anything, and the head has not been read yet either, so every stage is still
    // exactly where it was. The chain is intact; the only thing that is gone is a checkpoint, and
    // llama-server already has a path for "no cache data": re-process the prompt. The server asks
    // through llama_context_split_state_stale() (src/llama-ext.h) and must keep that apart from
    // remote_failure, which is fatal for the request.
    //
    // Cleared at the start of every state_read(), so it always describes the attempt that just
    // failed rather than a previous one.
    mutable bool state_stale = false;

private:
    // remote_failure as a C string, nullptr while the chain is healthy. Every state operation
    // starts by asking: a dead chain's state must not be asked for (one socket timeout per stage
    // per call, for an answer nobody can use). See the definition for the measurement.
    const char * state_unavailable() const;

    // Rebuild bookkeeping (try_rebuild). `rebuild_last_ms` is a steady_clock stamp of the last
    // attempt, -1 before the first one; `rebuild_attempts` exists so the log can count attempts
    // instead of leaving an operator to guess whether the throttle is working.
    int64_t  rebuild_last_ms   = -1;
    uint64_t rebuild_attempts  = 0;

    // Makes the chain unusable for the next decode, with `op` in the reason. Used by the
    // operations the stage protocol does not carry: they are unreachable from llama-server today
    // (see the position notes at the top of this header), but if one ever arrives, the stages
    // would keep a stale copy of each other and every later hidden state would be wrong with
    // nothing to detect it.
    void reject_remote(const char * op);
};
