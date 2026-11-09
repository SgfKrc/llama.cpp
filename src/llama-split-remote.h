// [QLH-SPLIT] Client side of the stage protocol: a remote stage, seen from the driving process.
//
// This class exists so that decode_split does not have to know whether a stage is local or
// over the network. It owns one connection to one split-stage process and offers exactly the
// operations the chain needs: decode a batch of hidden rows, trim a sequence, clear.
//
// Error policy, and the reason it is not "retry until it works":
//  - seq_rm has one failure exit. A transport failure and a stage that refuses the range are
//    the same event for the chain: the KV of one stage is about to disagree with the others,
//    and every later token's hidden state would be computed from a position the head no longer
//    has. That is remembered as the chain's failure reason (llama_memory_split::remote_failure,
//    set in llama-memory-split.cpp) and reported to common/common.cpp, which fails the request
//    for a remote chain and still aborts for a local one -- a local stage cannot be blamed on
//    the network, so there is nothing to recover. Reporting "the stage refused" differently
//    from "we could not ask" would only move that decision to a caller that does not exist.
//  - a decode transport failure maps to -3 ("compute failed"), which is what llama-server
//    already understands as a hard error for that request. It does not retry: a stage that
//    dropped mid-decode cannot be resumed without corrupting positions, and the chain has no
//    n_past to resynchronize from.
//  - clear has no error channel at all (llama_memory_i::clear returns void), so a failed clear
//    is recorded as a dead-chain reason and poisons the next decode instead of being dropped.
//  - the state ops are part of the checkpoint path, where a failure is not a failed request but
//    a chain that must not keep serving: they report the failure to the caller, which turns it
//    into a failed state restore (common/common.cpp and tools/server treat that as fatal for a
//    checkpoint).
#pragma once

#include "llama.h"
#include "llama-split-net.h"
#include "llama-split-proto.h"

#include <cstdint>
#include <string>
#include <vector>

// [QLH-SPLIT] Why this class carries the "ask for candidates" flag instead of the sampler doing
// the trip itself: the sampler lives in common/ and has no idea a stage exists; this is the only
// layer that knows a decode is going over a socket. Keeping the wire choice here is also what
// lets the fallback rule stay on the driving side, where the sampler chain is.

// [QLH-SPLIT] v9: what one frame asks of the layer-input tap channel.
//
// `ids` are *source-model* layer ids, and every stage of a chain is sent the same list: each one
// answers the ids inside its own [layer_lo, layer_hi) and drops the rest (see
// DECODE_FLAG_LAYER_INP). That is why this side never has to know where the chain was cut beyond
// what it already knows -- the head's own range -- and why a relay that owns taps answers them like
// any tail would.
struct llama_split_tap_req {
    const std::vector<uint32_t> * ids    = nullptr; // nullptr or empty: do not ask
    uint32_t                      n_embd = 0;       // trunk width of one tap row
};

// The taps a stage answered for one frame. `ids` holds the source-layer ids it owned, in reply
// order, and `rows` is n_rows * ids.size() * n_embd floats: the first tap's row for batch row 0,
// the second tap's row for row 0, ..., then row 1. Empty means "this stage had nothing to answer",
// which is not an error -- a stage that owns none of the named layers ignores the request.
struct llama_split_tap_out {
    std::vector<uint32_t> ids;
    std::vector<float>    rows;
    uint32_t              n_embd = 0;
};

struct llama_split_remote {
    llama_split_remote() = default;
    ~llama_split_remote();

    llama_split_remote(const llama_split_remote &) = delete;
    llama_split_remote & operator=(const llama_split_remote &) = delete;

    // Connects to host:port. Returns false on failure; `err` gets a human-readable reason.
    //
    // This only opens the socket. Every connection then starts with one OP_HELLO (hello() below),
    // because a TCP connection says nothing about which stage answered: a stage restarted with a
    // different --layer-range, a different model or a different depth would otherwise be driven
    // as if it were the one the chain was built from, and every later token would be wrong with
    // no error anywhere. Attach calls connect() then hello(); reconnect() repeats both.
    //
    // PROTO_VER is still checked on every reply header as well. A stage from another build is
    // caught there, and now also at the first frame (its HELLO is refused, or it refuses ours),
    // which is what makes a version skew a legible protocol error instead of a misread reply.
    bool connect(const std::string & endpoint, std::string & err);

    // [QLH-SPLIT] v4 identity handshake: sends `expect` and compares the stage's answer with it.
    //
    // `expect` is what the caller knows this position in the chain must be. Fields the caller
    // cannot pin yet are ID_ANY_I32 (see llama-split-proto.h); role is pinned by position (every
    // stage but the last is a relay). `got` (optional) receives the stage's own identity even
    // when the comparison fails, so a caller can show both sides in one message.
    //
    // On success this records the answer as `confirmed`, which is what reconnect() later demands
    // bit for bit: a stage that answers differently after a restart is not the stage this chain
    // was built from, even if the new answer would be a valid stage somewhere else in a chain.
    //
    // On failure `err` is written to be printed as-is: it names the mismatching numbers and ends
    // with what an operator should check (the wrong stage listening on the endpoint).
    bool hello(const qlh_split::stage_id & expect, qlh_split::stage_id * got, std::string & err);

    // [QLH-SPLIT] Re-establishes the connection to the endpoint this client was attached to,
    // after the chain was marked dead, and proves it got the same stage back.
    //
    // Two differences from connect(), both load-bearing:
    //  - it does NOT send BYE first. BYE is best-effort politeness for a peer that is going
    //    away; after a failure the peer is by definition not answering, so a BYE is a write into
    //    a socket nobody is draining -- buffered at best, blocked for the whole write timeout at
    //    worst, before the reconnect even starts.
    //  - every wait is bounded by a rebuild budget instead of the data-path timeout. A stage
    //    that accepts the connection but never answers (suspended: the kernel still completes
    //    the handshake out of the listen backlog) has to fail the attempt in seconds, not in
    //    120 s, or the first request after a failure pays for the whole outage.
    //
    // The identity check runs inside that budget: after the TCP connect, the connection is
    // greeted again and the answer must equal the identity confirmed at attach time. A stage
    // restarted with a different --layer-range / model / depth answers the handshake and is
    // then refused -- the chain stays dead and this returns false with the reason. Treating
    // "connected again" as "the chain is back" is the silent-wrong-token bug this check exists
    // to make impossible, so no caller may skip it.
    //
    // On success the socket carries the short timeout, and restore_data_timeout() puts it back on
    // the data path. Both exist separately because the rebuild's clear() runs in between and is
    // exactly the operation that has to be bounded.
    bool reconnect(std::string & err);

    // Back to the data-path read timeout. Returns false when it could not be set, which the
    // caller treats as a failed rebuild: a connection with no failure bound is not usable.
    bool restore_data_timeout();

    bool connected() const { return qlh_split::net::valid(sock); }

    // [QLH-SPLIT] v8: can this stage draft with its own MTP head? True when its OP_HELLO reported
    // has_nextn, i.e. it owns a NextN block and built a draft context over it.
    //
    // Read from the identity confirmed at attach time, not from a live probe: whether a stage can
    // draft is a property of the process that answered the handshake, and a stage restarted without
    // the head answers ST_UNSUPPORTED later, which is handled as "no drafts" by mtp_draft(). This
    // accessor only has to answer "is asking worth trying", never "will it work".
    bool mtp_ready() const { return have_confirmed && confirmed.has_nextn != 0; }

    // The endpoint this client was asked to connect to. Kept for error messages: when a stage
    // fails, the operator has to know which one, and the endpoint is the only name a stage has
    // on this side of the wire.
    const std::string & endpoint() const { return ep; }

    // Runs one decode. Input is n_tokens rows of n_embd floats; `positions` holds n_tokens *
    // n_pos entries, section-major (the M-RoPE sections of every token, exactly as the head's
    // graph used them -- see the note on llama_batch_allocr in llama-split-proto.h).
    // On success either `out_hidden` (n_tokens * n_embd), `out_logits` (n_vocab) or
    // `out_cands` (the top-k candidates) is filled, depending on this stage's role and on
    // `cand` -- the caller asks, the stage's role disposes.
    // Returns 0 on success, 1 when the stage is out of KV room, negative on error.
    //
    // [QLH-SPLIT] v5 candidate channel. `cand` (nullptr = send no candidate request) asks the
    // tail to run the steps listed in it and answer with cand_req::top_k (id, logit) pairs
    // instead of the whole row. `cand_bias` and `cand_hist` carry the logit biases and the recent
    // token window those steps need; both are only read when `cand` is set.
    //
    // This is a wire-level choice and nothing more: the caller is responsible for having checked
    // that the k candidates are a faithful substitute for the row (grammar, pre-sampling
    // logprobs, top_k <= 0 -- see the fallback rule in common/sampling.h). Nothing here inspects
    // the sampler, because this layer must stay usable by a caller that has no common/ at all.
    //
    // [QLH-SPLIT] v7 per-row verify channel. `verify_rows` asks the tail for one token id per
    // batch row instead of one row of logits for the last one, and `out_verify` receives those
    // ids in row order (n_tokens entries, 4 B each). The two channels answer different questions
    // and are mutually exclusive; `cand != nullptr && verify_rows` is refused by the stage.
    //
    // Like the candidate channel this is only a wire choice: the caller must have established
    // that every row it will read is a greedy argmax (common_sampler_verify_rows_check), because
    // a token id cannot be re-sampled from -- there is no distribution in it.
    // [QLH-SPLIT] v8 nextn channel. `nextn_rows` asks the tail to append the h_nextn rows of the
    // rows it answers for (all rows of a verify frame, the last one otherwise) to whichever reply
    // it gives; `out_nextn` receives them, row-major, and `out_nextn_embd` their width (0 when the
    // stage did not answer them). A tail with no NextN block simply ignores the flag, so this call
    // reports "no rows" rather than an error: the frame itself was fine, the capability is missing,
    // and the caller drops one request's drafts.
    //
    // [QLH-SPLIT] v9 layer-input tap channel. `tap.ids` asks for the input rows of those source-model
    // layers; every stage the frame is sent to answers the ones inside its own range and `out_tap`
    // receives them ([n_tokens][n_answered][n_embd] plus the answered ids). A stage that owns none
    // ignores the request, so `out_tap` comes back empty and that is not an error either -- the caller
    // is the one that knows whether a missing tap matters, and it answers "draft without the
    // injection" instead of failing the request.
    int decode(const std::vector<float> & hidden,
               const std::vector<int32_t> & positions,
               uint32_t n_pos,
               uint32_t n_embd,
               int32_t seq_id,
               const qlh_split::cand_req * cand,
               const std::vector<llama_logit_bias> & cand_bias,
               const std::vector<llama_token> & cand_hist,
               bool verify_rows,
               bool nextn_rows,
               const llama_split_tap_req & tap,
               std::vector<float> & out_hidden,
               std::vector<float> & out_logits,
               uint32_t & out_n_vocab,
               std::vector<qlh_split::token_cand> & out_cands,
               std::vector<int32_t> & out_verify,
               std::vector<float> & out_nextn,
               uint32_t & out_nextn_embd,
               llama_split_tap_out & out_tap,
               std::string & err);

    // [QLH-SPLIT] v8: run this stage's MTP head for one draft round.
    //
    // `h` is one hidden row (n_embd floats): t_h_nextn of position pos-1, as the tail itself sent
    // it back on the decode that produced `token`. `token` sits at `pos`; the k drafts are for
    // pos+1 .. pos+k. Returns the number of drafts written to `out` (0 for "this stage cannot
    // draft"), or -1 on a transport error, which poisons the chain exactly like a failed decode.
    //
    // The k steps are recursive inside the stage, so this is one round trip by construction. The
    // drafts are greedy: the tail answers the argmax of each step, which is what the local chain's
    // draft sampler produces (top_k = 10, then cur_p->data[0]) and what the greedy verify channel
    // can accept.
    int mtp_draft(const std::vector<float> & h, int32_t token, int32_t pos, uint32_t k,
                  int32_t seq_id, std::vector<int32_t> & out, std::string & err);

    // Drops [p0, p1) of a sequence. Returns true only when the stage actually did it.
    bool seq_rm(int32_t seq_id, int32_t p0, int32_t p1, std::string & err);

    bool clear(std::string & err);

    // ---- sequence state (llama_state_seq_*) ----
    //
    // The stage holds the same kind of memory the head does, so the only thing a checkpoint
    // cannot get from the wire is the size of its blob -- and that is what state_size() is for.
    // All three report the stage's seq_pos_max alongside, because a chain's stages are
    // position-synchronized by construction and a mismatch is the only cheap way to notice that
    // one of them is no longer the one the checkpoint was taken from.

    // Size of the stage's state in bytes, without transferring it.
    bool state_size(int32_t seq_id, uint32_t flags, uint32_t & n_bytes, int32_t & pos_max, std::string & err);

    // [QLH-SPLIT] v6: asks the stage to keep a snapshot of its sequence state and to answer with
    // a handle for it. 8 bytes cross the wire in each direction instead of the blob.
    bool state_take(int32_t seq_id, uint32_t flags, uint64_t & handle, int32_t & pos_max, std::string & err);

    // Restores the snapshot `handle` names. Returns false only for a real failure. `stale` is set
    // (with a true return) when the stage no longer holds that snapshot: nothing was applied and
    // the chain is untouched, so the caller must treat it as "no checkpoint", never as a dead
    // chain. `pos_max_before` reports the stage's position as it was before the attempt.
    bool state_give(int32_t seq_id, uint32_t flags, uint64_t handle, int32_t pos_expected,
                    int32_t & pos_max_before, bool & stale, std::string & err);

    // The pre-v6 inline path, kept only for the state blob a slot file written by an older build
    // still carries ({... 12.56 MiB of state ...} instead of a handle). Nothing in a v6 process
    // produces it; reading it back is what keeps an old --slot-save-path file from being parsed
    // as a handle.
    bool state_load(int32_t seq_id, uint32_t flags, const std::vector<uint8_t> & blob,
                    int32_t pos_expected, int32_t & pos_max_before, std::string & err);

    void close();

    // [QLH-SPLIT] Same, with the BYE optional. A rebuild closes a connection whose peer already
    // stopped answering, so it must not spend a write timeout saying goodbye to it.
    void close(bool send_bye);

private:
    qlh_split::net::sock_t sock = qlh_split::net::invalid;

    // Set by connect(), kept after a failed re-connect so an error message can still name the
    // stage the operator asked for.
    std::string ep;

    // [QLH-SPLIT] The identity this stage answered during a successful hello(). reconnect()
    // demands this exact identity back: the chain is built around one specific stage per
    // endpoint, and a restart that changes any of these numbers is a different stage, even if
    // it would be valid somewhere else.
    qlh_split::stage_id confirmed{};
    bool                have_confirmed = false;

    // Whether the peer has completed a HELLO on the current socket. Only a greeted connection
    // gets the polite BYE on close: a connection that never finished the handshake is not in a
    // state where the stage is waiting for it, and sending one anyway just produces a protocol
    // error in the stage's log during a failed attach.
    bool greeted = false;

    // The one connect path: `connect_ms` 0 means an ordinary blocking connect, `io_ms` is the
    // send/recv bound installed on the resulting socket.
    bool connect_endpoint(const std::string & endpoint, std::string & err, int connect_ms, int io_ms);

    bool send_all(const void * src, size_t n);
    bool recv_all(void * dst, size_t n);
};
