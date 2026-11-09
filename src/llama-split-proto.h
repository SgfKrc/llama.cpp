// [QLH-SPLIT] Wire protocol between stages of a split chain.
//
// The chain already has a clean seam: a stage takes hidden rows in and gives hidden rows (or
// logits) out. This header is that seam on the wire -- nothing more.
//
// Framing is fixed-width and little-endian on both sides, with no negotiation: the two ends
// are built from the same source tree, so version skew is not a supported state. A request
// carries exactly one operation; a reply carries exactly its result. There is no pipelining and
// no multiplexing, because decode_split drives the stages in order anyway.
//
// Every field is explicit about width rather than using `unsigned long` or `size_t`, so a
// frame means the same thing on both ends of a connection.
//
// The ops are exactly the memory operations llama-server actually reaches on a chain (the
// measurement that decided this is in llama-memory-split.h); there is nothing here for
// seq_cp / seq_add / seq_keep / seq_div, which a serving path never calls.
#pragma once

#include <cstdint>

namespace qlh_split {

constexpr uint32_t MAGIC       = 0x514C4853; // 'QLHS'
// Bumped whenever a frame layout or an op number changes. There is no negotiation, so this is
// the only thing that turns "a stage from an older build" into a clean protocol error instead of
// a misread reply header. v2: dropped the unused OP_POS and rsp_header::pos_max.
// v3: added the state ops (llama_state_seq_*) and the n_bytes / pos_max header fields they use.
// v4: added OP_HELLO, the identity handshake every connection now starts with, and made a HELLO
//     the *first* frame a stage accepts. This has to be a version bump and not just a new op:
//     a v3 stage would answer a v4 HELLO with "bad request" and a v3 client would never send one,
//     so without the bump both sides would identify themselves only in the logs. With it, each
//     end refuses the other on the first frame -- which is the correct outcome, because the
//     identity check only means anything if the two ends were built together.
// v5: added the candidate channel. A decode may now ask the tail for its top-k candidates
//     (DECODE_FLAG_CANDIDATES) instead of the whole row of logits, and the reply says which of
//     the two it carries (rsp_header::flags). This is a version bump rather than a flag-only
//     change because a v4 tail would ignore the flag, read the extra request bytes as the next
//     header, and desynchronize the connection -- the worst kind of skew, since it looks like a
//     network fault.
// v6: the state ops carry a *handle* instead of the blob (STATE_FLAG_HANDLE), and a stage may
//     answer ST_STALE for a handle it no longer holds. Before this, every context checkpoint
//     pulled the stage's whole recurrent state down the wire and pushed it back on restore:
//     6,733,996 B for an 8-layer qwen35 tail, 13,172,908 B for the 8-layer qwen35moe tail, per
//     checkpoint, and twice per request on the measured chat path. The stage already holds that
//     state in its own memory, so the head only needs a *name* for it. A v5 stage would read the
//     handle record as the first bytes of a blob and desynchronize the connection, which is why
//     this is a version bump and not a flag-only change.
// v7: the per-row verify channel (DECODE_FLAG_VERIFY_ROWS). A decode may ask the tail for one
//     *token id* per batch row instead of one row of logits for the last row, which is what
//     speculative verification needs: k drafts + 1 target row, each reduced to its argmax by the
//     stage that already has the row in memory. The reply carries n_tokens * int32 (16 B for the
//     default k = 3) instead of 993 KB of logits. Flag-only on both sides -- the frame layouts do
//     not move -- but a v6 tail would *ignore* the flag and answer with the single logits row,
//     and the head would then parse 993 KB as a token array. Refusing on the version is the same
//     argument as v5: a silently misparsed reply is worse than a clean protocol error.
// v8: the MTP draft channel (OP_MTP_DRAFT) and the h_nextn rows a decode can answer with
//     (DECODE_FLAG_NEXTN_ROWS). The draft head lives in the tail segment -- it is a full
//     transformer block appended after the trunk, so only the range that owns the model's last
//     layer loads it -- and until now a remote chain could only draft with ngram, because the
//     only way to run that head was a local context. v8 lets the head hand the tail one hidden
//     row plus one token and get k draft tokens back, with the k-step recursion happening inside
//     the stage (each step's input h is the previous step's output, so it cannot be batched).
//     Two layout changes force the bump: OP_MTP_DRAFT is a new frame, and a decode reply that
//     carries h_nextn rows appends n_rows * n_embd floats that a v7 head would read as the next
//     reply's header. stage_id also gains has_nextn in a byte that was padding, which is why a
//     v8 stage and a v7 head must refuse each other rather than trade a 28-byte struct with a
//     different meaning.
// v9: the layer-input tap channel (DECODE_FLAG_LAYER_INP). DFlash and DSpark draft by injecting the
//     *input* row of a set of trunk layers (dflash.target_layers) into the draft model's KV cache.
//     In a split chain the upper taps are computed by the tail, so the head cannot read them at all
//     -- which is what kept both drafters local-only, while the measured loss of the alternative
//     (MTP over a chain: -24.4% acceptance) says they are worth having remotely. A decode can now
//     name *source-model* layer ids and get back the rows of the ids each stage owns.
//     Layout changes: the request grows a layer_inp_req block when the flag is set, a reply that
//     carried taps appends n_taps ids plus n_rows * n_taps * n_embd floats, and rsp_header names the
//     answered tap count in a byte that was padding. A v8 head would read the ids as the next
//     reply's header, so this is a version bump and not a flag-only change.
constexpr uint32_t PROTO_VER   = 9;
constexpr uint32_t MAX_TOKENS  = 65536;      // per frame; guards against a corrupt length
constexpr uint32_t MAX_EMBD    = 65536;      // widest hidden row we accept
constexpr uint32_t MAX_STATE   = 1u << 30;   // largest sequence state we accept (1 GiB)

// [QLH-SPLIT] v7: the most result rows a tail answers in one per-row verify frame.
//
// This is a *shared constant*, not a stage command-line option, because the head has to know it
// before it asks: a tail is built with n_outputs_max rows of logits and a frame that asks for
// more is refused. Both ends come from the same source tree (that is what PROTO_VER means), so
// one number here is enough -- the same argument this file already makes for every other
// capacity it pins.
//
// A speculative round verifies n_draft + 1 rows (the sampled token plus every draft). The default
// draft block is 3 (common_params_speculative_draft::n_max), and the server caps a remote chain's
// draft block at MAX_VERIFY_ROWS - 1 (tools/server/server-context.cpp, get_n_draft_max), so a
// request that asks for more is shortened, never silently truncated on the wire.
//
// The cost is on the tail's side only: n_outputs_max bounds the logits node the *reserve* pass
// allocates, so the stage holds MAX_VERIFY_ROWS * n_vocab * 4 B of compute buffer -- 16.9 MB for
// the 248k-vocabulary model this fork ships, against 3.3 GB at -b 2048. Paying that once at
// startup is what makes a verify frame possible at all; paying it per frame would be the bug.
constexpr uint32_t MAX_VERIFY_ROWS = 17;

// [QLH-SPLIT] v8: the most draft tokens one OP_MTP_DRAFT may ask for.
//
// Not a tuning knob: it is MAX_VERIFY_ROWS - 1, i.e. the number of drafts a verify frame can
// carry. A round that asked for more could not be verified, so the cap is the same number the
// draft block is capped by (tools/server/server-context.cpp, get_n_draft_max) and both ends read
// it from here rather than inventing a second limit.
constexpr uint32_t MAX_MTP_DRAFT = MAX_VERIFY_ROWS - 1;

// [QLH-SPLIT] v9: the most layer-input taps one decode frame may ask for.
//
// A property of the draft model, not a tuning knob: DFlash names 8 trunk layers and DSpark 5, and a
// frame that asked for more would be answered with more rows per batch row than any draft consumes.
// The cap exists so a corrupt or hostile n_taps cannot turn into a huge allocation in a stage; the
// real value is attached to the model (dflash.target_layers), and both ends read the cap from here.
constexpr uint32_t MAX_LAYER_INP_TAPS = 16;

enum op : uint8_t {
    OP_DECODE     = 1,   // hidden in, hidden or logits out; the data path of llama_decode
    OP_SEQ_RM     = 2,   // llama_memory_i::seq_rm -- the only mutation llama-server performs per request
    OP_CLEAR      = 3,   // llama_memory_i::clear -- warmup and the server's seq_rm capability probe
    // [QLH-SPLIT] v6: the state ops. With STATE_FLAG_HANDLE (see below) OP_STATE_SAVE asks the
    // stage to keep the snapshot and answer with its handle, and OP_STATE_LOAD names one instead
    // of carrying the bytes; without it they still move the blob itself, which is what a slot
    // file written by a pre-v6 build contains.
    OP_STATE_SAVE = 4,   // llama_state_seq_get_size / get_data: fetch (or just measure) a state blob
    OP_STATE_LOAD = 5,   // llama_state_seq_set_data: push a state blob back
    OP_BYE        = 6,   // clean shutdown, so a closing chain does not look like a broken pipe
    // [QLH-SPLIT] v4: identity handshake. The first frame on every connection, request and reply.
    // See stage_id below for what it carries and why these fields and not a model hash.
    OP_HELLO      = 7,
    // [QLH-SPLIT] v8: draft k tokens with the MTP head this stage owns. See mtp_req for the
    // request body and the note there for why one frame can carry a whole draft block.
    OP_MTP_DRAFT  = 8,
};

// Which end of a chain a stage serves. It is part of the identity, not a tuning flag: the stage
// decides from its own role whether a decode answers with hidden rows or logits, so a tail in a
// relay position (or a relay in the tail position) hands the next stage a vector it cannot use.
// reconnecting does not change the role, so it is compared like every other field.
enum role : uint8_t {
    ROLE_ANY   = 0, // in a HELLO request: this position does not pin the role (never sent as a stage's own role)
    ROLE_RELAY = 1,
    ROLE_TAIL  = 2,
};

// Sentinel for "this field is not pinned" in a HELLO *request*. Only the layer_hi of a stage that
// is not the chain's last one is unknown before that stage answers: it is whatever the next stage
// starts at, so this side learns it from this stage's own reply. Every other field is known from
// the head's model, and a reconnect() pins all of them (it compares against the identity it
// confirmed at attach time).
constexpr int32_t ID_ANY_I32 = -1; // no real layer / layer count is negative

// [QLH-SPLIT] Identity of one stage: what a HELLO request asks for, what a HELLO reply reports.
//
// Why a handshake exists at all: reconnecting a stage that was restarted is now routine (the
// chain rebuilds itself on the next request from position 0), and a plain TCP reconnect says
// nothing about *which* stage answered. A stage restarted with a different --layer-range, a
// different model, or a different trunk depth produces plausible-but-wrong tokens from the first
// frame, with no error anywhere -- the exact class of failure this layer exists to make
// impossible.
//
// Why exactly these fields, and not a hash of the model file:
//  - layer_lo / layer_hi: the range this stage computes. Two stages with different ranges feed
//    the same hidden row through different layers, so every later token is wrong while the wire
//    stays perfectly well formed.
//  - n_layer_src: the trunk depth of the source model, i.e. what those layer numbers are
//    relative to. [16, 24) of a 24-layer model and [16, 24) of a 32-layer model are different
//    stages, and layer_lo/layer_hi alone cannot tell them apart.
//  - n_embd_inp / n_embd_out: the width of the hidden row the stage consumes and produces.
//    A hop carries exactly one width and the chain already validates it, but only after the
//    stage has been handed a decode; checking it at connect time is what makes a wrong model
//    fail at attach instead of mid-request. The two are separate fields because a stage that
//    could not carry one width is refused at startup -- seeing the disagreement here is how the
//    head notices it is not talking to a stage built from the same model.
//  - n_vocab: only the tail uses it (it is the logits dimension), but every stage carries it.
//    It is one more number a different model cannot accidentally share, and it makes "same
//    architecture, same width, different model" visible where the other fields agree.
//  - role: relay vs tail, as above.
//
// [QLH-SPLIT] v8 adds one field, and it is the first thing in this protocol that answers "what can
// you *do*" instead of "who are you": has_nextn says whether this stage holds a working MTP draft
// context. Unlike every other field it is deliberately *not* compared, because the chain does not
// require it to match -- a tail whose MTP context failed to build is still the right tail, and the
// only consequence is that the head does not speculate (see OP_MTP_DRAFT). It rides in a byte that
// was padding, so the struct keeps its size; that is not a compatibility argument (PROTO_VER
// covers that) but it does keep the frame layout the same shape it has always had.
//
// A full model hash is deliberately not used: it would read hundreds of MB on every reconnect
// (the chain rebuilds on the request path) to answer "is this the same file", while a stage
// restarted with a different --layer-range of the *same* file is already a different stage. These
// fields are the numbers the computation actually depends on; anything else is a proxy that
// costs much more and discriminates less.
//
// The numbers are also not a security boundary: the protocol has no authentication, and --host
// exists for a stage reached over an internal network or a tunnel. This identifies a stage, it
// does not authorize one.
struct stage_id {
    int32_t  layer_lo;
    int32_t  layer_hi;
    int32_t  n_layer_src;
    uint32_t n_embd_inp;
    uint32_t n_embd_out;
    uint32_t n_vocab;
    uint8_t  role;
    // [QLH-SPLIT] v8: non-zero when this stage can answer OP_MTP_DRAFT, i.e. it owns a NextN/MTP
    // block and built a draft context over it. 0 for a relay, for a tail range that does not end at
    // the model's last layer, and for a model without NextN layers.
    uint8_t  has_nextn;
    uint8_t  _pad[2];
};

static_assert(sizeof(stage_id) == 28, "stage_id is a wire structure; keep its size explicit");

// req_header::flags for the state ops: the low bits are llama_state_seq_flags (PARTIAL_ONLY = 1,
// ON_DEVICE = 2), the top bit asks OP_STATE_SAVE for the size and the position only. One op for
// both because the size query and the fetch are always issued back to back, and the reply header
// already carries what the query needs.
constexpr uint8_t STATE_FLAG_SIZE_ONLY = 0x80;

// [QLH-SPLIT] v6, req_header::flags for the state ops: "keep this snapshot yourself and tell me
// its name" / "restore the snapshot this name refers to".
//
// Why a handle at all: a checkpoint of a split chain is the head's state plus every stage's, and
// the stage's part is the recurrent state -- 6.7 MiB for an 8-layer qwen35 tail, 12.56 MiB for
// the same 8 layers of the 35B qwen35moe. The head never reads those bytes: it stores them and
// hands them back unchanged. Measured on a real link, that round trip was 0.8-1.2 s per
// checkpoint at 16 MB/s and 2.8 s at the 5 MB/s the field report was taken on -- more than the
// decode it was protecting. A stage asked to *keep* the snapshot instead transfers 8 bytes down
// and 8 bytes back, and the restore is a memcpy inside the process that already had the state.
//
// The handle is opaque to the head. It names a snapshot in the stage's own cache, is valid only
// for that stage *process*, and expires when the cache evicts it (ST_STALE, see status). Nothing
// about it is stable across a stage restart, which is exactly why an expired handle must be a
// recoverable answer and not a broken chain: the stage refuses before applying anything, so the
// chain is exactly where it was and the caller can re-process the prompt instead of restoring.
constexpr uint8_t STATE_FLAG_HANDLE = 0x40;

// The size of the record the head stores per remote stage in the v6 layout: {int32 pos_max,
// uint32 n_bytes, uint64 handle}. Named so both ends and the state layout comment agree.
constexpr uint32_t STATE_HANDLE_BYTES = sizeof(uint64_t);

// [QLH-SPLIT] v5, req_header::flags for OP_DECODE: "answer with the top-k candidates, not the
// whole row of logits".
//
// Why the candidates and not the row: the tail's logits row is n_vocab f32 (248 KB for the 2B
// model here, 993 KB over the wire with framing and the rest of the frame), and it is returned
// for every decoded token. On a 16 MB/s link that is ~59 ms per token, which is most of the
// difference between a cross-machine chain and a whole model (build-split/xhost/REPORT.md).
// The sampling chain truncates to k tokens in its `top_k` step anyway, and every step after
// top_k only ever looks at those k entries -- so the k (id, logit) pairs are exactly as good an
// input for the rest of the chain as the full row, at 8 bytes each. See cand_req for what the
// tail needs to reproduce the steps *before* top_k, and llama-split-remote.h for the fallback
// rule that keeps this flag off whenever the k candidates are not a faithful substitute.
//
// Bit 0 is free for OP_DECODE because the only other flags in this header are the state-op ones
// above, and the two ops never share a frame.
constexpr uint8_t DECODE_FLAG_CANDIDATES = 0x01;

// [QLH-SPLIT] v7, req_header::flags for OP_DECODE: "answer with one token id per batch row".
//
// Why a second flag and not the candidate channel with top_k == 1: the two answer different
// questions. The candidate channel replaces the *sampler* for one row (the tail runs the steps up
// to top_k and this side runs the rest), so it must know the chain's sampling parameters. This
// one replaces the *verification* of k + 1 rows: the head already decided that every row is read
// through a greedy sampler (common_sampler_verify_rows_check), and under argmax the answer for a
// row is just its token id -- no chain, no parameters, 4 bytes.
//
// Mutually exclusive with DECODE_FLAG_CANDIDATES; a frame that sets both is refused.
constexpr uint8_t DECODE_FLAG_VERIFY_ROWS = 0x02;

// [QLH-SPLIT] v8, req_header::flags for OP_DECODE: "also answer the h_nextn rows of the rows you
// marked as output".
//
// Why the tail has to send them at all: the MTP draft's input is t_h_nextn = RMSNorm(last trunk
// layer output), and in a split chain that last layer runs in the tail, so the head cannot compute
// the row it must hand to the next OP_MTP_DRAFT. It is not a speculative extra either: the row of
// the *accepted* position is the input of the next draft round, and the row of the last token of a
// plain decode is the input of the round after that.
//
// Which rows: exactly the ones the reply already answers for -- all of them for a verify frame,
// the last one for a plain tail decode (see batch.logits in tools/split-stage/main.cpp). That is
// what keeps a prefill from shipping n_tokens * n_embd floats: a prefill frame marks one output
// row, so it carries one h row (16 KB at n_embd 4096), not the whole prompt's worth.
//
// Eligible in the same frame as either output channel: the draft needs this row *while* it reads
// per-row tokens or candidates, so forbidding the combination would mean asking the tail the same
// question twice.
//
// A stage that cannot produce the rows ignores the flag rather than refusing the frame: the reply
// then has the v7 shape, which the head still parses (it reads the rows because of the *reply*
// flag, never because it asked), so a tail that lost its MTP context degrades to "no drafts"
// instead of failing the request.
constexpr uint8_t DECODE_FLAG_NEXTN_ROWS = 0x04;

// [QLH-SPLIT] v9, req_header::flags for OP_DECODE: "also answer the layer-input rows of these
// layers".
//
// Why a stage has to send them: DFlash and DSpark draft by injecting the *input row* of a set of
// trunk layers (dflash.target_layers -- 8 for DFlash, 5 for DSpark) into the draft model's KV cache,
// one row per batch position. That row is the hidden state entering layer il, so layer il has to run
// where it is read; in a split chain the upper taps run in the tail, and the head cannot reconstruct
// them from what the chain hands back (that is the last layer's output, not any layer's input). The
// local chain reads them straight out of the tail context's buffer (common/speculative.cpp,
// tap_ctx/tap_lid, zero bytes on the wire); a remote chain needs them here.
//
// Which ids: *source-model* layer ids, not stage-local ones. Every stage that receives the request
// answers the ids inside its own [layer_lo, layer_hi) and skips the rest, so the head needs to know
// nothing about where the chain was cut -- and a three-stage chain, where a middle relay owns some of
// the taps, works without any extra mapping. The answered ids travel back in the reply, which is what
// lets a stage answer a subset without either side guessing.
//
// Which rows: all of them, always. That is where this channel differs from the h_nextn one above: the
// draft injects *every* position of the frame into its KV, so a verify frame needs k + 1 rows and a
// prefill frame needs the whole ubatch. A prefill is therefore where the channel actually costs
// something (n_ubatch * n_taps * n_embd), not the k-row verify frame.
//
// Orthogonal to all three output channels: the rows are appended last (see RSP_FLAG_LAYER_INP).
constexpr uint8_t DECODE_FLAG_LAYER_INP = 0x08;

// [QLH-SPLIT] v5, rsp_header::flags: the reply to an OP_DECODE that carried
// DECODE_FLAG_CANDIDATES. The body is n_bytes worth of (int32 id, float logit) pairs, sorted by
// descending logit -- the same order and the same logit values llama_sampler_apply() leaves in
// cur_p after its top_k step. n_vocab is 0 in this reply: the whole row was *not* sent, so a
// caller that needs it (pre-sampling logprobs, grammar) must have asked for it up front. That
// decision belongs to the driving side and is made before the request starts, never inferred
// from a reply (see llama-split-remote.h).
constexpr uint8_t RSP_FLAG_CANDIDATES = 0x01;

// [QLH-SPLIT] v7, rsp_header::flags: the reply to an OP_DECODE that carried
// DECODE_FLAG_VERIFY_ROWS. n_tokens is the row count of the frame that was decoded, n_embd and
// n_vocab are 0, and the body is n_bytes worth of int32 token ids in batch row order --
// row i's entry is the argmax of row i's logits, i.e. exactly the token a greedy chain
// (llama_sampler_init_greedy, or temp 0) would have selected from that row.
//
// Ties resolve to the lowest token id, which is what the greedy sampler does as well
// (src/llama-sampler.cpp, llama_sampler_greedy_apply: only a strictly greater logit moves the
// selection).
constexpr uint8_t RSP_FLAG_VERIFY_ROWS = 0x02;

// [QLH-SPLIT] v8, rsp_header::flags: the reply to an OP_DECODE that carried
// DECODE_FLAG_NEXTN_ROWS. n_embd is the width of those rows (not 0, unlike the other two channel
// replies), and the body appends n_rows * n_embd f32 to whatever the reply already carries, in
// this order: the per-row token ids (if RSP_FLAG_VERIFY_ROWS), then the h_nextn rows, then the
// candidate pairs (if RSP_FLAG_CANDIDATES) or the logits row. n_rows is n_tokens for a verify
// frame and 1 otherwise -- the stage decides from the batch it built, and the head knows which of
// the two it asked for, so the count is never inferred from the byte length.
//
// They are raw f32, the same representation the local chain reads out of its tail context: this is
// the MTP head's input, and quantizing or truncating it would change the draft tokens, which is
// exactly what the draft/verify comparison in verify/verify_mtp.sh is there to catch.
constexpr uint8_t RSP_FLAG_NEXTN_ROWS = 0x04;

// [QLH-SPLIT] v9, rsp_header::flags: the reply to an OP_DECODE that carried DECODE_FLAG_LAYER_INP and
// that this stage owns at least one of the named layers to answer for.
//
// rsp_header::n_taps is the number of ids answered, n_embd their width, and the body ends with those
// n_taps int32 source-layer ids followed by n_rows * n_taps * n_embd f32: the first tap's row for
// batch row 0, the second tap's row for row 0, ..., then row 1. n_rows is the frame's own row count,
// which the head knows because it sent the frame; a stage that answered another count is caught by
// n_bytes, so no count is ever inferred from a byte length.
//
// The tap section is the *last* section of every reply, after the per-row ids, the h_nextn rows and
// the candidates or logits row. That keeps the head's parse of the answer it asked for identical to
// v8 and puts the taps behind it, one block, once per frame.
//
// n_taps = 0 with the flag clear is the answer of a stage that owns none of the named layers: it
// ignores the request instead of refusing the frame, the same way a stage without a NextN block
// ignores DECODE_FLAG_NEXTN_ROWS. A missing tap degrades one request's drafts; refusing the frame
// would fail it, and a draft is an optimization (llama-ext.h).
constexpr uint8_t RSP_FLAG_LAYER_INP = 0x08;

// [QLH-SPLIT] v8: the body of OP_MTP_DRAFT.
//
// One frame carries a whole draft block, and that is the point: the k drafts are recursive (draft
// i's input h is draft i-1's output h_nextn), so they cannot be batched, but the *round trip* can
// be one. The tail runs the k internal decodes and answers k token ids; a per-token op would put
// k extra round trips on the request path, which on the measured 18-22 ms hop is the whole win.
//
// The two scalars are the local chain's draft loop, unchanged (common/speculative.cpp,
// common_speculative_impl_draft_mtp::draft, the non-chain_heads branch): step i decodes the token
// at position pos + i, with the h row the previous step produced, and step 0 uses `hidden` -- the
// h_nextn row of position pos - 1 -- together with `token`, the token at `pos`.
//
// Drafting is greedy here on purpose. The local implementation's draft sampler is top_k = 10 and it
// takes cur_p->data[0], i.e. the argmax, so the tail answering the argmax of each of its own rows
// produces the same tokens; and the verify channel this feeds is only armed for a greedy chain in
// the first place (common_sampler_verify_rows_check), where argmax *is* the answer. A probabilistic
// drafter would need q over the full vocabulary to cross the wire, which is the cost this whole
// layer exists to avoid.
struct mtp_req {
    int32_t  token;  // the token at `pos` (the last accepted or sampled one)
    int32_t  pos;    // its position; draft i is decoded at pos + i
    uint32_t k;      // drafts to return, 1..MAX_MTP_DRAFT
    uint32_t _pad;
};

static_assert(sizeof(mtp_req) == 16, "mtp_req is a wire structure; keep its size explicit");

// [QLH-SPLIT] v9: the body section a DECODE_FLAG_LAYER_INP frame appends after everything else in the
// frame -- this fixed part, then n_taps int32 source-layer ids.
//
// Variable length rather than a fixed array: the tap list belongs to the draft model, not to this
// protocol, so a model with more taps must not need a protocol change, and a fixed array sized for
// the largest one would put its width in every frame for every chain.
//
// n_embd is here so a stage can refuse a frame from a different model up front, with a number in the
// message, instead of shipping a block whose stride nobody agrees on. It is the *trunk* width, not the
// hidden-row width the header carries: those are equal in every split chain this tree builds
// (llama_split_attach refuses a chain where they differ), but the tap row is the layer input, which
// is always the trunk width (src/llama-context.cpp, extract_layer_inputs).
struct layer_inp_req {
    uint32_t n_taps; // ids that follow; 1..MAX_LAYER_INP_TAPS
    uint32_t n_embd; // width of one tap row
    uint32_t _pad[2];
};

static_assert(sizeof(layer_inp_req) == 16, "layer_inp_req is a wire structure; keep its size explicit");

// [QLH-SPLIT] v5: the steps a tail must run before it can answer with candidates, in the order
// they appear in the driving side's sampler chain. Only the samplers that have to see the whole
// vocabulary are in here; everything after top_k runs on the candidates, on the side that has
// the rest of the chain.
//
// The tail cannot take a common_sampler: tools/split-stage links `llama` and not `common`, so it
// has only llama.h's C API. That is enough -- these three are exactly the steps top_k needs to
// be exact.
enum cand_step : uint8_t {
    CAND_STEP_PENALTIES   = 1, // llama_sampler_init_penalties; needs the recent token history
    CAND_STEP_TOP_N_SIGMA = 2, // llama_sampler_init_top_n_sigma; needs the whole row's mean/std
    CAND_STEP_TOP_K       = 3, // llama_sampler_init_top_k; always last
};

// [QLH-SPLIT] v5: what the tail must know to reproduce the steps up to and including top_k.
//
// Penalties are here because they run *before* top_k: a penalty raises or lowers logits across
// the whole vocabulary, so the top-k set of the penalized row is not the top-k set of the raw
// row. Applying them on the candidate set afterwards would be both late (the k set could already
// be wrong) and wrong (a token outside the k set can never come back). logit_bias is the same
// case, which is why it travels too -- see the layout note in req_header.
//
// The history is a window, not an increment. The tail rebuilds the penalty state from scratch on
// every frame (reset, then accept the window oldest-first), so a reconnected tail, a stage that
// was restarted, a seq_rm and a state restore all converge to the same answer without any
// bookkeeping on the wire. An increment would be smaller (4 bytes per token instead of a 64-token
// window per frame) and would silently diverge the first time the two ends disagree about which
// tokens were accepted -- exactly the failure this layer exists to make impossible. 64 tokens is
// 256 bytes per frame against 320 for the candidates themselves; there is nothing to save.
struct cand_req {
    int32_t  top_k;              // candidates to return (the chain's top_k)
    float    top_n_sigma;        // < 0: the chain has no top_n_sigma step to run here
    int32_t  penalty_last_n;     // 0 disables the penalties step
    float    penalty_repeat;
    float    penalty_freq;
    float    penalty_present;
    uint32_t n_bias;             // logit biases that follow: n_bias * {int32 id, float bias}
    uint32_t n_hist;             // then n_hist * int32 token, oldest first
    uint8_t  n_steps;            // steps below actually present, in order
    uint8_t  steps[3];           // cand_step values
    uint8_t  _pad[2];
};

static_assert(sizeof(cand_req) == 40, "cand_req is a wire structure; keep its size explicit");

// The candidate pairs themselves. Interleaved rather than two arrays: they are produced and
// consumed together, and one array is one recv.
struct token_cand {
    int32_t id;
    float   logit;
};

static_assert(sizeof(token_cand) == 8, "token_cand is a wire structure; keep its size explicit");

enum status : uint8_t {
    ST_OK       = 0,
    ST_NO_SPACE = 1,  // maps to llama_decode's "1 = retry with a smaller batch"
    ST_BAD_REQ  = 2,  // maps to -1
    ST_INTERNAL = 3,  // maps to -3
    // [QLH-SPLIT] v6: the handle names a snapshot this stage no longer holds -- its snapshot
    // cache evicted it, or this process is not the one that minted it. Nothing was applied, so
    // the chain is exactly where it was before the restore: the head answers "no checkpoint"
    // (re-process the prompt) instead of declaring the chain dead. See STATE_FLAG_HANDLE.
    ST_STALE    = 4,
    // [QLH-SPLIT] v8: this stage cannot draft. Either it never had a NextN/MTP block, its MTP
    // context failed to build at startup, or the range it serves does not end at the model's last
    // layer. Nothing was decoded and nothing moved, so the head answers "this request does not
    // speculate" and keeps serving: a chain whose fast path is unavailable is still a correct
    // chain, which is the whole error policy of this layer.
    ST_UNSUPPORTED = 5,
};

// A request is: header, then (n_tokens * n_pos * 4 bytes of positions), then
// (n_tokens * n_embd * 4 bytes of hidden), for OP_DECODE; for OP_STATE_LOAD the header is
// followed by n_bytes of state blob, and for the other ops it is the whole frame.
//
// [QLH-SPLIT] v5: an OP_DECODE that sets DECODE_FLAG_CANDIDATES follows the hidden rows with one
// cand_req, then n_bias * {int32 id, float bias}, then n_hist * int32 token. Every section is
// self-describing, so a tail reads exactly what the previous one told it to; a v4 tail rejects
// the frame on the version check instead of parsing the candidate block as the next header.
//
// [QLH-SPLIT] v7: an OP_DECODE that sets DECODE_FLAG_VERIFY_ROWS adds nothing to the frame. The
// rows are already described by n_tokens / n_embd / n_pos, and the rows the answer must cover are
// all of them -- that is the whole point of the flag, so there is nothing to negotiate.
//
// [QLH-SPLIT] v8: an OP_DECODE that sets DECODE_FLAG_NEXTN_ROWS also adds nothing to the frame --
// the rows it asks about are the ones the frame already describes. Only the reply grows, by
// n_rows * n_embd floats (see RSP_FLAG_NEXTN_ROWS).
//
// [QLH-SPLIT] v9: an OP_DECODE that sets DECODE_FLAG_LAYER_INP appends one layer_inp_req and then its
// n_taps int32 layer ids, after the candidate block when both are present. The ids are source-model
// layer numbers; a stage answers the ones inside its own range and skips the rest, so one frame can be
// sent to every stage of a chain without this side knowing where the cuts are.
//
// OP_MTP_DRAFT is header, then one mtp_req, then n_embd floats (one hidden row). The reply is the
// header and k int32 token ids, n_bytes = k * 4. n_embd is the width of the hidden row, n_tokens
// is 1, n_pos is 1 and seq_id is the sequence the draft belongs to; the draft context inside the
// stage is per sequence, so a frame that named a different sequence would draft over another
// request's KV.
//
// OP_HELLO is the one exception to "the header is the whole frame": it is followed by exactly
// one stage_id (the caller's expectations), and its reply is followed by the stage's own
// stage_id. Those two payloads are the reason this op exists; putting them here rather than in
// every header keeps the per-frame cost and the existing frame layouts untouched -- identity is
// established once per connection, not certified per decode.
// Positions are section-major -- pos[j * n_tokens + i] is M-RoPE section j of token i -- which
// is the layout llama_batch uses, and n_pos is on the wire so that a stage built for a different
// model rejects the frame instead of reading it at the wrong stride.
// What comes back is decided by the stage's role (relay or tail) and, for a tail, by
// DECODE_FLAG_CANDIDATES / DECODE_FLAG_VERIFY_ROWS: a relay answers hidden, a tail answers either
// the whole logits row, n_bytes/8 candidates (see RSP_FLAG_CANDIDATES) or n_tokens int32 token
// ids (see RSP_FLAG_VERIFY_ROWS). DECODE_FLAG_NEXTN_ROWS is orthogonal to all three: it appends
// the h_nextn rows, in the order documented on RSP_FLAG_NEXTN_ROWS. DECODE_FLAG_LAYER_INP is
// orthogonal to all four, including a relay's hidden rows: the tap ids and rows are the last two
// sections of the reply (see RSP_FLAG_LAYER_INP).
struct req_header {
    uint32_t magic;
    uint32_t version;
    uint8_t  op;
    uint8_t  flags;
    uint16_t _pad;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_pos;     // positions per token (llama_hparams::n_pos_per_embd)
    int32_t  seq_id;
    int32_t  pos0;
    int32_t  pos1;
    uint32_t n_bytes;   // OP_STATE_LOAD: state bytes that follow; OP_HELLO: one stage_id follows
};

struct rsp_header {
    uint32_t magic;
    uint32_t version;
    uint8_t  status;
    uint8_t  flags;     // [QLH-SPLIT] v5: RSP_FLAG_*; was padding before, so the size is unchanged
    // [QLH-SPLIT] v9: layer-input taps this reply answers (RSP_FLAG_LAYER_INP), else 0. It rides in a
    // byte that was padding, which is not a compatibility argument (PROTO_VER covers that) but keeps
    // the frame the shape it has always had -- the same trade stage_id::has_nextn makes.
    uint8_t  n_taps;
    uint8_t  _pad;
    uint32_t n_tokens;
    uint32_t n_embd;    // of the returned rows; 0 when returning logits
    uint32_t n_vocab;   // set when status is OK and the whole logits row follows
    int32_t  pos_max;   // OP_STATE_SAVE/LOAD: the stage's seq_pos_max at the time of the call
    uint32_t n_bytes;   // OP_STATE_SAVE: state bytes that follow; OP_HELLO: sizeof(stage_id);
                        // OP_DECODE with RSP_FLAG_CANDIDATES: n_cand * sizeof(token_cand)
                        // OP_DECODE with RSP_FLAG_VERIFY_ROWS: n_tokens * sizeof(int32_t)
                        // OP_DECODE with RSP_FLAG_LAYER_INP: everything above plus
                        //   n_taps * 4 + n_rows * n_taps * n_embd * 4
};

static_assert(sizeof(rsp_header) == 32, "rsp_header is a wire structure; keep its size explicit");

} // namespace qlh_split
