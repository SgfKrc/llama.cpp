// Wire protocol between stages of a split chain.
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

#include <cassert>
#include <cstdint>

namespace qlh_split {

constexpr uint32_t MAGIC       = 0x514C4853; // 'QLHS'

// PROTO_VER is part of every frame header. There is no negotiation: the two ends are built from
// the same source, and a stage that disagrees is refused on the first frame (OP_HELLO) instead of
// trading a struct with a different meaning.
//
// Any change to a frame layout, to an op number, or to the meaning of a byte that is currently
// padding is a PROTO_VER bump, never a flag-only change: an end that does not know the new shape
// reads the new bytes as the next frame's header and desynchronizes the connection, and a
// silently misparsed reply is worse than a clean protocol error. Every struct below pins its size
// with a static_assert, so a layout change is a build error on both ends.
constexpr uint32_t PROTO_VER   = 10;
constexpr uint32_t MAX_TOKENS  = 65536;      // per frame; guards against a corrupt length
constexpr uint32_t MAX_EMBD    = 65536;      // widest hidden row we accept
constexpr uint32_t MAX_STATE   = 1u << 30;   // largest sequence state we accept (1 GiB)

// The default for `--split-verify-rows`, the most result rows a tail answers in
// one decode frame.
//
// Both ends have to agree on it, and the number is in the identity handshake
// (stage_id::n_verify_rows) and in the stage's own command line, so this is only the value an
// operator gets when they say nothing -- both ends default to it, which is what makes "no flags at
// all" work.
//
// What the number means, and why it is a *bound* rather than an exact count: it sizes the logits
// node the tail's reserve pass allocates (cparams.n_outputs_max), and that buffer cannot grow after
// the context is built. So a frame is allowed to use fewer rows than this -- that is just waste --
// and a frame that asks for more is refused. A multi-sequence chain needs
//
//     n_verify_rows >= n_seq_max * (draft.n_max + 1)
//
// rows (every sequence may be in the middle of a k + 1 row verify block at the same time); the head
// checks that at attach time and refuses with the three numbers in the message.
//
// The cost is on the tail's side only: n_outputs_max * n_vocab * 4 B of compute buffer -- 16.9 MB
// for the 248k-vocabulary model this build targets at the default 17, against 3.3 GB at -b 2048.
// Paying that once at startup is what makes a verify frame possible at all; paying it per frame
// would be the bug.
constexpr uint32_t DEFAULT_SPLIT_VERIFY_ROWS = 17;

// The most layer-input taps one decode frame may ask for.
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
    // The state ops. With STATE_FLAG_HANDLE (see below) OP_STATE_SAVE asks the
    // stage to keep the snapshot and answer with its handle, and OP_STATE_LOAD names one instead
    // of carrying the bytes; without it they still move the blob itself, which is what a slot
    // file written by an older build contains.
    OP_STATE_SAVE = 4,   // llama_state_seq_get_size / get_data: fetch (or just measure) a state blob
    OP_STATE_LOAD = 5,   // llama_state_seq_set_data: push a state blob back
    OP_BYE        = 6,   // clean shutdown, so a closing chain does not look like a broken pipe
    // Identity handshake. The first frame on every connection, request and reply.
    // See stage_id below for what it carries and why these fields and not a model hash.
    OP_HELLO      = 7,
    // Draft k tokens with the MTP head this stage owns. See mtp_req for the
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

// Identity of one stage: what a HELLO request asks for, what a HELLO reply reports.
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
// One field is the first thing in this protocol that answers "what can you *do*" instead of
// "who are you": has_nextn says whether this stage holds a working MTP draft
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
    // non-zero when this stage can answer OP_MTP_DRAFT, i.e. it owns a NextN/MTP
    // block and built a draft context over it. 0 for a relay, for a tail range that does not end at
    // the model's last layer, and for a model without NextN layers.
    uint8_t  has_nextn;
    // The four numbers a multi-sequence chain has to agree on. Each is sized independently on
    // each end (by -np / -c / --rs-seq / --split-verify-rows), because a head with -np N drives N
    // sequences through one frame.
    //
    // Each one is a *hard* failure when the two ends disagree, not a performance difference:
    //  - n_seq_max: the stage's KV pool is cut into n_ctx / n_seq_max per sequence, and its recurrent
    //    cache holds max(1, n_seq_max) cells, one per sequence. Too small and the second sequence has
    //    nowhere to go (caught by check_seq_id, i.e. mid-request); too large and every sequence gets
    //    a smaller slice than the head thinks it has (ST_NO_SPACE mid-request).
    //  - n_ctx_seq: the per-sequence capacity the head computes its positions and its slot policy
    //    against. Two ends with the same n_ctx but different n_seq_max disagree here while their raw
    //    n_ctx still matches, which is exactly why n_ctx itself is NOT compared.
    //  - n_rs_seq: how far one sequence can roll back. Too small and a verify frame's rollback is
    //    refused (the request fails); too large and the stage's rs_idx assertion fires (the process
    //    dies). It is per sequence per layer, and orthogonal to n_seq_max.
    //  - n_verify_rows: the frame's output-row bound, i.e. the size the tail's logits node was built
    //    with. Too small and the tail's output_reserve() aborts -- that buffer cannot grow after the
    //    context exists.
    //
    // n_seq_max is a uint16 because LLAMA_MAX_SEQ is 256; it takes the place of the two padding
    // bytes, so the struct stays 4-byte aligned and only grows by the three 32-bit fields.
    uint16_t n_seq_max;
    uint32_t n_ctx_seq;
    uint32_t n_rs_seq;
    uint32_t n_verify_rows;
};

static_assert(sizeof(stage_id) == 40, "stage_id is a wire structure; keep its size explicit");

// req_header::flags for the state ops: the low bits are llama_state_seq_flags (PARTIAL_ONLY = 1,
// ON_DEVICE = 2), the top bit asks OP_STATE_SAVE for the size and the position only. One op for
// both because the size query and the fetch are always issued back to back, and the reply header
// already carries what the query needs.
constexpr uint8_t STATE_FLAG_SIZE_ONLY = 0x80;

// req_header::flags for the state ops: "keep this snapshot yourself and tell me
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

// The size of the record the head stores per remote stage with STATE_FLAG_HANDLE: {int32 pos_max,
// uint32 n_bytes, uint64 handle}. Named so both ends and the state layout comment agree.
constexpr uint32_t STATE_HANDLE_BYTES = sizeof(uint64_t);

// req_header::flags for OP_DECODE: "answer with the top-k candidates, not the
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
//
// The channel itself is per row -- this bit
// only says "a candidate block follows the `rows` section", and ROW_FLAG_CANDS on a row says which
// rows the block has a request for. One bit for the whole frame would have been enough to describe
// "the frame carries the section", but not *which* rows want it: two sequences have two top_k
// values, two penalty windows and two histories, and one cand_req cannot carry both.
constexpr uint8_t DECODE_FLAG_CANDIDATES = 0x01;

// req_header::flags for OP_DECODE: "answer with one token id per batch row".
//
// Why a second flag and not the candidate channel with top_k == 1: the two answer different
// questions. The candidate channel replaces the *sampler* for one row (the tail runs the steps up
// to top_k and this side runs the rest), so it must know the chain's sampling parameters. This
// one replaces the *verification* of k + 1 rows: the head already decided that every row is read
// through a greedy sampler (common_sampler_verify_rows_check), and under argmax the answer for a
// row is just its token id -- no chain, no parameters, 4 bytes.
//
// Mutually exclusive with DECODE_FLAG_CANDIDATES *on a row*: one row cannot both want a token id
// (argmax) and a candidate array, because the two replace the same answer. Different rows may carry
// different bits, so a frame-level combination is no longer a conflict -- the check moved to the row
// (ROW_FLAG_CANDS | ROW_FLAG_VERIFY on one row is refused by both ends, see ROW_FLAG_VERIFY).
//
// The frame bit is only a summary ("at least one row of this
// frame wants the per-row answer"); the authority is the per-row flag.
constexpr uint8_t DECODE_FLAG_VERIFY_ROWS = 0x02;

// req_header::flags for OP_DECODE: "also answer the h_nextn rows of the rows you
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
// then has the single-row shape, which the head still parses (it reads the rows because of the *reply*
// flag, never because it asked), so a tail that lost its MTP context degrades to "no drafts"
// instead of failing the request.
//
// Which rows want their h row is
// ROW_FLAG_NEXTN now, and the frame bit only says "this frame asks for the channel at all". The
// frame-level rule it used to carry -- "a verify frame answers every row, a plain decode its last
// one" -- could not express a batch where one sequence verifies (k + 1 rows) and another decodes
// (1 row): one bool cannot describe both.
constexpr uint8_t DECODE_FLAG_NEXTN_ROWS = 0x04;

// req_header::flags for OP_DECODE: "also answer the layer-input rows of these
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

// rsp_header::flags: the reply to an OP_DECODE that carried
// DECODE_FLAG_CANDIDATES. The body is n_bytes worth of (int32 id, float logit) pairs, sorted by
// descending logit -- the same order and the same logit values llama_sampler_apply() leaves in
// cur_p after its top_k step. n_vocab is 0 in this reply: the whole row was *not* sent, so a
// caller that needs it (pre-sampling logprobs, grammar) must have asked for it up front. That
// decision belongs to the driving side and is made before the request starts, never inferred
// from a reply (see llama-split-remote.h).
//
// One reply may answer several rows'
// candidate sets, so the section is self-describing: `uint32 n_cand_rows`, then n_cand_rows
// cand_row_ref {int32 row, uint32 n_bytes} ascending by row, then the payloads in that same order.
// n_bytes per row covers that row's pairs only. A single-row reply is that plus a 4-byte header.
constexpr uint8_t RSP_FLAG_CANDIDATES = 0x01;

// rsp_header::flags: the reply to an OP_DECODE that carried
// DECODE_FLAG_VERIFY_ROWS. n_tokens is the row count of the frame that was decoded, n_embd and
// n_vocab are 0, and the body is n_bytes worth of int32 token ids in batch row order --
// row i's entry is the argmax of row i's logits, i.e. exactly the token a greedy chain
// (llama_sampler_init_greedy, or temp 0) would have selected from that row.
//
// Ties resolve to the lowest token id, which is what the greedy sampler does as well
// (src/llama-sampler.cpp, llama_sampler_greedy_apply: only a strictly greater logit moves the
// selection).
//
// The body is one int32 per row of the
// reply's row table -- the table is what makes the answer addressable when only *some* rows of a
// frame were flagged ROW_FLAG_VERIFY. A table row that was not flagged VERIFY carries -1 ("no
// per-row answer for this row; read it through whichever channel you asked for"), which is why the
// section stays table-wide: the h_nextn section and this one share the table, and a hole in one of
// them would need a second row list to describe it.
constexpr uint8_t RSP_FLAG_VERIFY_ROWS = 0x02;

// rsp_header::flags: the reply to an OP_DECODE that carried
// DECODE_FLAG_NEXTN_ROWS. n_embd is the width of those rows (not 0, unlike the other two channel
// replies), and the body appends n_rows * n_embd f32 to whatever the reply already carries, in
// this order: the row table, then the per-row token ids (if RSP_FLAG_VERIFY_ROWS), then the h_nextn
// rows, then the candidate pairs (if RSP_FLAG_CANDIDATES) or the logits rows. n_rows is the row
// table's length, and the h rows are in table order -- so the head knows which batch row each h row
// belongs to without either side assuming "the last one".
//
// The rows are the ones the frame flagged ROW_FLAG_NEXTN (plus any
// other channel's rows that share the table -- see RSP_FLAG_VERIFY_ROWS for why the table is
// shared). A mixed frame therefore ships the verify block's rows *and* the plain decode's row, and
// the head sorts them per sequence by the row's own seq_id.
//
// They are raw f32, the same representation the local chain reads out of its tail context: this is
// the MTP head's input, and quantizing or truncating it would change the draft tokens, which is
// exactly what the draft/verify comparison in verify/verify_mtp.sh is there to catch.
constexpr uint8_t RSP_FLAG_NEXTN_ROWS = 0x04;

// rsp_header::flags: the reply to an OP_DECODE that carried DECODE_FLAG_LAYER_INP and
// that this stage owns at least one of the named layers to answer for.
//
// rsp_header::n_taps is the number of ids answered, n_embd their width, and the body ends with those
// n_taps int32 source-layer ids followed by n_rows * n_taps * n_embd f32: the first tap's row for
// batch row 0, the second tap's row for row 0, ..., then row 1. n_rows is the frame's own row count,
// which the head knows because it sent the frame; a stage that answered another count is caught by
// n_bytes, so no count is ever inferred from a byte length.
//
// The tap section is the *last* section of every reply, after the per-row ids, the h_nextn rows and
// the candidates or logits row. Appending it there means a reply with taps parses exactly like the
// reply to a frame without them, with one block added behind it: the block is read once per frame,
// after everything the frame already asked for.
//
// n_taps = 0 with the flag clear is the answer of a stage that owns none of the named layers: it
// ignores the request instead of refusing the frame, the same way a stage without a NextN block
// ignores DECODE_FLAG_NEXTN_ROWS. A missing tap degrades one request's drafts; refusing the frame
// would fail it, and a draft is an optimization (llama-ext.h).
constexpr uint8_t RSP_FLAG_LAYER_INP = 0x08;

// The body of OP_MTP_DRAFT.
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
    // Drafts to return. There is no protocol constant for this: the head asks for
    // what its own draft block is (common_params_speculative_draft::n_max), and the stage bounds it
    // by the output-row capacity it was built with (--split-verify-rows), which is the same number
    // that bounds a verify frame. A round that asked for more than that could not be verified.
    uint32_t k;
    uint32_t _pad;
};

static_assert(sizeof(mtp_req) == 16, "mtp_req is a wire structure; keep its size explicit");

// The body section a DECODE_FLAG_LAYER_INP frame appends after everything else in the
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

// The steps a tail must run before it can answer with candidates, in the order
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

// What the tail must know to reproduce the steps up to and including top_k.
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
    // The handle names a snapshot this stage no longer holds -- its snapshot
    // cache evicted it, or this process is not the one that minted it. Nothing was applied, so
    // the chain is exactly where it was before the restore: the head answers "no checkpoint"
    // (re-process the prompt) instead of declaring the chain dead. See STATE_FLAG_HANDLE.
    ST_STALE    = 4,
    // This stage cannot draft. Either it never had a NextN/MTP block, its MTP
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
// An OP_DECODE that sets DECODE_FLAG_CANDIDATES follows the hidden rows with one
// cand_req, then n_bias * {int32 id, float bias}, then n_hist * int32 token. Every section is
// self-describing, so a tail reads exactly what the previous one told it to; a tail that does not know the block rejects
// the frame on the version check instead of parsing the candidate block as the next header.
//
// An OP_DECODE that sets DECODE_FLAG_VERIFY_ROWS adds nothing to the frame. The
// rows are already described by n_tokens / n_embd / n_pos, and the rows the answer must cover are
// all of them -- that is the whole point of the flag, so there is nothing to negotiate.
//
// An OP_DECODE that sets DECODE_FLAG_NEXTN_ROWS also adds nothing to the frame --
// the rows it asks about are the ones the frame already describes. Only the reply grows, by
// n_rows * n_embd floats (see RSP_FLAG_NEXTN_ROWS).
//
// An OP_DECODE that sets DECODE_FLAG_LAYER_INP appends one layer_inp_req and then its
// n_taps int32 layer ids, after the candidate block when both are present. The ids are source-model
// layer numbers; a stage answers the ones inside its own range and skips the rest, so one frame can be
// sent to every stage of a chain without this side knowing where the cuts are.
//
// EVERY OP_DECODE frame ends with an n_tokens * sizeof(row_req) `rows` section.
// It is last among the fixed-width sections for a reason that is not cosmetic: a stage's rule is
// "read the whole frame body, then validate it", so a section placed before the hidden rows could not
// be validated any earlier -- it would only move where the bytes are read from. Fixed-width blocks
// first, self-describing small sections last is the convention cand_req and layer_inp_req
// already follow.
//
// The candidate block moved to the *end* of the frame, behind the
// rows section. It has to: its index names rows, and a reader that has not read the rows yet cannot
// check that the index describes rows this frame actually carries (or that every ROW_FLAG_CANDS row
// has an entry). The block is now one section for the whole frame instead of one per frame:
//   uint32 n_cand_rows, then n_cand_rows * cand_row_ref {int32 row, uint32 n_bytes}, ascending by
//   row, then each row's payload in that order: cand_req (40 B), n_bias * {int32,float},
//   n_hist * int32. n_bytes is that payload's size, so a reader can skip one row's request without
//   understanding it.
//
// The frame is therefore: header | positions | hidden | [layer_inp block] | rows | [cand block].
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
// One row of an OP_DECODE frame, carried in the `rows` section at the end of the
// frame body. One record per batch row, in batch order, so the stage builds its batch from what the
// head actually decoded instead of from one scalar repeated n_tokens times.
//
// Why seq_id had to become per-row: llama-server's continuous batching puts several slots' tokens in
// one batch (tools/server/server-context.cpp, pre_decode), each with its own seq_id, its own position
// and its own output flag. A frame that named one sequence for the whole batch would have written
// every row into that one sequence's KV -- plausible-looking wrong tokens, no error anywhere. That is
// why the driving side refused n_seq_max > 1 until this version.
//
// Why the flags byte, and why it rides here: the stage needs to know which rows the caller samples
// from, because that is what decides which rows are marked as outputs in its own batch. A
// frame-level convention ("the last row", or "every row of a verify frame") cannot
// express a batch where one sequence is prefilling and another is decoding.
//
// The section is fixed width (8 B per row) and always present, not behind a frame flag: per-row
// seq_id is not a capability a chain can decline, it is what the frame *is*. 512 rows is
// 4 KB next to the 8 MiB of hidden rows in the same frame, so there is nothing to save by making it
// conditional or compressed -- and a variable-length section would need its own length field, i.e.
// one more thing both ends must agree on.
struct row_req {
    int32_t seq_id;    // the sequence this row belongs to (llama_batch::seq_id[i][0])
    uint8_t flags;     // ROW_FLAG_*; 0 is "plain row, no extra channel"
    uint8_t _pad[3];   // explicit, so the record has the same width on both ends
};

static_assert(sizeof(row_req) == 8, "row_req is a wire structure; keep its size explicit");

// row_req::flags: this row's logits are sampled by the caller.
//
// This is the per-row form of what the batch's own `output` flag means on the driving side, and it
// is what the stage writes into batch.logits[i]. A prefill frame marks one row (the last prompt
// token, the one the first generated token comes from); a verify frame marks every row; a mixed
// frame marks one row per sequence that is sampling. A reply that carries logits answers exactly
// these rows, in ascending order, and n_logits says how many.
constexpr uint8_t ROW_FLAG_LOGITS = 0x01;

// row_req::flags: the per-sequence speculative channels.
//
// The three bits below are what makes a channel per *row* instead of per frame. A frame-level bool
// could only ever describe "the one sequence this frame belongs to": a
// batch where sequence A verifies k + 1 draft rows and sequence B decodes one token has two
// different answers to "does this row want a token id / an h row / candidates", and the flags byte
// is the only place in the frame where that fits without a new section.
//
// ROW_FLAG_CANDS  -- this row wants the candidate channel, and the frame's candidate block has an
//                    entry for it (its own cand_req, bias and history: two sequences have two
//                    top_k values and two penalty windows, so one shared request cannot serve both)
// ROW_FLAG_VERIFY -- this row wants a token id (the argmax of its row) instead of logits
// ROW_FLAG_NEXTN  -- this row wants its h_nextn row (the MTP draft's input for that position)
//
// Two rows may carry different bits; one row may not carry both ROW_FLAG_CANDS and ROW_FLAG_VERIFY,
// because those two replace the same answer (that mutual exclusion is per row, not per frame). A
// frame that sets both bits on one row is refused by the stage (ST_BAD_REQ) and asserted against
// by the head, which is the side that builds the flags.
constexpr uint8_t ROW_FLAG_CANDS  = 0x02;
constexpr uint8_t ROW_FLAG_VERIFY = 0x04;
constexpr uint8_t ROW_FLAG_NEXTN  = 0x08;

// One entry of a candidate block's index (see the frame layout note above and
// RSP_FLAG_CANDIDATES for the reply's copy). n_bytes is the size of this row's payload, which is
// what lets the reader walk the block without re-deriving cand_req's variable part.
struct cand_row_ref {
    int32_t  row;
    uint32_t n_bytes;
};

static_assert(sizeof(cand_row_ref) == 8, "cand_row_ref is a wire structure; keep its size explicit");

struct req_header {
    uint32_t magic;
    uint32_t version;
    uint8_t  op;
    uint8_t  flags;
    uint16_t _pad;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_pos;     // positions per token (llama_hparams::n_pos_per_embd)
    // For OP_DECODE this is only the *first* row's sequence, kept because it makes
    // a log line or a packet capture readable at a glance. The frame's own answer to "which sequence
    // owns row i" is the `rows` section, and that is the only thing the stage uses.
    int32_t  seq_id;
    int32_t  pos0;
    int32_t  pos1;
    uint32_t n_bytes;   // OP_STATE_LOAD: state bytes that follow; OP_HELLO: one stage_id follows
};

struct rsp_header {
    uint32_t magic;
    uint32_t version;
    uint8_t  status;
    uint8_t  flags;     // RSP_FLAG_*; was padding, so the size is unchanged
    // layer-input taps this reply answers (RSP_FLAG_LAYER_INP), else 0. It rides in a
    // byte that was padding, which is not a compatibility argument (PROTO_VER covers that) but keeps
    // the frame the shape it has always had -- the same trade stage_id::has_nextn makes.
    uint8_t  n_taps;
    uint8_t  _pad;
    // How many rows of logits the logits section answers. 0 when the reply carries
    // no logits row (a relay, a verify answer, a candidate answer). 1 is what a
    // plain decode always answered; more than 1 is the case a multi-sequence frame introduces (a
    // whole-row fallback for a verify frame: k + 1 rows of logits in one reply).
    //
    // It is a separate field rather than "the logits section's width divided by n_vocab" because
    // the reader needs to know the section's width *before* it can trust anything derived from
    // n_bytes, and because a reply may carry logits for a subset of the rows it answered (the row
    // table says which).
    uint32_t n_logits;
    // Rows this reply answers -- also the length of the row table that immediately
    // follows this header. For a relay or verify answer that is the frame's row count, for the other
    // two it is 1; in both cases it is the *length of one table* rather than an implied convention.
    // The h_nextn section and the verify ids section both cover exactly these rows, in table order.
    uint32_t n_tokens;
    uint32_t n_embd;    // of the returned rows; 0 when returning logits
    uint32_t n_vocab;   // set when status is OK and n_logits rows of logits follow
    int32_t  pos_max;   // OP_STATE_SAVE/LOAD: the stage's seq_pos_max at the time of the call
    uint32_t n_bytes;   // OP_STATE_SAVE: state bytes that follow; OP_HELLO: sizeof(stage_id);
                        // OP_DECODE with RSP_FLAG_CANDIDATES: the candidate section's total width
                        //   (4 + n_cand_rows * sizeof(cand_row_ref) + the pairs)
                        // OP_DECODE with RSP_FLAG_VERIFY_ROWS: n_tokens * sizeof(int32_t)
                        // OP_DECODE with RSP_FLAG_LAYER_INP: everything above plus
                        //   n_taps * 4 + n_rows * n_taps * n_embd * 4
};

static_assert(sizeof(rsp_header) == 36, "rsp_header is a wire structure; keep its size explicit");

// A1: THE body layout. This table is the single source of truth for how a reply body is
// cut into sections -- both ends read it, neither adds nor subtracts section widths on its own.
//
// Why it exists: the same rule used to be implemented twice. tools/split-stage/main.cpp added the
// section widths up at seven call sites to produce rsp_header::n_bytes, and
// src/llama-split-remote.cpp derived the same sections back by subtracting them from the n_bytes it
// had received. One was an addition, the other a subtraction, and both had to change for every new
// section -- two diffs per section, and the one place a new section could be added to the
// writer and forgotten in the reader (the symptom being a body read at the wrong offset, i.e. the
// next frame's header parsed out of hidden rows).
//
// The sections, in wire order (see the RSP_FLAG_* notes above):
//
//   row table    n_rows * int32                    always
//   verify ids   n_rows * int32                    RSP_FLAG_VERIFY_ROWS
//   h_nextn rows nextn_rows() * n_embd * f32       RSP_FLAG_NEXTN_ROWS
//   output       candidates | logits rows | hidden rows -- exactly one of the three
//   tap ids      n_taps * int32                    RSP_FLAG_LAYER_INP
//   tap rows     n_frame * n_taps * n_embd * f32   RSP_FLAG_LAYER_INP
//
// The candidate "output" is self-describing too --
//   uint32 n_cand_rows, n_cand_rows * cand_row_ref, then the pairs -- and it is the only section
//   whose width this table cannot know, because it is the chain's top_k (no header field carries
//   it). Everything else still comes from this table.
//
// The row table comes first, immediately after the header, and every per-row
// section below it covers exactly the rows it names, in table order. That is what makes one reply
// able to answer "these three of the frame's five rows" without either side assuming a convention
// like "the last one". It is also what makes n_logits readable: the logits section holds one row per
// entry of the table, n_logits == n_rows when it carries logits, 0 when it carries none.
//
// "one row per entry of the table" is exact for the verify ids and
// the h rows, and it is *almost* exact for logits: a frame may mix rows that want the per-row token
// id (ROW_FLAG_VERIFY) with rows that want their whole logits row (a plain decode sharing the
// batch), and the two answers cannot share one table entry. The rule both ends derive from the
// flags they already agree on is: **the logits section answers the table rows that were not flagged
// ROW_FLAG_VERIFY, in table order**, and n_logits is that count. With no verify row in the frame
// it degenerates to "all of them", i.e. the leading-n_logits rule the field was
// introduced with.
//
// Two row counts are in play, and they are not the same number:
//  - n_rows (the row table's length, and rsp_header::n_tokens) is how many rows the reply *answers*.
//    The verify ids and the h_nextn rows cover these.
//  - n_frame is how many rows the *frame* had. A tap row exists for every position of the batch, so
//    the tap section is sized from it even when the reply answers a single row (a candidate or a
//    plain answer answers one row but was decoded from a whole ubatch).
//
// The candidate section is the one width this table cannot know: it is the chain's top_k, which no
// header field carries (the reader recovers the count from n_bytes / sizeof(token_cand)). It is
// therefore 0 here and filled in by with_cand() on the answering side; the reader subtracts this
// table's total from n_bytes. That subtraction is not a second layout -- every fixed section still
// comes from this table, and the candidates are the only section whose width is decided at runtime.
//
// total is what rsp_header::n_bytes must be. On an answer that carries candidates it is the width of
// everything *but* the candidates (so the count can be recovered); with_cand() returns the layout
// whose total is the whole body.
struct body_layout {
    uint32_t rows_bytes      = 0; // the row table: one int32 per answered row
    uint32_t positions_bytes = 0; // verify ids: one int32 per answered row
    uint32_t cand_bytes      = 0; // candidate pairs; runtime width, see with_cand()
    uint32_t verify_bytes    = 0; // the verify channel's width: the ids section, nothing else
    uint32_t nextn_bytes     = 0; // h_nextn rows
    uint32_t tap_ids_bytes   = 0; // layer-input source ids
    uint32_t tap_rows_bytes  = 0; // layer-input rows, [row][tap][n_embd]
    uint32_t logits_bytes    = 0; // n_logits rows of logits
    uint32_t hidden_bytes    = 0; // the hidden rows (relay answer)
    uint32_t total           = 0; // sum of the sections above; rsp_header::n_bytes

    uint8_t  flags    = 0; // the rsp_header::flags this layout describes
    uint32_t n_tokens = 0; // answered rows: the row table's length (rsp_header::n_tokens)
    uint32_t n_frame  = 0; // frame rows: the stride of the tap section
    uint32_t n_embd   = 0; // row width of the h_nextn and tap sections
    uint32_t n_taps   = 0; // tap ids this reply answers
    uint32_t n_logits = 0; // rows of logits in the output section

    // Where a section starts. Named rather than numeric so a reader does not re-derive the order --
    // re-deriving it is the other half of the duplication this table removes.
    enum class seg : uint8_t { rows, verify_ids, nextn, output, tap_ids, tap_rows };

    uint32_t output_bytes() const { return cand_bytes + logits_bytes + hidden_bytes; }
    uint32_t offset_of(seg s) const;
    // The h_nextn section covers exactly the rows the reply answers -- no convention left.
    uint32_t nextn_rows() const { return n_tokens; }
    bool     has_taps() const { return (flags & RSP_FLAG_LAYER_INP) != 0; }

    // 64-bit width of the whole tap section. The answering side builds it in 32 bits (a frame that
    // large cannot be sent anyway, n_bytes is uint32), but a reader has to be able to ask "is this
    // section alone larger than the body I was told about" without the multiplication wrapping
    // first: n_frame * n_taps * n_embd * 4 does not fit 32 bits at the protocol caps.
    uint64_t tap_section_bytes64() const;

    // The whole body, with the one section whose width only the answering side knows. n_bytes is the
    // candidate section's *total* size: its 4-byte count, its index and its payloads.
    body_layout with_cand_bytes(uint32_t bytes) const;

    // Self-check shared by both ends: does this layout describe exactly this body? A mismatch means
    // the two ends disagree about the frame, and reading on would take the next frame's header for
    // this body.
    bool matches(uint32_t body_bytes) const { return total == body_bytes; }
};

// A1: the self-check on both ends is an explicit `if`, NOT an `assert`, and that is not
// redundant belt-and-braces: libllama and split-stage are built Release (/DNDEBUG), where an assert
// compiles away -- exactly in the binary that talks on the wire. So the stage refuses to send a body
// whose pieces do not add up to `total` (ST_INTERNAL, see answer() in tools/split-stage/main.cpp),
// and the head refuses a reply whose n_bytes is not `total` (-3, see decode() in
// src/llama-split-remote.cpp). A reply that carries candidates is the one exception on the reading
// side: its section width is the remainder, so the head checks that instead. Do not "simplify" these
// into an assert -- that would delete the check from every build that ships.

// A1: the compile-time half of the self-check. layout_of() is called with request flags
// by the head and with reply flags by the stage; the two enums have to stay numerically identical or
// the same table would mean two different things on the two ends.
static_assert(DECODE_FLAG_CANDIDATES == RSP_FLAG_CANDIDATES, "decode/reply candidate bit must agree");
static_assert(DECODE_FLAG_VERIFY_ROWS == RSP_FLAG_VERIFY_ROWS, "decode/reply verify bit must agree");
static_assert(DECODE_FLAG_NEXTN_ROWS  == RSP_FLAG_NEXTN_ROWS,  "decode/reply nextn bit must agree");
static_assert(DECODE_FLAG_LAYER_INP   == RSP_FLAG_LAYER_INP,   "decode/reply layer-inp bit must agree");

inline uint32_t body_layout::offset_of(seg s) const {
    switch (s) {
        case seg::rows:       return 0;
        case seg::verify_ids: return rows_bytes;
        case seg::nextn:      return rows_bytes + positions_bytes;
        case seg::output:     return rows_bytes + positions_bytes + nextn_bytes;
        case seg::tap_ids:    return rows_bytes + positions_bytes + nextn_bytes + output_bytes();
        case seg::tap_rows:   return rows_bytes + positions_bytes + nextn_bytes + output_bytes() +
                                     tap_ids_bytes;
    }
    return 0;
}

inline uint64_t body_layout::tap_section_bytes64() const {
    return (uint64_t) tap_ids_bytes +
           (uint64_t) n_frame * (uint64_t) n_taps * (uint64_t) n_embd * sizeof(float);
}

inline body_layout body_layout::with_cand_bytes(uint32_t bytes) const {
    body_layout l = *this;
    l.cand_bytes = bytes;
    l.total = l.rows_bytes + l.positions_bytes + l.cand_bytes + l.nextn_bytes + l.logits_bytes +
              l.hidden_bytes + l.tap_ids_bytes + l.tap_rows_bytes;
    return l;
}

// A1: derive the body layout of one reply from its header fields.
//
// flags is rsp_header::flags (the RSP_FLAG_* bits; they are numerically the DECODE_FLAG_* ones, see
// the static_asserts above). n_rows is the reply's own row count (rsp_header::n_tokens, i.e. the row
// table's length) and n_frame the frame's row count, which the tap section is strided by -- see the
// note on body_layout for why those are two numbers. n_embd is the row width of the h_nextn and tap
// sections, n_vocab the logits width (0 when the reply carries no logits row), n_taps the ids this
// reply answers (0 when it carries no tap section), n_logits how many rows of logits it carries.
//
// The output section is chosen by the flags, in this order: candidates, else the verify ids (which
// *are* that answer's output), else logits rows when the reply states a vocabulary, else the relay's
// hidden rows. That is exactly the precedence the reader has always used, now stated once.
//
// Candidates and verify ids are not alternatives -- a frame can
// have candidate rows and verify rows at once, and the reply then carries both sections (the verify
// ids before the h rows, the candidates as the output). Candidates and *logits* are still
// alternatives: one output section, one answer, so the head does not arm the candidate channel for a
// frame that also has a row only a whole logits row can answer (see request_remote_candidates in
// tools/server/server-context.cpp).
inline body_layout layout_of(uint8_t flags, uint32_t n_rows, uint32_t n_frame, uint32_t n_embd,
                             uint32_t n_vocab, uint32_t n_taps, uint32_t n_logits) {
    body_layout l;
    l.flags    = flags;
    l.n_tokens = n_rows;
    l.n_frame  = n_frame;
    l.n_embd   = n_embd;
    l.n_taps   = (flags & RSP_FLAG_LAYER_INP) ? n_taps : 0;

    // The row table is not conditional: every reply answers *some* rows and says which.
    l.rows_bytes = n_rows * (uint32_t) sizeof(int32_t);

    l.positions_bytes = (flags & RSP_FLAG_VERIFY_ROWS) ? n_rows * (uint32_t) sizeof(int32_t) : 0;
    l.verify_bytes    = l.positions_bytes;

    l.nextn_bytes = (flags & RSP_FLAG_NEXTN_ROWS)
            ? l.nextn_rows() * n_embd * (uint32_t) sizeof(float)
            : 0;

    if (flags & RSP_FLAG_CANDIDATES) {
        // Filled in by with_cand_bytes(): the section is self-describing (count + index + pairs),
        // and its width is the chain's top_k, which no header field carries.
    } else if (n_vocab > 0) {
        // One row per answered row; n_logits says how many, and a reply that states a
        // vocabulary without saying how many rows is a corrupt frame the reader refuses.
        //
        // This is checked *before* RSP_FLAG_VERIFY_ROWS on purpose: the verify
        // ids are a per-row section of their own (positions_bytes above), not the output, so a
        // frame that mixes verified rows with a row only a whole logits row can answer carries
        // both. The rows this section covers are the table rows not flagged ROW_FLAG_VERIFY.
        l.n_logits     = n_logits;
        l.logits_bytes = n_logits * n_vocab * (uint32_t) sizeof(float);
    } else if (flags & RSP_FLAG_VERIFY_ROWS) {
        // The per-row ids are this answer's whole output; there is no third section.
    } else {
        l.hidden_bytes = n_rows * n_embd * (uint32_t) sizeof(float);
    }

    if (l.has_taps()) {
        l.tap_ids_bytes  = l.n_taps * (uint32_t) sizeof(int32_t);
        l.tap_rows_bytes = n_frame * l.n_taps * n_embd * (uint32_t) sizeof(float);
    }

    l.total = l.rows_bytes + l.positions_bytes + l.cand_bytes + l.nextn_bytes + l.logits_bytes +
              l.hidden_bytes + l.tap_ids_bytes + l.tap_rows_bytes;

    // Runtime half of the self-check: every section this table names has to be inside total. A
    // section added to one side of the sum and not the other is the failure mode this table exists
    // to make impossible, and this catches the day someone edits one line of it.
    assert(l.total == l.rows_bytes + l.positions_bytes + l.cand_bytes + l.nextn_bytes +
                      l.logits_bytes + l.hidden_bytes + l.tap_ids_bytes + l.tap_rows_bytes);
    return l;
}

} // namespace qlh_split
