// N=2 split check: two models (0:20 and 20:24), two contexts, llama_split_attach, and
// whether the combined logits match the whole model in a single context.
//
// This is the first end-to-end evidence that a split context works. If argmax differs, the
// hidden hand-off or the position semantics is wrong, and cross-process testing is pointless.
#include "llama.h"
#include "../src/llama-ext.h"   // relay channel: llama_set_embeddings_nextn / get_embeddings_nextn_ith
#include "../src/llama-split-net.h"    // loopback listener used by the fake stage
#include "../src/llama-split-proto.h"  // the fake stage answers OP_HELLO

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Model path comes from argv[1], else QLH_SPLIT_MODEL (the name verify/verify.sh uses).
// There is deliberately no built-in default: a hardcoded absolute path from someone's machine
// is not acceptable upstream. With neither given, print one line and return 0 (skip).
static const char * g_model_path = nullptr;

static const char * model_path() {
    return g_model_path;
}

static int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> toks(text.size() + 8);
    const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                 toks.data(), (int32_t) toks.size(), true, true);
    if (n < 0) {
        toks.resize(-n);
        llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                       toks.data(), (int32_t) toks.size(), true, true);
    } else {
        toks.resize(n);
    }
    return toks;
}

// This process and the split-stage it talks to must run the same backend: G/G2 compare the
// hidden rows carried over the hop bit for bit, not the rounding of two backend kernels.
// llama_model_default_params() is -1 (whole model on GPU) while split-stage defaults to -ngl 0
// (CPU), so this side is pinned to CPU too. Measured: unpinned, [G] logits[0] differs by 0.9%
// and the remote chain is not bit-identical; pinned, both match bit for bit (same stage
// processes, stage unchanged).
static llama_model_params model_params() {
    llama_model_params p = llama_model_default_params();
    p.n_gpu_layers = 0;
    return p;
}

// Decode a batch and return the argmax of the last output row plus its first 8 values
static int run(llama_context * ctx, const std::vector<llama_token> & toks, float * first8) {
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        printf("    decode rc = %d\n", rc);
        return -1;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        printf("    no logits\n");
        return -1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    for (int i = 0; i < 8; ++i) {
        first8[i] = logits[i];
    }
    return argmax(logits, n_vocab);
}

int main(int argc, char ** argv) {
    g_model_path = argc > 1 && argv[1][0] ? argv[1] : getenv("QLH_SPLIT_MODEL");
    if (g_model_path == nullptr || g_model_path[0] == '\0') {
        printf("[SKIP] split-smoke: no model path given; pass a GGUF as argv[1] or set QLH_SPLIT_MODEL\n");
        return 0;
    }

    llama_backend_init();

    // Get the vocab from a vocab_only load (cheap in memory)
    llama_model_params vp = model_params();
    vp.vocab_only = true;
    llama_model * vmodel = llama_model_load_from_file(model_path(), vp);
    if (vmodel == nullptr) {
        printf("vocab load failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vmodel);

    const std::string text = "你好";
    const std::vector<llama_token> toks = tokenize(vocab, text);
    printf("tokens = %zu\n", toks.size());

    // ONLY_SPLIT=1 skips baseline/[A]/[B]/[C] and runs the chain only: used to check whether
    // the earlier experiments pollute the chain's embedding buffer.
    const bool only_split = getenv("ONLY_SPLIT") != nullptr;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 256;
    cp.n_batch = 256;
    // n_ubatch defaults to 512, larger than the n_batch set here; the shim clamps it to n_batch too.
    cp.n_ubatch = 256;
    // Recurrent layers cannot roll back partially by default (0 = no rollback) and seq_rm fails
    // outright. Enable it only on segments that need trimming; n_ubatch must hold the snapshot depth.
    cp.n_rs_seq = 8;
    // Same reason as model_params(): split-stage defaults to op_offload off while this process
    // defaults to on (the upstream value), and on lets the GPU take over large host-tensor ops.
    // Measured (build-vulkan-new, both sides -ngl 0): the 3-token [G] matches bit for bit on
    // both sides (below op_offload's minimum batch) while the 20-token remote chain does not
    // (first diff at 0). Pinned to off, both match bit for bit.
    cp.op_offload = false;

    // --- baseline: whole model in one context ---
    int base_argmax = -1;
    float base8[8] = {0};
    // The 3-stage chain is compared bit for bit, so the baseline keeps more than argmax and the first 8 values.
    std::vector<float> base_logits;
    if (!only_split) {
        llama_model_params mp = model_params();
        llama_model * m = llama_model_load_from_file(model_path(), mp);
        if (m == nullptr) {
            printf("baseline model load failed\n");
            return 1;
        }
        llama_context * c = llama_init_from_model(m, cp);
        base_argmax = run(c, toks, base8);
        printf("baseline: argmax=%d  logits[0..7]=", base_argmax);
        for (float v : base8) printf(" %.3f", v);
        printf("\n");
        const float * lg = llama_get_logits_ith(c, -1);
        if (lg != nullptr) {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
            base_logits.assign(lg, lg + nv);
        }
        llama_free(c);
        llama_model_free(m);
    }

    // --- experiment B: whole-model input hidden at layer 20 (reference for the head segment output) ---
    std::vector<float> layer20_input;
    if (!only_split) {
        llama_model_params mp = model_params();
        llama_model * m = llama_model_load_from_file(model_path(), mp);
        llama_context_params cpb = cp;
        cpb.embeddings    = true;
        cpb.n_outputs_max = 0;
        cpb.n_outputs_max_per_seq = 0;
        llama_context * c = llama_init_from_model(m, cpb);
        llama_set_embeddings_layer_inp(c, 20, true);   // want the input of layer 20 = output of layer 19

        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
        if (llama_decode(c, b) == 0) {
            // Use the dedicated exit, not get_embeddings_ith: that one returns res->t_embd
            // (after output_norm), not the input of layer 20.
            const float * h = llama_get_embeddings_layer_inp(c, 20);
            printf("[B] whole-model layer-20 input hidden[0..7] = ");
            if (h) {
                for (int i = 0; i < 8; ++i) printf("%.5f ", h[i]);
                layer20_input.assign(h, h + 2048);
            }
            printf("\n");
        } else {
            printf("[B] decode failed\n");
        }
        llama_free(c);
        llama_model_free(m);
    }

    // --- experiment A: head segment only, one context ---
    // Separates "the embeddings_nextn graph itself is broken" from "two contexts sharing a backend conflict".
    if (!only_split) {
        llama_model_params p1 = model_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 20;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);
        if (m1 == nullptr) { printf("A: model load failed\n"); return 1; }

        llama_context_params cp_head = cp;
        cp_head.embeddings            = true;
        cp_head.n_outputs_max         = 0;
        cp_head.n_outputs_max_per_seq = 0;

        fprintf(stderr, "[A] creating single head context\n");
        llama_context * c1 = llama_init_from_model(m1, cp_head);
        fprintf(stderr, "[A] created = %p\n", (void *) c1);
        fprintf(stderr, "[A] decoding\n");

        // The head hands its pre-norm hidden to the next segment through the layer_inp slot
        // (t_layer_inp[n_layer]), not t_embd: t_embd is the value after output_norm and is not the
        // same tensor as [B]'s whole-model layer-20 input (measured: they differ by one RMSNorm).
        // Arm the same slot as [B] so both sides compare the same quantity.
        llama_set_embeddings_layer_inp(c1, 20, true);

        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
        const int rc = llama_decode(c1, b);
        fprintf(stderr, "[A] decode rc = %d\n", rc);
        if (rc == 0) {
            const float * h = llama_get_embeddings_layer_inp(c1, 20);
            printf("[A] head(0:20) output hidden[0..7] = ");
            if (h) { for (int i = 0; i < 8; ++i) printf("%.5f ", h[i]); }
            printf("\n");
            // The regression criterion is "[B] and [A] hidden bit-identical": comparing the first 8 values misses deviations elsewhere.
            if (h && layer20_input.size() == 2048) {
                printf("[A] vs [B] hidden 2048 dims %s\n",
                       memcmp(h, layer20_input.data(), 2048 * sizeof(float)) == 0
                       ? "bit-identical" : "different");
            }
        }
        llama_free(c1);
        llama_model_free(m1);
    }

    // --- experiment C: tail segment alone, fed [B]'s correct hidden ---
    if (!only_split && !layer20_input.empty()) {
        llama_model_params p2 = model_params();
        p2.split_layer_lo = 20;
        p2.split_layer_hi = 24;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);
        llama_context * c2 = llama_init_from_model(m2, cp);

        llama_batch_ext * b = llama_batch_ext_init(c2);
        const llama_embd row = { layer20_input.data(), 1, 2048 };
        llama_batch_ext_add_embd(b, 0, row);
        const llama_pos p0 = 0;
        llama_batch_ext_set_pos(b, 0, &p0);
        llama_batch_ext_set_output_logits(b, 0, true);

        const int rc = llama_process(c2, LLAMA_PROCESS_TYPE_DECODE, b);
        if (rc == 0) {
            const float * lg = llama_get_logits_ith(c2, 0);
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(c2)));
            printf("[C] tail-alone logits[0..7] = ");
            for (int i = 0; i < 8; ++i) printf("%.3f ", lg[i]);
            printf("  argmax=%d\n", argmax(lg, nv));
        } else {
            printf("[C] tail-alone decode rc=%d\n", rc);
        }
        llama_batch_ext_free(b);
        llama_free(c2);
        llama_model_free(m2);
    }

    // --- split: 0:20 + 20:24, two contexts chained ---
    int split_argmax = -1;
    float split8[8] = {0};
    {
        llama_model_params p1 = model_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 20;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_model_params p2 = model_params();
        p2.split_layer_lo = 20;
        p2.split_layer_hi = 24;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);

        printf("head model layers=%d  tail model layers=%d\n",
               m1 ? llama_model_n_layer(m1) : -1, m2 ? llama_model_n_layer(m2) : -1);

        if (m1 == nullptr || m2 == nullptr) {
            printf("split model load failed\n");
            return 1;
        }

        // The head must declare the embedding output at construction: this params field changes
        // the graph output shape and buffer planning cannot be changed at run time. The pre-norm
        // hidden comes out through the layer_inp slot (armed at attach).
        llama_context_params cp_head = cp;
        cp_head.embeddings            = true;
        // Derive the output row count from n_batch instead of the default cap: the embeddings
        // output buffer size is computed from it, and leaving the default makes the graph ask for
        // a larger buffer than allocated. The shim sets it the same way.
        cp_head.n_outputs_max         = 0;  // derive from n_batch
        // Default is 1: one output row per sequence. Embedding output needs one row per token,
        // and leaving the default makes the graph request a larger embedding buffer than
        // allocated. The shim also sets it to 0.
        cp_head.n_outputs_max_per_seq = 0;
        fprintf(stderr, "[smoke] creating head context\n");
        llama_context * c1 = llama_init_from_model(m1, cp_head);
        fprintf(stderr, "[smoke] head context = %p\n", (void *) c1);
        fprintf(stderr, "[smoke] creating tail context\n");
        llama_context * c2 = llama_init_from_model(m2, cp);
        fprintf(stderr, "[smoke] tail context = %p\n", (void *) c2);
        if (c1 == nullptr || c2 == nullptr) {
            printf("split context init failed\n");
            return 1;
        }

        const char * remote_ep = getenv("QLH_SPLIT_REMOTE");
        if (remote_ep != nullptr && remote_ep[0] != '\0') {
            // Cross-process mode: the tail lives in a split-stage process, no local c2.
            const char * eps[1] = { remote_ep };
            char err[256] = {0};
            fprintf(stderr, "[smoke] attaching remote %s\n", remote_ep);
            const int n = llama_split_attach_remote(c1, eps, 1, err, sizeof(err));
            fprintf(stderr, "[smoke] attached %d remote stage(s)%s%s\n", n,
                    n < 0 ? " err=" : "", n < 0 ? err : "");
            if (n < 0) {
                return 1;
            }
        } else {
            llama_context * stages[1] = { c2 };
            fprintf(stderr, "[smoke] attaching local\n");
            // A failure must be treated as one: attach used to be void, so a mis-built chain kept running and only drifted in logits.
            if (llama_split_attach(c1, stages, 1) != 1) {
                printf("llama_split_attach([0,20) + [20,24)) failed\n");
                return 1;
            }
            fprintf(stderr, "[smoke] attached local\n");
        }
        printf("split n_stages = %d\n", llama_split_n_stages(c1));

        fprintf(stderr, "[smoke] decoding\n");
        split_argmax = run(c1, toks, split8);
        fprintf(stderr, "[smoke] decoded\n");
        printf("split   : argmax=%d  logits[0..7]=", split_argmax);
        for (float v : split8) printf(" %.3f", v);
        printf("\n");

        llama_free(c1);
        llama_free(c2);
        llama_model_free(m1);
        llama_model_free(m2);
    }

    // --- experiment H: after a remote stage fails the chain is marked untrusted, but the process does not abort ---
    // Regression test for the measured problem "one network hiccup = the whole server dies". The
    // observed shape: a stage hangs in recv for 120 s, seq_rm fails, common.cpp calls GGML_ABORT.
    // Here it is split into three deterministic claims:
    //   1. a failed remote seq_rm returns false, marks the chain (llama_context_split_failed() == 1)
    //      and names the stage, endpoint and range in the reason;
    //   2. the next decode must be refused (-3); never compute tokens from a KV that has drifted;
    //   3. a second seq_rm does not touch the network (fails immediately), otherwise every later
    //      request waits another 120 s.
    // A fake stage that accepts once and closes at once replaces a hanging real stage: attach has
    // no handshake, so this is "the chain was just built and the peer vanished", with no 120 s
    // timeout and a deterministic test.
    {
        // Whole-model reference: without a remote stage there is never an untrusted chain
        {
            llama_model_params pm = model_params();
            llama_model * m = llama_model_load_from_file(model_path(), pm);
            llama_context * c = m != nullptr ? llama_init_from_model(m, cp) : nullptr;
            if (c != nullptr) {
                printf("[H] whole model: split_failed=%d failure='%s' (expect 0 and empty)\n",
                       (int) llama_context_split_failed(c), llama_context_split_failure(c));
                llama_free(c);
            }
            if (m != nullptr) llama_model_free(m);
        }

        // Counter-example input: make seq_rm fail on the whole model and see which branch it takes.
        // A hybrid model refuses a partial rollback when n_rs_seq=0 (the rollback <= n_rs_seq check
        // in llama-memory-recurrent.cpp), so "decode 3 tokens then seq_rm(0, 2, -1)" is a whole-model
        // seq_rm that always fails. It returns false while split_failed is still 0, so the if in
        // common_context_seq_rm takes the original GGML_ABORT branch and nothing changed for
        // non-chain setups. Running it for real would kill this process, so only the input and
        // the decision are shown here.
        {
            llama_context_params cp0 = cp;
            cp0.n_rs_seq = 0; // no rollback snapshots
            llama_model_params pm = model_params();
            llama_model * m = llama_model_load_from_file(model_path(), pm);
            llama_context * c = m != nullptr ? llama_init_from_model(m, cp0) : nullptr;
            if (c != nullptr) {
                const std::vector<llama_token> four = tokenize(vocab, "你好，世界");
                llama_batch b = llama_batch_get_one(const_cast<llama_token *>(four.data()), (int32_t) four.size());
                const int rc = llama_decode(c, b);
                const bool rm = rc == 0 && llama_memory_seq_rm(llama_get_memory(c), 0, 2, -1);
                printf("[H] whole model n_rs_seq=0: decode rc=%d, seq_rm(0,2,-1)=%d, split_failed=%d "
                       "(expect a false seq_rm with split_failed=0 -> common.cpp keeps GGML_ABORT)\n",
                       rc, (int) rm, (int) llama_context_split_failed(c));
                llama_free(c);
            }
            if (m != nullptr) llama_model_free(m);
        }

        using namespace qlh_split::net;
        sock_t ls = invalid;
        uint16_t port = 0;
        if (!global_init()) {
            printf("[H] WSAStartup failed\n");
        } else {
            ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        }
        sockaddr_in addr{};
        bool listening = false;
        if (valid(ls)) {
            addr.sin_family      = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            addr.sin_port        = 0; // let the system pick a port, so tests do not fight over one
            int alen = (int) sizeof(addr);
            listening =
                bind(ls, (const sockaddr *) &addr, sizeof(addr)) == 0 &&
                getsockname(ls, (sockaddr *) &addr, &alen) == 0 &&
                listen(ls, 1) == 0;
            if (listening) {
                port = ntohs(addr.sin_port);
            }
        }

        if (!listening) {
            printf("[H] cannot listen on loopback\n");
        } else {
            llama_model_params p1 = model_params();
            p1.split_layer_lo = 0;
            p1.split_layer_hi = 20;
            llama_model * m1 = llama_model_load_from_file(model_path(), p1);

            llama_context_params cph = cp;
            cph.embeddings            = true;
            cph.n_outputs_max         = 0;
            cph.n_outputs_max_per_seq = 0;

            llama_context * c1 = m1 != nullptr ? llama_init_from_model(m1, cph) : nullptr;

            const std::string ep = "127.0.0.1:" + std::to_string(port);
            const char * eps[1] = { ep.c_str() };
            char err[512] = {0};
            const int n = c1 != nullptr
                    ? llama_split_attach_remote(c1, eps, 1, err, sizeof(err))
                    : -1;
            printf("[H] attach a stage that is about to disappear (%s) -> %d (expect 1)\n",
                   ep.c_str(), n);

            if (n == 1) {
                // v4: a successful attach must first complete one OP_HELLO. This fake stage answers
                // with the identity "the segment head expects", so the test still exercises the later
                // "stage dies mid-flight" instead of attach being rejected by the handshake. Echoing
                // the expected value is the simplest form: this segment is supposed to be the tail
                // head expects.
                sock_t peer = accept(ls, nullptr, nullptr);
                if (valid(peer)) {
                    qlh_split::req_header hello{};
                    qlh_split::stage_id   expect{};
                    if (recv_all(peer, &hello, sizeof(hello)) &&
                        hello.magic   == qlh_split::MAGIC &&
                        hello.version == qlh_split::PROTO_VER &&
                        hello.op      == qlh_split::OP_HELLO &&
                        recv_all(peer, &expect, sizeof(expect))) {
                        qlh_split::stage_id self{};
                        self.layer_lo    = expect.layer_lo;
                        self.layer_hi    = expect.layer_hi; // the tail head expects; upper bound known
                        self.n_layer_src = expect.n_layer_src;
                        self.n_embd_inp  = expect.n_embd_inp;
                        self.n_embd_out  = expect.n_embd_out;
                        self.n_vocab     = expect.n_vocab;
                        self.role        = qlh_split::ROLE_TAIL;
                        // v10: echo the four chain identity numbers as well. They are not optional:
                        // head compares them field by field at attach and any missing one rejects the
                        // handshake, so the test would no longer reach the "stage dies mid-flight"
                        // case it is meant to cover.
                        self.n_seq_max     = expect.n_seq_max;
                        self.n_ctx_seq     = expect.n_ctx_seq;
                        self.n_rs_seq      = expect.n_rs_seq;
                        self.n_verify_rows = expect.n_verify_rows;
                        qlh_split::rsp_header rsp{};
                        rsp.magic   = qlh_split::MAGIC;
                        rsp.version = qlh_split::PROTO_VER;
                        rsp.status  = qlh_split::ST_OK;
                        rsp.n_bytes = (uint32_t) sizeof(self);
                        send_all(peer, &rsp, sizeof(rsp));
                        send_all(peer, &self, sizeof(self));
                    }
                    close_sock(peer);
                }

                llama_memory_t chain = llama_get_memory(c1);

                const int64_t t0 = ggml_time_ms();
                const bool rm1 = llama_memory_seq_rm(chain, 0, 0, -1);
                const int64_t dt1 = ggml_time_ms() - t0;
                printf("[H] seq_rm(0, 0, -1) on the dead stage -> %d in %lld ms (expect 0)\n",
                       (int) rm1, (long long) dt1);
                printf("[H] split_failed after the failure = %d (expect 1)\n",
                       (int) llama_context_split_failed(c1));
                printf("[H] failure reason = %s\n", llama_context_split_failure(c1));

                // Must be refused, not computed from a KV that has drifted
                const std::vector<llama_token> one = { toks[0] };
                llama_batch b = llama_batch_get_one(const_cast<llama_token *>(one.data()), 1);
                const int rc = llama_decode(c1, b);
                printf("[H] decode after the chain died -> %d (expect -3)\n", rc);

                // Trim again: must not touch the network, otherwise every later request waits another 120 s
                const int64_t t1 = ggml_time_ms();
                const bool rm2 = llama_memory_seq_rm(chain, 0, 0, -1);
                const int64_t dt2 = ggml_time_ms() - t1;
                printf("[H] second seq_rm -> %d in %lld ms (expect 0, no socket wait)\n",
                       (int) rm2, (long long) dt2);
            }

            if (c1 != nullptr) llama_free(c1);
            if (m1 != nullptr) llama_model_free(m1);
        }

        if (valid(ls)) {
            close_sock(ls);
        }
    }

    if (base_argmax < 0) {
        printf("\n=== baseline skipped, split argmax=%d ===\n", split_argmax);
        return 0;
    }
    // --- experiment D: seq_rm broadcast ---
    // Trim pos >= 2 and check that the tail segment's own memory shrinks too. Verifying only
    // what the aggregate view returns is not enough: drift appears exactly when one segment
    // trims and the other does not.
    {
        llama_model_params p1 = model_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 20;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_model_params p2 = model_params();
        p2.split_layer_lo = 20;
        p2.split_layer_hi = 24;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);

        llama_context_params cph = cp;
        cph.embeddings    = true;
        cph.n_outputs_max = 0;
        cph.n_outputs_max_per_seq = 0;

        llama_context * c1 = llama_init_from_model(m1, cph);
        llama_context * c2 = llama_init_from_model(m2, cp);
        llama_context * stages[1] = { c2 };
        const int attached = llama_split_attach(c1, stages, 1);

        // Use a multi-token input, otherwise pos is only 0 and there is nothing to trim at 1 or above
        const std::vector<llama_token> many = tokenize(vocab, "你好，世界");
        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(many.data()), (int32_t) many.size());
        const int rc = attached == 1 ? llama_decode(c1, b) : -1;
        printf("[D] decode %zu tokens rc=%d (attached=%d)\n", many.size(), rc, attached);
        if (rc == 0) {
            llama_memory_t chain = llama_get_memory(c1);
            llama_memory_t tailm = llama_get_memory(c2);
            const llama_pos h0 = llama_memory_seq_pos_max(chain, 0);
            const llama_pos t0 = llama_memory_seq_pos_max(tailm, 0);
            printf("[D] before: chain_pos_max=%d tail_pos_max=%d\n", (int) h0, (int) t0);

            const bool removed = llama_memory_seq_rm(chain, 0, 2, -1);
            const llama_pos h1 = llama_memory_seq_pos_max(chain, 0);
            const llama_pos t1 = llama_memory_seq_pos_max(tailm, 0);
            printf("[D] seq_rm(p0=2) = %d, after: chain_pos_max=%d tail_pos_max=%d\n",
                   (int) removed, (int) h1, (int) t1);
            printf("[D] %s\n", (h1 < h0 && t1 < t0) ? "broadcast applied: both segments trimmed"
                   : (t1 == t0 ? "broadcast failed: tail not trimmed" : "error"));
        }
        llama_free(c1);
        llama_free(c2);
        llama_model_free(m1);
        llama_model_free(m2);
    }

    // --- experiment E: 3-stage chain 0:8 + 8:16 + 16:24 ---
    // [A]/[B] above prove a single segment only, and two segments prove one hand-off. Three
    // segments require both relay segments to export the pre-norm hidden while the tail still
    // gets the right input. Compare logits bit for bit: a matching argmax is far too easy to
    // hit by luck.
    {
        const int ranges[3][2] = {{0, 8}, {8, 16}, {16, 24}};
        llama_model   * models[3] = {nullptr, nullptr, nullptr};
        llama_context * ctxs[3]   = {nullptr, nullptr, nullptr};
        bool ok = true;

        for (int s = 0; s < 3 && ok; ++s) {
            llama_model_params p = model_params();
            p.split_layer_lo = ranges[s][0];
            p.split_layer_hi = ranges[s][1];
            models[s] = llama_model_load_from_file(model_path(), p);
            if (models[s] == nullptr) {
                printf("[E] model [%d, %d) load failed\n", ranges[s][0], ranges[s][1]);
                ok = false;
                break;
            }
            llama_context_params cps = cp;
            if (s < 2) {
                // head and middle relay segments: hidden goes through the layer_inp slot; output rows derive from n_batch.
                cps.embeddings    = true;
                cps.n_outputs_max = 0;
                cps.n_outputs_max_per_seq = 0;
            }
            ctxs[s] = llama_init_from_model(models[s], cps);
            if (ctxs[s] == nullptr) {
                printf("[E] context %d init failed\n", s);
                ok = false;
            }
        }

        if (ok) {
            const int n_att = llama_split_attach(ctxs[0], &ctxs[1], 2);
            printf("[E] attach 0:8 + 8:16 + 16:24 -> %d, n_stages = %d (expect 2)\n",
                   n_att, llama_split_n_stages(ctxs[0]));
            if (n_att == 2) {
                float e8[8] = {0};
                const int e_argmax = run(ctxs[0], toks, e8);
                printf("[E] 3-stage: argmax=%d  logits[0..7]=", e_argmax);
                for (float v : e8) printf(" %.3f", v);
                printf("\n");

                const float * lg = llama_get_logits_ith(ctxs[0], -1);
                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(models[0]));
                bool bit_equal = lg != nullptr && base_logits.size() == (size_t) nv;
                if (bit_equal) {
                    for (int i = 0; i < nv; ++i) {
                        if (memcmp(&lg[i], &base_logits[i], sizeof(float)) != 0) {
                            printf("[E] first diff: logits[%d] chain=%.9g baseline=%.9g\n",
                                   i, lg[i], base_logits[i]);
                            bit_equal = false;
                            break;
                        }
                    }
                }
                printf("[E] %s\n", bit_equal ? "3-stage chain and whole-model logits bit-identical"
                       : "3-stage chain and whole-model logits differ");
            }
        }

        // Free head first, then the segments: head's split_memory holds the segments' memory, and the reverse order leaves dangling views.
        llama_free(ctxs[0]);
        llama_free(ctxs[1]);
        llama_free(ctxs[2]);
        for (int s = 0; s < 3; ++s) {
            llama_model_free(models[s]);
        }
    }

    // --- experiment F: counter-examples: non-contiguous layer ranges must be rejected at attach ---
    // head 0:8 followed by 12:16 (8:12 missing); 0:8 + 8:16 (does not reach the model's last layer
    // 24); and head 8:20 followed by 20:24 (tiles 8..24 contiguously and reaches the last layer,
    // but layers 0..7 were never computed). All three used to be accepted silently and then
    // decoded plausible-looking garbage. A failed check must return -1 and leave the chain
    // single-segment.
    {
        llama_model_params p1 = model_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 8;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_model_params p2 = model_params();
        p2.split_layer_lo = 12;
        p2.split_layer_hi = 16;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);

        llama_model_params p3 = model_params();
        p3.split_layer_lo = 8;
        p3.split_layer_hi = 16;
        llama_model * m3 = llama_model_load_from_file(model_path(), p3);

        // head starts at layer 8: with 20:24 attached the tiling 8..24 is contiguous and covers the
        // last layer; the only flaw is that 0..7 is never computed. This is the counter-example
        // for split_attach's invariant "head must start at layer 0".
        llama_model_params p4 = model_params();
        p4.split_layer_lo = 8;
        p4.split_layer_hi = 20;
        llama_model * m4 = llama_model_load_from_file(model_path(), p4);

        llama_model_params p5 = model_params();
        p5.split_layer_lo = 20;
        p5.split_layer_hi = 24;
        llama_model * m5 = llama_model_load_from_file(model_path(), p5);

        if (m1 == nullptr || m2 == nullptr || m3 == nullptr || m4 == nullptr || m5 == nullptr) {
            printf("[F] model load failed\n");
        } else {
            llama_context * c1 = llama_init_from_model(m1, cp);
            llama_context * c2 = llama_init_from_model(m2, cp);
            llama_context * c3 = llama_init_from_model(m3, cp);
            llama_context * c4 = llama_init_from_model(m4, cp);
            llama_context * c5 = llama_init_from_model(m5, cp);

            if (c1 == nullptr || c2 == nullptr || c3 == nullptr || c4 == nullptr || c5 == nullptr) {
                printf("[F] context init failed\n");
            } else {
                llama_context * gap[1] = { c2 };
                printf("[F] head 0:8 + stage 12:16 (gap 8:12) -> %d (expect -1)\n",
                       llama_split_attach(c1, gap, 1));

                llama_context * short_chain[1] = { c3 };
                printf("[F] head 0:8 + stage 8:16 (ends at 16 of 24) -> %d (expect -1)\n",
                       llama_split_attach(c1, short_chain, 1));

                llama_context * late_head[1] = { c5 };
                printf("[F] head 8:20 + stage 20:24 (head misses 0:8) -> %d (expect -1)\n",
                       llama_split_attach(c4, late_head, 1));

                printf("[F] n_stages after rejected attaches = %d (expect 0)\n",
                       llama_split_n_stages(c1));
            }

            llama_free(c1);
            llama_free(c2);
            llama_free(c3);
            llama_free(c4);
            llama_free(c5);
        }
        llama_model_free(m1);
        llama_model_free(m2);
        llama_model_free(m3);
        llama_model_free(m4);
        llama_model_free(m5);
    }

    // --- common entry for experiments G / G2: the remote chain's endpoint list ---
    // Triggered by QLH_SPLIT_REMOTE2="host:port,host:port"; when unset, G/G2 are skipped entirely.
    std::vector<std::string> remote_eps;
    if (const char * two = getenv("QLH_SPLIT_REMOTE2"); two != nullptr && two[0] != '\0') {
        const std::string s = two;
        size_t p = 0;
        while (true) {
            const size_t c = s.find(',', p);
            if (c == std::string::npos) {
                remote_eps.push_back(s.substr(p));
                break;
            }
            remote_eps.push_back(s.substr(p, c - p));
            p = c + 1;
        }
    }

    // --- experiment G: 3-stage cross-process (local head 0:8 + remote 8:16 + remote 16:24) ---
    // The remote relay segment never worked before (role detection ran it as a tail, the client got
    // logits and used them as hidden, silently computing wrong values), so the first end-to-end run
    // needs strong evidence: compare the whole vocab bit for bit, a matching argmax is far too
    // easy to hit by luck.
    if (!remote_eps.empty() && !base_logits.empty()) {
        const std::vector<std::string> & eps = remote_eps;

        llama_model_params p1 = model_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 8;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_context_params cph = cp;
        // The head hands its last-layer output to the first remote relay (layer_inp slot), the same convention as the local chain.
        cph.embeddings            = true;
        cph.n_outputs_max         = 0;
        cph.n_outputs_max_per_seq = 0;

        llama_context * c1 = m1 != nullptr ? llama_init_from_model(m1, cph) : nullptr;
        if (c1 == nullptr) {
            printf("[G] head 0:8 context init failed\n");
        } else {
            std::vector<const char *> ceps;
            for (const auto & ep : eps) {
                ceps.push_back(ep.c_str());
            }
            char err[512] = {0};
            const int n = llama_split_attach_remote(c1, ceps.data(), (int32_t) ceps.size(), err, sizeof(err));
            printf("[G] attach %zu remote stage(s) -> %d%s%s\n", eps.size(), n,
                   n < 0 ? " err=" : "", n < 0 ? err : "");
            if (n == (int) eps.size()) {
                float g8[8] = {0};
                const int g_argmax = run(c1, toks, g8);
                const float * lg    = llama_get_logits_ith(c1, -1);
                const int     nv    = (int) base_logits.size();

                int  first_diff = -1;
                bool bit_equal  = lg != nullptr;
                if (bit_equal) {
                    for (int i = 0; i < nv; ++i) {
                        if (memcmp(&lg[i], &base_logits[i], sizeof(float)) != 0) {
                            first_diff = i;
                            bit_equal  = false;
                            break;
                        }
                    }
                }
                printf("[G] argmax=%d (baseline %d), logits all %d dims: %s\n",
                       g_argmax, base_argmax, nv, bit_equal ? "bit-identical" : "different");
                if (first_diff >= 0) {
                    printf("[G] first diff: logits[%d] chain=%.9g baseline=%.9g\n",
                           first_diff, lg[first_diff], base_logits[first_diff]);
                }

                // The remote KV is really trimmed: this path did not exist before (seq_rm never
                // reached remote segments). Trim a multi-token input at p0=2; the stage must log
                // OP_SEQ_RM seq=0 p0=2 p1=-1 and pos_max must go from 3 to 1; head's pos_max is
                // the same number.
                llama_memory_t chain = llama_get_memory(c1);
                llama_memory_seq_rm(chain, 0, 0, -1);

                const std::vector<llama_token> many = tokenize(vocab, "你好，世界");
                llama_batch bm = llama_batch_get_one(const_cast<llama_token *>(many.data()), (int32_t) many.size());
                const int rc_many = llama_decode(c1, bm);
                const llama_pos pm0 = llama_memory_seq_pos_max(chain, 0);
                const bool removed = rc_many == 0 ? llama_memory_seq_rm(chain, 0, 2, -1) : false;
                const llama_pos pm1 = llama_memory_seq_pos_max(chain, 0);
                printf("[G] decoded %zu tokens rc=%d, chain pos_max=%d; seq_rm(0, 2, -1) -> %d, pos_max=%d\n",
                       many.size(), rc_many, (int) pm0, removed, (int) pm1);
            }
            llama_free(c1);
        }
        llama_model_free(m1);
    }

    // --- experiment G2: remote chain checkpoint round-trip (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ---
    // Server prefix reuse relies on checkpoints, and a checkpoint blob carries only the recurrent
    // state (qwen35 has no SWA, so PARTIAL_ONLY writes no KV at all). Verify save -> decode again ->
    // restore -> trim the extra suffix -> decode the same segment again: the logits must match the
    // first run bit for bit, and must match the whole model going through the same flow. Both
    // directions are needed, otherwise "a self-consistent but wrong chain restored" would pass.
    if (!remote_eps.empty()) {
        std::vector<llama_token> seq = tokenize(vocab,
                "Name the capital of France. Name the capital of Japan. "
                "Name the capital of Peru. Name the capital of Norway. ");
        if (seq.size() < 40) {
            seq.resize(40, 1000);
        }
        const int n_tail = 20;
        const int n0     = (int) seq.size() - n_tail;

        // llama_batch_get_one() renumbers positions from 0, while "decode after save" must continue
        // after the existing positions; use a batch with explicit positions, otherwise the two
        // decodes overwrite each other's cells and the checkpoint is not what gets tested.
        const auto make_batch = [](const std::vector<llama_token> & toks, int pos0) {
            llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
            b.n_tokens = (int32_t) toks.size();
            for (int i = 0; i < b.n_tokens; ++i) {
                b.token[i]     = toks[i];
                b.pos[i]       = pos0 + i;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = 0;
                b.logits[i]    = i + 1 == b.n_tokens ? 1 : 0;
            }
            return b;
        };

        std::vector<llama_token> head_toks(seq.begin(), seq.begin() + n0);
        std::vector<llama_token> tail_toks(seq.begin() + n0, seq.end());

        // Returns the logits of the re-decode after restore; false on failure.
        const auto roundtrip = [&](llama_context * c, const char * tag, std::vector<float> & out) -> bool {
            const int nv = (int) llama_vocab_n_tokens(vocab);

            // The same pair of split-stage processes was just used by [G] and seq 0 still holds
            // positions; without clearing, the first decode is rejected by the stage because "batch
            // start position < existing position" (the M-RoPE position check).
            llama_memory_seq_rm(llama_get_memory(c), 0, 0, -1);

            llama_batch b0 = make_batch(head_toks, 0);
            int rc = llama_decode(c, b0);
            llama_batch_free(b0);
            if (rc != 0) { printf("[G2] %s: decode(%d) rc=%d\n", tag, n0, rc); return false; }

            const size_t sz = llama_state_seq_get_size_ext(c, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            std::vector<uint8_t> blob(sz);
            const size_t got = sz > 0
                    ? llama_state_seq_get_data_ext(c, blob.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
                    : 0;
            if (sz == 0 || got != sz) {
                printf("[G2] %s: save size=%zu got=%zu\n", tag, sz, got);
                return false;
            }

            llama_batch b1 = make_batch(tail_toks, n0);
            rc = llama_decode(c, b1);
            if (rc != 0) { printf("[G2] %s: decode tail rc=%d\n", tag, rc); llama_batch_free(b1); return false; }
            std::vector<float> first;
            {
                const float * lg = llama_get_logits_ith(c, -1);
                if (lg == nullptr) { printf("[G2] %s: no logits\n", tag); return false; }
                first.assign(lg, lg + nv);
            }

            if (llama_state_seq_set_data_ext(c, blob.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != sz) {
                printf("[G2] %s: restore failed (%zu bytes)\n", tag, sz);
                return false;
            }
            if (!llama_memory_seq_rm(llama_get_memory(c), 0, n0, -1)) {
                printf("[G2] %s: seq_rm(%d, -1) after restore failed\n", tag, n0);
                return false;
            }

            rc = llama_decode(c, b1);
            llama_batch_free(b1);
            if (rc != 0) { printf("[G2] %s: re-decode rc=%d\n", tag, rc); return false; }
            const float * lg = llama_get_logits_ith(c, -1);
            if (lg == nullptr) { printf("[G2] %s: no logits after restore\n", tag); return false; }
            out.assign(lg, lg + nv);

            int  first_diff = -1;
            bool same       = true;
            for (int i = 0; i < nv; ++i) {
                if (memcmp(&out[i], &first[i], sizeof(float)) != 0) {
                    first_diff = i;
                    same       = false;
                    break;
                }
            }
            printf("[G2] %s: checkpoint %zu bytes (%.3f MiB), save->restore->re-decode %s%s\n",
                   tag, sz, sz / 1048576.0, same ? "bit-identical" : "different",
                   first_diff >= 0 ? (" (first diff at " + std::to_string(first_diff) + ")").c_str() : "");
            printf("[G2] %s: first argmax=%d  out argmax=%d  first[0..3]=%.6f %.6f %.6f %.6f\n",
                   tag, argmax(first.data(), nv), argmax(out.data(), nv),
                   first[0], first[1], first[2], first[3]);
            return same;
        };

        std::vector<float> logits_whole;
        std::vector<float> logits_chain;
        bool ok_whole = false;
        bool ok_chain = false;

        { // whole-model baseline
            llama_model_params pm = model_params();
            llama_model * m = llama_model_load_from_file(model_path(), pm);
            llama_context * c = m != nullptr ? llama_init_from_model(m, cp) : nullptr;
            if (c == nullptr) {
                printf("[G2] whole-model context init failed\n");
            } else {
                ok_whole = roundtrip(c, "whole", logits_whole);
            }
            if (c != nullptr) llama_free(c);
            if (m != nullptr) llama_model_free(m);
        }

        // Local 3-stage chain: the same round-trip. The remote chain's state path was added in the
        // aggregate memory while the local chain still takes the original path, so this step is the
        // evidence that the local chain was not broken (and it is stronger than [E]: 20 tokens, so
        // positions other than 0 take part).
        std::vector<float> logits_local;
        bool ok_local = false;
        {
            const int ranges[3][2] = {{0, 8}, {8, 16}, {16, 24}};
            llama_model   * ms[3] = {nullptr, nullptr, nullptr};
            llama_context * cs[3] = {nullptr, nullptr, nullptr};
            bool ok = true;
            for (int st = 0; st < 3 && ok; ++st) {
                llama_model_params p = model_params();
                p.split_layer_lo = ranges[st][0];
                p.split_layer_hi = ranges[st][1];
                ms[st] = llama_model_load_from_file(model_path(), p);
                if (ms[st] == nullptr) { ok = false; break; }
                llama_context_params cps = cp;
                if (st < 2) {
                    cps.embeddings            = true;
                    cps.n_outputs_max         = 0;
                    cps.n_outputs_max_per_seq = 0;
                }
                cs[st] = llama_init_from_model(ms[st], cps);
                if (cs[st] == nullptr) { ok = false; }
            }
            if (ok && llama_split_attach(cs[0], &cs[1], 2) == 2) {
                ok_local = roundtrip(cs[0], "local", logits_local);
            } else {
                printf("[G2] local chain setup failed\n");
            }
            // Free head first: its split_memory holds the segments' memory
            if (cs[0] != nullptr) llama_free(cs[0]);
            if (cs[1] != nullptr) llama_free(cs[1]);
            if (cs[2] != nullptr) llama_free(cs[2]);
            for (int st = 0; st < 3; ++st) {
                if (ms[st] != nullptr) llama_model_free(ms[st]);
            }
        }

        { // remote chain
            llama_model_params p1 = model_params();
            p1.split_layer_lo = 0;
            p1.split_layer_hi = 8;
            llama_model * m1 = llama_model_load_from_file(model_path(), p1);

            llama_context_params cph = cp;
            cph.embeddings            = true;
            cph.n_outputs_max         = 0;
            cph.n_outputs_max_per_seq = 0;

            llama_context * c1 = m1 != nullptr ? llama_init_from_model(m1, cph) : nullptr;
            if (c1 == nullptr) {
                printf("[G2] chain head context init failed\n");
            } else {
                std::vector<const char *> ceps;
                for (const auto & ep : remote_eps) {
                    ceps.push_back(ep.c_str());
                }
                char err[512] = {0};
                if (llama_split_attach_remote(c1, ceps.data(), (int32_t) ceps.size(), err, sizeof(err)) < 0) {
                    printf("[G2] attach failed: %s\n", err);
                } else {
                    ok_chain = roundtrip(c1, "chain", logits_chain);

                    // Counter-example: after save, clear the whole chain (remote pos_max -> -1); the
                    // restore must then fail. This is the only guard against the silent divergence of
                    // "stage cleared or restarted": the stage notices its position is behind the
                    // snapshot and refuses BEFORE applying it, so head is not restored either.
                    llama_memory_t mem = llama_get_memory(c1);
                    llama_batch b0 = make_batch(head_toks, 0);
                    llama_decode(c1, b0);
                    llama_batch_free(b0);

                    const size_t sz = llama_state_seq_get_size_ext(c1, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    std::vector<uint8_t> blob(sz);
                    const size_t got = sz > 0 ? llama_state_seq_get_data_ext(c1, blob.data(), sz, 0,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : 0;
                    llama_memory_seq_rm(mem, 0, 0, -1); // clear the whole chain
                    const size_t back = (got == sz && sz > 0) ? llama_state_seq_set_data_ext(c1, blob.data(), sz, 0,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : sz;
                    printf("[G2] restore a %zu-byte checkpoint onto the cleared chain -> %zu (expect 0)\n",
                           sz, back);
                }
            }
            if (c1 != nullptr) llama_free(c1);
            if (m1 != nullptr) llama_model_free(m1);
        }

        const auto compare = [&](const char * name, const std::vector<float> & other, bool ok_other) {
            bool same = ok_whole && ok_other && logits_whole.size() == other.size() && !logits_whole.empty();
            int  first_diff = -1;
            if (same) {
                for (size_t i = 0; i < logits_whole.size(); ++i) {
                    if (memcmp(&logits_whole[i], &other[i], sizeof(float)) != 0) {
                        first_diff = (int) i;
                        same       = false;
                        break;
                    }
                }
            }
            printf("[G2] %s vs whole after checkpoint restore: %s (%zu dims%s)\n", name,
                   same ? "bit-identical" : "different", logits_whole.size(),
                   first_diff >= 0 ? (" first diff at " + std::to_string(first_diff)).c_str() : "");
        };

        compare("local chain", logits_local, ok_local);
        compare("remote chain", logits_chain, ok_chain);
        if (!ok_whole || !ok_local || !ok_chain) {
            printf("[G2] roundtrip incomplete (whole=%d local=%d remote=%d)\n",
                   (int) ok_whole, (int) ok_local, (int) ok_chain);
        }
    }

    printf("\n=== %s ===\n", base_argmax == split_argmax ? "ARGMAX match" : "ARGMAX mismatch");
    return base_argmax == split_argmax ? 0 : 2;
}
