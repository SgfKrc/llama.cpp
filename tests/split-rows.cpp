// Two numeric preconditions of speculative decoding on a chain, verified bit for bit.
//
// 1. The tail segment must produce one logits row for every row the caller asked logits for.
//    decode_split used to mark only the last row as output_logits: plain decoding happens to
//    need only that row, while speculative verification needs one argmax per row of k+1 rows.
//    Here every row of the whole batch is marked as output and compared row by row against the
//    whole model.
// 2. The chain's MTP draft must match the whole model's MTP numerically: the same (token,
//    h_nextn) input fed to "the MTP context built over the whole model" and "the MTP context
//    built over the tail segment model" must give bit-identical logits. That is the direct
//    evidence that "speculation on the chain and on the whole model compute the same thing";
//    the acceptance rate alone cannot show it.
// 3. [F] MTP draft on the remote chain (over the wire): the remote tail runs k steps on the same
//    (h, token, pos) and the k resulting tokens must match whole model + MTP one by one; the h
//    row it uses must be the whole model's h_nextn (compare those wire bytes bit for bit first).
//    Requires QLH_SPLIT_REMOTE pointing at the tail segment at the head boundary.
//
// Usage: split-rows.exe [model.gguf]   model path = argv[1], else QLH_SPLIT_MODEL; with neither,
//        print [SKIP] and exit 0. Output is one [PASS]/[FAIL] line per check, exit code 0 = all pass.

#include "llama.h"
#include "../src/llama-ext.h"   // llama_set_embeddings_nextn / get_embeddings_nextn_ith

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

// Input batch shared by the whole model and the chain: n tokens, positions 0..n-1, every row
// needs logits. Note that llama_batch_add() was replaced by llama_batch_ext on this branch, so
// the struct is filled in directly.
static llama_batch make_batch(const std::vector<llama_token> & toks) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    b.n_tokens = (int32_t) toks.size();
    for (int32_t i = 0; i < b.n_tokens; ++i) {
        b.token[i]    = toks[i];
        b.pos[i]      = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]   = 1; // every row needs logits
    }
    return b;
}

int main(int argc, char ** argv) {
    g_model_path = argc > 1 && argv[1][0] ? argv[1] : getenv("QLH_SPLIT_MODEL");
    if (g_model_path == nullptr || g_model_path[0] == '\0') {
        printf("[SKIP] split-rows: no model path given; pass a GGUF as argv[1] or set QLH_SPLIT_MODEL\n");
        return 0;
    }

    llama_backend_init();

    llama_model_params vp = llama_model_default_params();
    vp.vocab_only = true;
    llama_model * vmodel = llama_model_load_from_file(model_path(), vp);
    if (vmodel == nullptr) {
        printf("[FAIL] vocab load\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vmodel);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // 5 tokens: at least 3 rows are needed to expose the old "only the last row has logits" behaviour
    const std::vector<llama_token> toks = tokenize(vocab, "The capital of France is");
    const int32_t n_tok = (int32_t) toks.size();
    printf("tokens = %d\n", n_tok);
    if (n_tok < 3) {
        printf("[FAIL] prompt too short\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 256;
    cp.n_batch  = 256;
    cp.n_ubatch = 256;
    cp.n_rs_seq = 8;

    // ---------------------------------------------------------------- [A] whole model, all rows as output
    std::vector<float> base; // [n_tok][n_vocab]
    {
        llama_model_params mp = llama_model_default_params();
        llama_model * m = llama_model_load_from_file(model_path(), mp);
        llama_context * c = llama_init_from_model(m, cp);
        llama_batch b = make_batch(toks);
        if (llama_decode(c, b) != 0) {
            printf("[FAIL] whole-model decode\n");
            return 1;
        }
        base.resize((size_t) n_tok * n_vocab);
        for (int32_t i = 0; i < n_tok; ++i) {
            const float * lg = llama_get_logits_ith(c, i);
            if (lg == nullptr) {
                printf("[FAIL] whole-model get_logits_ith(%d) = null\n", i);
                return 1;
            }
            memcpy(base.data() + (size_t) i * n_vocab, lg, sizeof(float) * n_vocab);
        }
        printf("[A] whole-model rows: %d, argmax[last] = %d\n", n_tok, argmax(base.data() + (size_t) (n_tok - 1) * n_vocab, n_vocab));
        llama_batch_free(b);
        llama_free(c);
        llama_model_free(m);
    }

    // ---------------------------------------------------------------- [B] chain (head 0:20 + tail 20:24), all rows as output
    llama_model_params ph = llama_model_default_params();
    ph.split_layer_lo = 0;
    ph.split_layer_hi = 20;
    llama_model * m_head = llama_model_load_from_file(model_path(), ph);

    llama_model_params pt = llama_model_default_params();
    pt.split_layer_lo = 20;
    pt.split_layer_hi = 24;
    llama_model * m_tail = llama_model_load_from_file(model_path(), pt);

    if (m_head == nullptr || m_tail == nullptr) {
        printf("[FAIL] split model load\n");
        return 1;
    }

    llama_context_params cp_head = cp;
    cp_head.embeddings            = true;
    cp_head.n_outputs_max         = 0;
    cp_head.n_outputs_max_per_seq = 0;

    llama_context_params cp_tail = cp;
    cp_tail.n_outputs_max         = 0;
    cp_tail.n_outputs_max_per_seq = 0;

    llama_context * c_head = llama_init_from_model(m_head, cp_head);
    llama_context * c_tail = llama_init_from_model(m_tail, cp_tail);
    if (c_head == nullptr || c_tail == nullptr) {
        printf("[FAIL] split context create\n");
        return 1;
    }
    // The tail must produce h_nextn (the MTP input), so enable it here; the whole-model side does the same for the reference
    llama_set_embeddings_nextn(c_tail, true, /*masked*/ false);

    if (llama_split_attach(c_head, &c_tail, 1) != 1) {
        printf("[FAIL] llama_split_attach\n");
        return 1;
    }

    std::vector<float> chain;
    std::vector<float> h_tail; // h_nextn of the last row
    {
        llama_batch b = make_batch(toks);
        if (llama_decode(c_head, b) != 0) {
            printf("[FAIL] chain decode\n");
            return 1;
        }
        chain.resize((size_t) n_tok * n_vocab);
        int n_null = 0;
        for (int32_t i = 0; i < n_tok; ++i) {
            const float * lg = llama_get_logits_ith(c_head, i);
            if (lg == nullptr) {
                // Old behaviour: rows other than the last have no logits. This is the evidence of where speculative verification would fail.
                n_null++;
                continue;
            }
            memcpy(chain.data() + (size_t) i * n_vocab, lg, sizeof(float) * n_vocab);
        }
        printf("[B] chain rows with logits: %d/%d\n", n_tok - n_null, n_tok);

        const float * h = llama_get_embeddings_nextn_ith(c_tail, n_tok - 1);
        if (h == nullptr) {
            printf("[FAIL] tail nextn embeddings null\n");
            return 1;
        }
        h_tail.assign(h, h + 2048);
        llama_batch_free(b);
    }

    int n_row_ok = 0;
    for (int32_t i = 0; i < n_tok; ++i) {
        const float * lg = llama_get_logits_ith(c_head, i);
        if (lg == nullptr) {
            printf("  row %d: no logits\n", i);
            continue;
        }
        const bool same = memcmp(lg, base.data() + (size_t) i * n_vocab, sizeof(float) * n_vocab) == 0;
        printf("  row %d: argmax chain=%d whole=%d %s\n", i,
               argmax(lg, n_vocab), argmax(base.data() + (size_t) i * n_vocab, n_vocab),
               same ? "bit-identical" : "differs");
        n_row_ok += same ? 1 : 0;
    }
    printf("[%s] every chain logits row matches the whole model bit for bit (%d/%d rows)\n",
           n_row_ok == n_tok ? "PASS" : "FAIL", n_row_ok, n_tok);

    // ---------------------------------------------------------------- [E] remote per-row verify channel
    int n_verify_ok   = 0;
    int n_verify_rows = 0;
    bool verify_all_ok = true;
    //
    // Build a "verify frame": k = 3 draft rows + 1 target row = 4 rows, all with results. This is
    // exactly the shape of the default speculative block
    // (common_params_speculative_draft::n_max = 3).
    //
    // The criterion is not "something came back" but "the 4 returned token ids equal, one by one,
    // the argmax the whole model computes for the same 4 rows": as soon as the remote side gets
    // it wrong (row misalignment, wrong argmax, a missing row) this differs immediately, while
    // the other chain tests (last row only) cannot see it at all.
    //
    // Requires a running tail segment: QLH_SPLIT_REMOTE=host:port (verify.sh does not set it, so
    // this is skipped by default). See verify/verify_mtp.sh for how to start one.
    const char * ep = getenv("QLH_SPLIT_REMOTE");
    if (ep != nullptr && ep[0] != '\0') {
        std::vector<llama_token> toks4(toks.begin(), toks.begin() + 4);

        // Reference: the whole model's per-row argmax for these 4 rows (same batch shape and positions as the remote frame)
        std::vector<int32_t> ref;
        {
            llama_model_params mp = llama_model_default_params();
            llama_model * m = llama_model_load_from_file(model_path(), mp);
            llama_context * c = llama_init_from_model(m, cp);
            llama_batch b = make_batch(toks4);
            if (llama_decode(c, b) != 0) {
                printf("[FAIL] whole-model 4-row decode\n");
                return 1;
            }
            for (int32_t i = 0; i < 4; ++i) {
                const float * lg = llama_get_logits_ith(c, i);
                if (lg == nullptr) {
                    printf("[FAIL] whole-model 4-row get_logits_ith(%d) = null\n", i);
                    return 1;
                }
                ref.push_back(argmax(lg, n_vocab));
            }
            llama_batch_free(b);
            llama_free(c);
            llama_model_free(m);
        }

        llama_model_params ph2 = llama_model_default_params();
        ph2.split_layer_lo = 0;
        ph2.split_layer_hi = 20;
        llama_model * m_head2 = llama_model_load_from_file(model_path(), ph2);
        if (m_head2 == nullptr) {
            printf("[FAIL] remote-chain head load\n");
            return 1;
        }
        llama_context_params cp2 = cp;
        cp2.embeddings = true;
        llama_context * c2 = llama_init_from_model(m_head2, cp2);
        if (c2 == nullptr) {
            printf("[FAIL] remote-chain head context\n");
            return 1;
        }

        char err[512] = {0};
        const char * eps[1] = { ep };
        const int n_attached = llama_split_attach_remote(c2, eps, 1, err, sizeof(err));
        if (n_attached != 1) {
            printf("[FAIL] remote attach (%s): %s\n", ep, err);
            return 1;
        }

        n_verify_rows = 4;
        llama_batch b = make_batch(toks4);
        llama_split_request_verify_rows(c2, 0, true);
        const int rc = llama_decode(c2, b);
        if (rc != 0) {
            printf("[FAIL] remote verify decode rc=%d\n", rc);
            return 1;
        }

        for (int32_t i = 0; i < 4; ++i) {
            const llama_token got = llama_split_verified_token(c2, i);
            const bool same = got != LLAMA_TOKEN_NULL && (int32_t) got == ref[i];
            // No logits row comes back in this frame: the verify channel returns token ids only, so getting logits must be null.
            const bool no_logits = llama_get_logits_ith(c2, i) == nullptr;
            printf("  verify row %d: tail=%d whole-argmax=%d %s%s\n", i, (int) got, ref[i],
                   same ? "same" : "differs", no_logits ? " (no logits, as expected)" : " (unexpected logits)");
            if (same && no_logits) {
                n_verify_ok++;
            }
        }
        if (n_verify_ok != n_verify_rows) {
            verify_all_ok = false;
        }
        printf("[%s] remote chain %d-row argmax matches the whole model row by row (%d/%d rows)\n",
               n_verify_ok == n_verify_rows ? "PASS" : "FAIL", n_verify_rows, n_verify_ok, n_verify_rows);

        llama_batch_free(b);
        llama_split_request_verify_rows(c2, 0, false);
        llama_free(c2);
        llama_model_free(m_head2);
    } else {
        printf("[SKIP] remote per-row verify channel (QLH_SPLIT_REMOTE not set)\n");
    }

    // ---------------------------------------------------------------- [C] whole-model MTP vs [D] tail MTP
    // Identical input: the last token plus its own h_nextn (position pos = n_tok)
    const llama_token tok_last = toks[n_tok - 1];

    auto mtp_logits = [&](llama_context * ctx_emb, llama_model * model_mtp, std::vector<float> & out, std::vector<float> & h_out) -> bool {
        const float * h = llama_get_embeddings_nextn_ith(ctx_emb, n_tok - 1);
        if (h == nullptr) {
            return false;
        }
        h_out.assign(h, h + 2048);

        llama_context_params cpm = llama_context_default_params();
        cpm.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cpm.n_ctx    = 256;
        cpm.n_batch  = 8;
        cpm.n_ubatch = 8;
        cpm.n_seq_max = 1;
        cpm.n_rs_seq  = 0;
        llama_context * c = llama_init_from_model(model_mtp, cpm);
        if (c == nullptr) {
            printf("    (mtp context create failed)\n");
            return false;
        }

        llama_batch_ext * b = llama_batch_ext_init(c);
        const int32_t idx = llama_batch_ext_add_token(b, 0, tok_last);
        const llama_pos p0 = n_tok; // same meaning as the server's pos0: the position right after the prompt
        llama_batch_ext_set_pos(b, idx, &p0);
        const llama_embd st = { h_out.data(), 1, 2048 };
        if (!llama_batch_ext_set_embd_state(b, idx, st)) {
            printf("    (set_embd_state refused)\n");
            llama_batch_ext_free(b);
            llama_free(c);
            return false;
        }
        llama_batch_ext_set_output_logits(b, idx, true);

        const int rc = llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b);
        bool ok = false;
        if (rc == 0) {
            const float * lg = llama_get_logits_ith(c, 0);
            if (lg != nullptr) {
                out.assign(lg, lg + n_vocab);
                ok = true;
            }
        } else {
            printf("    (mtp decode rc=%d)\n", rc);
        }
        llama_batch_ext_free(b);
        llama_free(c);
        return ok;
    };

    // Whole-model-side h: a separate whole-model context used only to get h_nextn
    std::vector<float> h_whole;
    std::vector<float> mtp_whole;
    std::vector<float> h_chain;
    std::vector<float> mtp_chain;
    {
        llama_model_params mp = llama_model_default_params();
        // The whole model does not load MTP layers by default (load_mtp=0; only a tail segment
        // turns it on via tail_keeps). Building a whole-model MTP context as the reference
        // requires turning it on explicitly, otherwise n_layer_nextn=0 and the nextn weights in
        // the MTP graph are empty.
        mp.load_mtp = true;
        llama_model * m = llama_model_load_from_file(model_path(), mp);
        llama_context * c = llama_init_from_model(m, cp);
        llama_set_embeddings_nextn(c, true, /*masked*/ false);
        llama_batch b = make_batch(toks);
        if (llama_decode(c, b) != 0) {
            printf("[FAIL] whole-model decode (nextn)\n");
            return 1;
        }
        const bool ok = mtp_logits(c, m, mtp_whole, h_whole);
        llama_batch_free(b);
        llama_free(c);
        llama_model_free(m);
        if (!ok) {
            printf("[FAIL] whole-model MTP decode\n");
            return 1;
        }
    }
    if (!mtp_logits(c_tail, m_tail, mtp_chain, h_chain)) {
        printf("[FAIL] chain-tail MTP decode (lazy tok_embd?)\n");
        return 1;
    }

    const bool h_same   = memcmp(h_whole.data(), h_chain.data(), 2048 * sizeof(float)) == 0;
    const bool mtp_same = memcmp(mtp_whole.data(), mtp_chain.data(), sizeof(float) * n_vocab) == 0;
    printf("  h_nextn(last row): %s\n", h_same ? "whole model and tail bit-identical" : "differs");
    printf("  MTP logits: argmax whole=%d tail=%d  %s\n",
           argmax(mtp_whole.data(), n_vocab), argmax(mtp_chain.data(), n_vocab),
           mtp_same ? "bit-identical" : "differs");
    printf("[%s] chain MTP draft matches whole-model MTP bit for bit\n", (h_same && mtp_same) ? "PASS" : "FAIL");

    // ---------------------------------------------------------------- [F] remote MTP draft
    //
    // Criterion: let the *remote tail segment* run k MTP steps on the same (h, token, pos) and
    // let "whole model + MTP" run the same k steps on *exactly the same input*; the k resulting
    // tokens must match *one by one*.
    //
    // How the input is made identical: h comes from the h_nextn this remote frame returned (the
    // row the tail computed itself), and that row is first compared bit for bit against the whole
    // model's h_nextn - only then does the test continue. Both sides therefore receive exactly
    // the same bytes, and a difference can only come from "the remote k steps computed it wrong"
    // (position, h chain, argmax, reply row count). This is the most error-prone place, and the
    // one the acceptance rate cannot show.
    //
    // Also requires QLH_SPLIT_REMOTE pointing at a tail segment *matching the head boundary* (this
    // test's head is 0:20, so the tail segment needs --layer-range 20:24 --role tail). See
    // verify/verify_mtp.sh.
    bool remote_mtp_ok = true;
    if (ep != nullptr && ep[0] != '\0') {
        // Reference implementation of whole model + MTP: the same steps as tools/split-stage's
        // OP_MTP_DRAFT (1 row per step, h chained from the previous step's t_h_nextn, token taken
        // from argmax), so tokens can be compared one by one.
        const auto mtp_draft_local = [&](llama_model * model_mtp, const std::vector<float> & h0,
                                         llama_token tok0, llama_pos pos0, int k,
                                         std::vector<int32_t> & out) -> bool {
            llama_context_params cpm = llama_context_default_params();
            cpm.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
            cpm.n_ctx    = 256;
            cpm.n_batch  = 8;
            cpm.n_ubatch = 8;
            cpm.n_seq_max             = 1;
            cpm.n_rs_seq              = 0;
            cpm.n_outputs_max         = 1;
            cpm.n_outputs_max_per_seq = 1;
            llama_context * c = llama_init_from_model(model_mtp, cpm);
            if (c == nullptr) {
                printf("    (whole-model mtp context create failed)\n");
                return false;
            }
            llama_set_embeddings_nextn(c, true, /*masked*/ true);
            llama_memory_clear(llama_get_memory(c), true);

            std::vector<float> h = h0;
            llama_token tok = tok0;
            for (int i = 0; i < k; ++i) {
                llama_batch_ext * b = llama_batch_ext_init(c);
                const int32_t idx = llama_batch_ext_add_token(b, 0, tok);
                const llama_pos p = pos0 + i;
                llama_batch_ext_set_pos(b, idx, &p);
                llama_batch_ext_set_embd_state(b, idx, llama_embd{ h.data(), 1, h.size() });
                llama_batch_ext_set_output_logits(b, idx, true);
                const int rc = llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b);
                if (rc != 0) {
                    printf("    (whole-model mtp step %d rc=%d)\n", i, rc);
                    llama_batch_ext_free(b);
                    llama_free(c);
                    return false;
                }
                const float * lg = llama_get_logits_ith(c, idx);
                if (lg == nullptr) {
                    llama_batch_ext_free(b);
                    llama_free(c);
                    return false;
                }
                out.push_back(argmax(lg, n_vocab));
                tok = out.back();
                if (i + 1 < k) {
                    const float * hn = llama_get_embeddings_nextn_ith(c, idx);
                    if (hn == nullptr) {
                        llama_batch_ext_free(b);
                        llama_free(c);
                        return false;
                    }
                    memcpy(h.data(), hn, h.size() * sizeof(float));
                }
                llama_batch_ext_free(b);
            }
            llama_free(c);
            return true;
        };

        // Remote chain: head 0:20 + remote tail; this frame only asks for the last h_nextn row (plain decoding)
        llama_model_params ph3 = llama_model_default_params();
        ph3.split_layer_lo = 0;
        ph3.split_layer_hi = 20;
        llama_model * m_head3 = llama_model_load_from_file(model_path(), ph3);
        llama_context_params cp3 = cp;
        cp3.embeddings = true;
        llama_context * c3 = m_head3 != nullptr ? llama_init_from_model(m_head3, cp3) : nullptr;
        if (c3 == nullptr) {
            printf("[FAIL] remote-mtp head context\n");
            return 1;
        }
        char err3[512] = {0};
        const char * eps3[1] = { ep };
        if (llama_split_attach_remote(c3, eps3, 1, err3, sizeof(err3)) != 1) {
            printf("[FAIL] remote-mtp attach (%s): %s\n", ep, err3);
            return 1;
        }

        llama_split_request_nextn_rows(c3, 0, true);
        // This stage process already holds seq 0 KV from [E] (positions 0..3); [F] feeds the
        // whole prompt again from position 0, so clear both ends first - which is what
        // llama-server does on a cache miss (seq_rm is broadcast to remote segments through
        // llama_memory_split). Without clearing, the tail rejects this frame with "positions must
        // be consistent", which has nothing to do with MTP but looks like MTP is broken.
        llama_memory_seq_rm(llama_get_memory(c3), 0, -1, -1);
        llama_batch bF = make_batch(toks);
        const int rcF = llama_decode(c3, bF);
        llama_batch_free(bF);
        if (rcF != 0) {
            printf("[FAIL] remote-mtp decode rc=%d\n", rcF);
            return 1;
        }

        const float *   rows_wire = nullptr;
        const int32_t * rows_ids  = nullptr;
        uint32_t        embd_wire = 0;
        const uint32_t n_rows_wire = llama_split_take_nextn_rows(c3, &rows_wire, &rows_ids, &embd_wire);
        bool okF = true;
        // The rows are a table now, not a block anchored at row0.
        const int32_t row0_wire = (rows_ids != nullptr && n_rows_wire == 1) ? rows_ids[0] : -1;
        if (n_rows_wire != 1 || rows_wire == nullptr || embd_wire != 2048 || row0_wire != n_tok - 1) {
            printf("  wire h_nextn rows: n=%u row0=%d n_embd=%u (expect 1 / %d / 2048)\n",
                   n_rows_wire, row0_wire, embd_wire, n_tok - 1);
            okF = false;
        } else {
            const bool wire_same = memcmp(rows_wire, h_whole.data(), 2048 * sizeof(float)) == 0;
            printf("  wire h_nextn(last row): %s\n",
                   wire_same ? "bit-identical to whole-model h_nextn" : "differs from whole-model h_nextn");
            okF = okF && wire_same;
        }

        std::vector<int32_t> ids_remote;
        if (okF) {
            std::vector<float> h_wire(rows_wire, rows_wire + 2048);
            std::vector<llama_token> ids(3);
            uint32_t n_ids = 0;
            const int rcD = llama_context_split_mtp_draft(c3, h_wire.data(), (int32_t) tok_last,
                                                          (int32_t) n_tok, 3, 0,
                                                          ids.data(), &n_ids);
            ids.resize(n_ids);
            if (rcD <= 0 || ids.empty()) {
                printf("  remote mtp draft: rc=%d n=%u\n", rcD, n_ids);
                okF = false;
            } else {
                ids_remote.assign(ids.begin(), ids.end());
            }
        }

        // Reference: whole model + MTP fed the *same* h (the wire row), the same token, the same pos, the same 3 steps
        std::vector<int32_t> ids_whole;
        {
            llama_model_params mp = llama_model_default_params();
            mp.load_mtp = true;
            llama_model * m = llama_model_load_from_file(model_path(), mp);
            if (m == nullptr) {
                printf("[FAIL] whole-model load for remote-mtp reference\n");
                return 1;
            }
            std::vector<float> h_ref = okF ? std::vector<float>(rows_wire, rows_wire + 2048) : h_whole;
            if (!mtp_draft_local(m, h_ref, tok_last, (llama_pos) n_tok, 3, ids_whole)) {
                llama_model_free(m);
                printf("[FAIL] whole-model MTP draft reference\n");
                return 1;
            }
            llama_model_free(m);
        }

        if (okF) {
            const bool same = ids_remote.size() == ids_whole.size();
            int n_same = 0;
            for (size_t i = 0; i < ids_remote.size() && i < ids_whole.size(); ++i) {
                const bool eq = ids_remote[i] == ids_whole[i];
                n_same += eq ? 1 : 0;
                printf("  draft[%zu]: remote=%d whole-mtp=%d %s\n", i, ids_remote[i], ids_whole[i],
                       eq ? "same" : "differs");
            }
            okF = okF && same && n_same == (int) ids_whole.size();
        }

        remote_mtp_ok = okF;
        printf("[%s] remote tail's %zu MTP drafts match whole-model MTP one by one\n",
               okF ? "PASS" : "FAIL", ids_whole.size());

        llama_free(c3);
        llama_model_free(m_head3);
    } else {
        printf("[SKIP] remote MTP draft (QLH_SPLIT_REMOTE not set)\n");
    }

    llama_free(c_head); // split_attach borrows; freeing head before tail is fine
    llama_free(c_tail);
    llama_model_free(m_head);
    llama_model_free(m_tail);
    llama_model_free(vmodel);
    llama_backend_free();

    return (n_row_ok == n_tok && h_same && mtp_same && verify_all_ok && remote_mtp_ok) ? 0 : 1;
}