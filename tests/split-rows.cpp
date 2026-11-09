// [QLH-SPLIT] 链上投机解码的两条数值前提，逐位验证。
//
// 1. tail 段必须为「caller 请求了 logits 的每一行」各产出一行 logits。以前 decode_split 只给
//    最后一行置 output_logits —— 普通解码恰好只需要那一行，投机验证却要 k+1 行各自一个 argmax。
//    这里一次性把整个 batch 的每一行都标成 output，逐行逐位比对整模。
// 2. 链上的 MTP draft 必须与整模 MTP 数值一致：同一个 (token, h_nextn) 输入喂给
//    「整模建的 MTP context」和「tail 段模型建的 MTP context」，logits 要逐位相同。这才是
//    「链上投机和整模投机算的是同一个东西」的直接证据 —— 只看接受率看不出来。
// 3. [F] 远程链的 MTP draft（v8，走 wire）：远程 tail 对同一 (h, token, pos) 跑 k 步，得到的 k 个
//    token 必须与整模 + MTP 逐个相同；而且它用的那一行 h 必须就是整模的 h_nextn（先把这条 wire
//    上的字节逐位比一遍）。需要 QLH_SPLIT_REMOTE 指向 head 分界处的 tail 段。
//
// 用法: split-rows.exe [model.gguf]   模型路径 = argv[1]，否则 QLH_SPLIT_MODEL；都没有则打印
//       [SKIP] 并 exit 0。输出每行 [PASS]/[FAIL]，退出码 0 = 全过。

#include "llama.h"
#include "../src/llama-ext.h"   // llama_set_embeddings_nextn / get_embeddings_nextn_ith

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// [QLH-SPLIT] 模型路径来自 argv[1]，其次 QLH_SPLIT_MODEL（verify/verify.sh 用的就是这个名字）。
// 故意不设内置缺省值：写死某个人机器上的绝对路径，上游不会接受。两样都没给时打印一行说明
// 并 return 0（skip），这样在没有模型的机器上它也不会挂掉。
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

// 整模/链共用的输入 batch：n 个 token，位置 0..n-1，**每一行都要 logits**
// 注意这个分支里 llama_batch_add() 已经被 llama_batch_ext 取代，这里直接填结构体。
static llama_batch make_batch(const std::vector<llama_token> & toks) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    b.n_tokens = (int32_t) toks.size();
    for (int32_t i = 0; i < b.n_tokens; ++i) {
        b.token[i]    = toks[i];
        b.pos[i]      = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]   = 1; // 每一行都要 logits
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

    // 5 个 token：要有 >=3 行，才能看出「只有最后一行有 logits」这个旧行为
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

    // ---------------------------------------------------------------- [A] 整模，全部行 output
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

    // ---------------------------------------------------------------- [B] 链（head 0:20 + tail 20:24），全部行 output
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
    cp_head.embeddings           = true;
    cp_head.embeddings_pre_norm  = true;
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
    // tail 段要产出 h_nextn（MTP 的输入），必须在这里打开；整模那侧同理，用于对照
    llama_set_embeddings_nextn(c_tail, true, /*masked*/ false);

    if (llama_split_attach(c_head, &c_tail, 1) != 1) {
        printf("[FAIL] llama_split_attach\n");
        return 1;
    }

    std::vector<float> chain;
    std::vector<float> h_tail; // 最后一行的 h_nextn
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
                // 旧行为：非最后一行没有 logits。这条就是投机验证会死在哪里的证据。
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
               same ? "逐位相同" : "不同");
        n_row_ok += same ? 1 : 0;
    }
    printf("[%s] 链上每一行 logits 与整模逐位相同 (%d/%d 行)\n",
           n_row_ok == n_tok ? "PASS" : "FAIL", n_row_ok, n_tok);

    // ---------------------------------------------------------------- [E] 远程逐行验证通道
    int n_verify_ok   = 0;
    int n_verify_rows = 0;
    bool verify_all_ok = true;
    //
    // 造一个「验证帧」：k = 3 个 draft + 1 个 target 行 = 4 行，全部要结果。这正好是默认投机块    // （common_params_speculative_draft::n_max = 3）的形状。
    //
    // 判据不是「有没有回东西」，而是「回的 4 个 token id 与整模对这同样 4 行算出的 argmax 逐个
    // 相同」——远程侧一旦算错（行错位、argmax 取错、少回一行），这里立刻不一样，而链上其它测试
    // （只看最后一行）完全看不出来。
    //
    // 需要一个已启动的 tail 段：QLH_SPLIT_REMOTE=host:port（verify.sh 不设它，所以默认跳过）。
    // 启动方式见 verify/verify_mtp.sh。
    const char * ep = getenv("QLH_SPLIT_REMOTE");
    if (ep != nullptr && ep[0] != '\0') {
        std::vector<llama_token> toks4(toks.begin(), toks.begin() + 4);

        // 参照：整模对这 4 行逐行 argmax（与远程帧完全相同的 batch 形状与位置）
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
        cp2.embeddings          = true;
        cp2.embeddings_pre_norm = true;
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
        llama_split_request_verify_rows(c2, true);
        const int rc = llama_decode(c2, b);
        if (rc != 0) {
            printf("[FAIL] remote verify decode rc=%d\n", rc);
            return 1;
        }

        for (int32_t i = 0; i < 4; ++i) {
            const llama_token got = llama_split_verified_token(c2, i);
            const bool same = got != LLAMA_TOKEN_NULL && (int32_t) got == ref[i];
            // 这一帧没有任何一行 logits 回传：验证通道只回 token id，取 logits 必须是 null。
            const bool no_logits = llama_get_logits_ith(c2, i) == nullptr;
            printf("  verify row %d: tail=%d whole-argmax=%d %s%s\n", i, (int) got, ref[i],
                   same ? "相同" : "不同", no_logits ? " (无 logits，符合预期)" : " (意外有 logits)");
            if (same && no_logits) {
                n_verify_ok++;
            }
        }
        if (n_verify_ok != n_verify_rows) {
            verify_all_ok = false;
        }
        printf("[%s] 远程链 %d 行 argmax 与整模逐个相同 (%d/%d 行)\n",
               n_verify_ok == n_verify_rows ? "PASS" : "FAIL", n_verify_rows, n_verify_ok, n_verify_rows);

        llama_batch_free(b);
        llama_split_request_verify_rows(c2, false);
        llama_free(c2);
        llama_model_free(m_head2);
    } else {
        printf("[SKIP] 远程逐行验证通道（未设 QLH_SPLIT_REMOTE）\n");
    }

    // ---------------------------------------------------------------- [C] 整模 MTP vs [D] tail MTP
    // 输入完全相同：最后一个 token + 它自己的 h_nextn（位置 pos = n_tok）
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
        const llama_pos p0 = n_tok; // 与 server 的 pos0 同语义：紧接着 prompt 的位置
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

    // 整模一侧的 h：单独一个整模 context，只用来取 h_nextn
    std::vector<float> h_whole;
    std::vector<float> mtp_whole;
    std::vector<float> h_chain;
    std::vector<float> mtp_chain;
    {
        llama_model_params mp = llama_model_default_params();
        // 整模默认不带 MTP 层（load_mtp=0，只有 tail 段会因 tail_keeps 自动打开）。整模建 MTP
        // context 做对照时必须显式打开，否则 n_layer_nextn=0，MTP 图里 nextn 权重是空的。
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
    printf("  h_nextn(last row): %s\n", h_same ? "整模与 tail 逐位相同" : "不同");
    printf("  MTP logits: argmax whole=%d tail=%d  %s\n",
           argmax(mtp_whole.data(), n_vocab), argmax(mtp_chain.data(), n_vocab),
           mtp_same ? "逐位相同" : "不同");
    printf("[%s] 链上 MTP draft 与整模 MTP 逐位相同\n", (h_same && mtp_same) ? "PASS" : "FAIL");

    // ---------------------------------------------------------------- [F] 远程 MTP draft（v8）
    //
    // 判据（v8 的核心验收）：让 **远程 tail 段** 对同一 (h, token, pos) 跑 k 步 MTP，与「整模 + MTP」
    // 在**完全相同的输入**上跑同样的 k 步，得到的 k 个 token **逐个相同**。
    //
    // 输入是怎么做到「完全相同」的：h 取自远程链这一帧回传的 h_nextn（tail 自己算的那一行），
    // 然后先把这一行与整模的 h_nextn 逐位比一次 —— 相同才继续。所以两边喂进去的字节完全一样，
    // 差异只可能来自「远程那 k 步算错了」（位置、h 链、argmax、回包行数）。这正是最容易错、
    // 而接受率看不出来的地方。
    //
    // 同样需要 QLH_SPLIT_REMOTE 指向一个 **与 head 分界一致的** tail 段（本测试 head 是 0:20，
    // 所以 tail 段要 --layer-range 20:24 --role tail）。见 verify/verify_mtp.sh。
    bool remote_mtp_ok = true;
    if (ep != nullptr && ep[0] != '\0') {
        // 整模 + MTP 的参照实现：与 tools/split-stage 的 OP_MTP_DRAFT 同一套步骤
        // （每步 1 row，h 用上一步的 t_h_nextn 链下去，token 取 argmax），这样可以逐 token 比。
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

        // 远程链：head 0:20 + 远程 tail；这一帧只要求最后的 h_nextn 行（普通解码）
        llama_model_params ph3 = llama_model_default_params();
        ph3.split_layer_lo = 0;
        ph3.split_layer_hi = 20;
        llama_model * m_head3 = llama_model_load_from_file(model_path(), ph3);
        llama_context_params cp3 = cp;
        cp3.embeddings          = true;
        cp3.embeddings_pre_norm = true;
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

        llama_split_request_nextn_rows(c3, true);
        // 这个 stage 进程在 [E] 里已经放过 seq 0 的 KV（位置 0..3）；[F] 要从位置 0 重新喂整个
        // prompt，所以先把两端都清掉 —— 这正是 llama-server 在 cache miss 时做的事（seq_rm 会
        // 经 llama_memory_split 广播到远端段）。不清的话 tail 会以 "positions must be
        // consistent" 拒绝这一帧，那和 MTP 无关，却会看起来像 MTP 坏了。
        llama_memory_seq_rm(llama_get_memory(c3), 0, -1, -1);
        llama_batch bF = make_batch(toks);
        const int rcF = llama_decode(c3, bF);
        llama_batch_free(bF);
        if (rcF != 0) {
            printf("[FAIL] remote-mtp decode rc=%d\n", rcF);
            return 1;
        }

        const float *  rows_wire = nullptr;
        int32_t        row0_wire = -1;
        uint32_t       embd_wire = 0;
        const uint32_t n_rows_wire = llama_split_take_nextn_rows(c3, &rows_wire, &row0_wire, &embd_wire);
        bool okF = true;
        if (n_rows_wire != 1 || rows_wire == nullptr || embd_wire != 2048 || row0_wire != n_tok - 1) {
            printf("  wire h_nextn rows: n=%u row0=%d n_embd=%u (期望 1 / %d / 2048)\n",
                   n_rows_wire, row0_wire, embd_wire, n_tok - 1);
            okF = false;
        } else {
            const bool wire_same = memcmp(rows_wire, h_whole.data(), 2048 * sizeof(float)) == 0;
            printf("  wire h_nextn(last row): %s\n",
                   wire_same ? "与整模 h_nextn 逐位相同" : "与整模 h_nextn 不同");
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

        // 参照：整模 + MTP，喂**同一个** h（wire 那一行）、同一 token、同一 pos、同样 3 步
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
                       eq ? "相同" : "不同");
            }
            okF = okF && same && n_same == (int) ids_whole.size();
        }

        remote_mtp_ok = okF;
        printf("[%s] 远程 tail 的 %zu 个 MTP draft 与整模 MTP 逐个相同\n",
               okF ? "PASS" : "FAIL", ids_whole.size());

        llama_free(c3);
        llama_model_free(m_head3);
    } else {
        printf("[SKIP] 远程 MTP draft（未设 QLH_SPLIT_REMOTE）\n");
    }

    llama_free(c_head); // split_attach 是借用，先释放 head 再释放 tail 无妨
    llama_free(c_tail);
    llama_model_free(m_head);
    llama_model_free(m_tail);
    llama_model_free(vmodel);
    llama_backend_free();

    return (n_row_ok == n_tok && h_same && mtp_same && verify_all_ok && remote_mtp_ok) ? 0 : 1;
}