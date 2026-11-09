// [QLH-SPLIT] DFlash 草稿模型在分段链上要读的 8 个 tap，逐位验证。
//
// DFlash sidecar 的 dflash.target_layers 用的是**源模型**的层号 2/7/12/17/23/28/33/38。
// 链把源模型切成 head [0,32) + tail [32,40)，于是 33/38 落在 tail 上、变成局部 1/6。
// 这里把两条路径的 tap 拼接逐位比较：
//   A. 整模：8 个 tap 全从同一个 40 层 context 读；
//   B. 链：6 个从 head 读（局部即全局 2/7/12/17/23/28），2 个从 tail 读（local 1/6）。
// 同时比较两条路径的 logits（每行）逐位相同。
//
// [QLH-SPLIT] v9：第三个模式是**远程链**。tail 在 split-stage 进程里，33/38 这两层没有任何本地
// context 可以读，只能靠 OP_LAYER_INP 通道把行传回来。因为整模 23.5 GB + 一个 stage 进程放不进
// 31.6 GB 的机器，这个模式分成两次运行（中间由脚本起停 stage）：
//   dflash-taps.exe MODEL --whole-save F --draft DRAFT   # 整模，tap 列表从 DRAFT 的 target_layers 读
//   dflash-taps.exe MODEL --remote-compare F --endpoint 127.0.0.1:9331
//   dflash-taps.exe MODEL --remote-compare F --expect-taps 5   # DSpark：5 个 tap，1 个在远程
// 目标模型和 stage 都由脚本给，tap 列表由 sidecar 决定，所以 DFlash 和 DSpark 用的是同一条命令。
//
// 为什么必须落在单 ubatch：跨 ubatch 切分时 GEMM 累加顺序不同，实测 head 自己的 tap 与
// 整模差 4e-4（29 token / n_ubatch=12）——这是浮点性质、不是 bug，所以逐位比较只在
// 「n_ubatch 一次装下整个 prompt」时才有意义。
//
// 用法: dflash-taps.exe [model.gguf]   模型路径 = argv[1]，否则 QLH_SPLIT_MODEL；都没有则打印
//       [SKIP] 并 exit 0。输出每行 [PASS]/[FAIL]，退出码 0 = 全过。

#include "llama.h"
#include "../src/llama-ext.h"   // llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp
                                // + v9: llama_split_take_layer_inp / llama_model_target_layer_ids

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

// [QLH-SPLIT] 这个测试要 35B 模型（8 路 tap 分布在 40 个 block 上）。路径优先级：
// argv[1] > QLH_SPLIT_MODEL（和 tests/split-smoke.cpp、tests/split-rows.cpp 用同一个环境变量名）。
// 故意不设内置缺省值：写死某个人机器上的绝对路径，上游不会接受。两样都没给时打印一行说明
// 并 return 0（skip），这样在没有模型的机器上它也不会挂掉。
static const char * default_model_path(int argc, char ** argv) {
    if (argc > 1 && argv[1][0]) {
        return argv[1];
    }
    const char * env = getenv("QLH_SPLIT_MODEL");
    return env && env[0] ? env : nullptr;
}

// dflash.target_layers，即源模型的 40 个 block 里的这些「层输入」。
// [QLH-SPLIT] v9: 这是 DFlash 的缺省列表；--draft 给了 sidecar 时改用它的 dflash.target_layers，
// 这样 DSpark（5 个 tap）走的是同一条命令，而且测的是真实实现会读的那几个层。
static const int TARGET_LAYERS[] = {2, 7, 12, 17, 23, 28, 33, 38};
static const uint32_t N_TAPS = 8;

static std::vector<int32_t> g_taps(TARGET_LAYERS, TARGET_LAYERS + N_TAPS);

// 链的切分点：head 载入 [0,32)，tail 载入 [32,40)
static const int32_t HEAD_END = 32;

// 把 llama.cpp 的 INFO/DEBUG 噪声关掉，只留错误：这个测试的输出要能直接贴进验收报告
static void silent_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
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

// n 个 token，位置 0..n-1，每一行都要 logits
static llama_batch make_batch(const std::vector<llama_token> & toks) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    b.n_tokens = (int32_t) toks.size();
    for (int32_t i = 0; i < b.n_tokens; ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 1;
    }
    return b;
}

// 把某 ctx 上已启用的 tap 读成 [n_taps][n_tok][n_embd]
static bool read_taps(llama_context * ctx, const std::vector<std::pair<llama_context *, uint32_t>> & taps,
                      int32_t n_tok, int32_t n_embd, std::vector<std::vector<float>> & out) {
    out.assign(taps.size(), std::vector<float>((size_t) n_tok * n_embd, 0.0f));
    if (n_embd <= 0) {
        printf("[FAIL] n_embd = %d\n", n_embd);
        return false;
    }
    for (size_t k = 0; k < taps.size(); ++k) {
        const float * p = llama_get_embeddings_layer_inp(taps[k].first, taps[k].second);
        if (p == nullptr) {
            printf("  tap %zu (ctx layer %u): null\n", k, taps[k].second);
            return false;
        }
        memcpy(out[k].data(), p, (size_t) n_tok * n_embd * sizeof(float));
    }
    return true;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

// 整模路径的产出
struct result {
    std::vector<std::vector<float>> taps; // [8][n_tok*n_embd]
    std::vector<float>              logits; // [n_tok*n_vocab]
    int32_t                          n_vocab = 0;
    // [QLH-SPLIT] v9：远程链只能拿到「最后一个位置的 logits」（协议里逐行 logits 从不过线，那是 v7
    // 通道存在的理由），所以「链 == 整模」要在能拿到 logits 的每个位置上都比：prefill 的最后一行，
    // 加上贪心续写 n_gen 步的每一步。续写同时把每步的 8 路 tap 也存下来——1 行的帧是这次新通道最
    // 小的形状（8 KB/tap，永远走 bulk 路径），它也该与整模逐位相同。
    std::vector<float>              last_row_logits;        // [n_vocab] the row a chain can produce
    std::vector<int32_t>            gen_tokens;             // [n_gen]
    std::vector<float>              gen_logits;             // [n_gen][n_vocab]
    std::vector<std::vector<float>> gen_taps;               // [n_gen][n_taps][n_embd]
    // [QLH-SPLIT] v9: 一帧「k=2 verify」形状的帧（3 行，逐行 argmax + tap 都是输出）。这是生产里
    // DFlash 在远程链上真正跑的帧形状（DECODE_FLAG_VERIFY_ROWS | DECODE_FLAG_LAYER_INP），也是唯一
    // 同时带「逐行 ids」和「tap 段」的回复形状，所以它必须单独验一次。
    std::vector<int32_t>            vf_ids;                 // [kVerifyRows] argmax per row
    std::vector<std::vector<float>> vf_taps;                // [kVerifyRows][n_taps][n_embd]
};

static const int32_t kGenSteps   = 8;
static const int32_t kVerifyRows = 3; // k = 2 -> 3 rows, the shape a speculative round verifies
static const int32_t kVerifyTokens[kVerifyRows] = { 271, 71093, 12305 };

static int32_t n_tok_of(const std::vector<llama_token> & toks) { return (int32_t) toks.size(); }

static int32_t argmax_of(const float * v, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
}

// last_row_only: mark only the last row as an output, which is the shape a *remote* chain computes
// (one logits row per frame -- per-row logits never cross the wire). A comparison against a whole
// model that computed 15 rows instead compares two different lm_head GEMM shapes, and those differ
// by ~1e-4 for the usual reason (accumulation order), which would look like a channel bug.
static bool run_whole(const std::string & path, const std::vector<llama_token> & toks, int32_t & n_embd,
                      int32_t n_vocab, result & r, bool last_row_only = false) {
    llama_model_params mp = llama_model_default_params();
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (m == nullptr) {
        printf("[FAIL] whole-model load\n");
        return false;
    }

    // 注意：vocab_only 模型的 hparams 没读全，llama_model_n_embd() 会返回 0；
    // 隐藏维必须从真正加载的模型拿，否则下面的逐位比较是对空 buffer 做的（会假通过）。
    n_embd = (int32_t) llama_model_n_embd(m);
    if (n_embd <= 0) {
        printf("[FAIL] whole-model n_embd = %d\n", n_embd);
        llama_model_free(m);
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 256;
    cp.n_batch  = 256;
    cp.n_ubatch = 256;

    llama_context * c = llama_init_from_model(m, cp);
    if (c == nullptr) {
        printf("[FAIL] whole-model context\n");
        llama_model_free(m);
        return false;
    }

    // DFlash 要的 8 路层输入，整模上就是源层号本身
    std::vector<std::pair<llama_context *, uint32_t>> taps;
    for (size_t k = 0; k < g_taps.size(); ++k) {
        taps.emplace_back(c, (uint32_t) g_taps[k]);
        llama_set_embeddings_layer_inp(c, (uint32_t) g_taps[k], true);
    }

    llama_batch b = make_batch(toks);
    if (last_row_only) {
        for (int32_t i = 0; i + 1 < n_tok_of(toks); ++i) {
            b.logits[i] = 0;
        }
    }
    if (llama_decode(c, b) != 0) {
        printf("[FAIL] whole-model decode\n");
        llama_batch_free(b);
        llama_free(c);
        llama_model_free(m);
        return false;
    }

    const int32_t n_tok = (int32_t) toks.size();
    if (!read_taps(c, taps, n_tok, n_embd, r.taps)) {
        printf("[FAIL] whole-model taps\n");
        llama_batch_free(b);
        llama_free(c);
        llama_model_free(m);
        return false;
    }

    r.n_vocab = n_vocab;
    r.logits.assign((size_t) n_tok * n_vocab, 0.0f);
    for (int32_t i = 0; i < n_tok; ++i) {
        if (last_row_only && i + 1 < n_tok) {
            continue; // not an output row in this shape; asking would only log an error
        }
        const float * lg = llama_get_logits_ith(c, i);
        if (lg == nullptr) {
            printf("[FAIL] whole-model logits row %d is null\n", i);
            llama_batch_free(b);
            llama_free(c);
            llama_model_free(m);
            return false;
        }
        memcpy(r.logits.data() + (size_t) i * n_vocab, lg, (size_t) n_vocab * sizeof(float));
    }

    printf("[A] whole model: %d tokens, %u taps\n", n_tok, (unsigned) g_taps.size());

    // [QLH-SPLIT] v9: the greedy continuation, so the remote chain has positions whose logits it can be
    // compared against byte for byte. Each step is a one-row frame -- the smallest shape the tap channel
    // serves, and the one a per-token decode actually uses.
    r.gen_taps.assign(kGenSteps, std::vector<float>((size_t) g_taps.size() * n_embd, 0.0f));
    r.gen_logits.resize((size_t) kGenSteps * n_vocab);
    {
        // sized for the largest frame this loop decodes: the 1-row steps and the kVerifyRows row
        // verify-shaped frame at the end (llama_batch_init allocates exactly n_tokens entries)
        llama_batch gb = llama_batch_init(kVerifyRows, 0, 1);
        llama_pos p = n_tok;
        llama_token tok = argmax_of(r.logits.data() + (size_t) (n_tok - 1) * n_vocab, n_vocab);
        for (int32_t s = 0; s < kGenSteps; ++s) {
            r.gen_tokens.push_back((int32_t) tok);
            gb.n_tokens          = 1;
            gb.token[0]          = tok;
            gb.pos[0]            = p;
            gb.n_seq_id[0]       = 1;
            gb.seq_id[0][0]      = 0;
            gb.logits[0]         = 1;
            if (llama_decode(c, gb) != 0) {
                printf("[FAIL] whole-model generation step %d\n", s);
                llama_batch_free(gb);
                llama_batch_free(b);
                llama_free(c);
                llama_model_free(m);
                return false;
            }
            const float * lg = llama_get_logits_ith(c, 0);
            if (lg == nullptr) {
                printf("[FAIL] whole-model generation step %d has no logits\n", s);
                llama_batch_free(gb);
                llama_batch_free(b);
                llama_free(c);
                llama_model_free(m);
                return false;
            }
            memcpy(r.gen_logits.data() + (size_t) s * n_vocab, lg, (size_t) n_vocab * sizeof(float));
            for (size_t k = 0; k < g_taps.size(); ++k) {
                const float * t = llama_get_embeddings_layer_inp(c, (uint32_t) g_taps[k]);
                if (t == nullptr) {
                    printf("[FAIL] whole-model generation step %d: tap %zu is null\n", s, k);
                    llama_batch_free(gb);
                    llama_batch_free(b);
                    llama_free(c);
                    llama_model_free(m);
                    return false;
                }
                memcpy(r.gen_taps[s].data() + k * (size_t) n_embd, t, (size_t) n_embd * sizeof(float));
            }
            tok = argmax_of(lg, n_vocab);
            p++;
        }
        // [QLH-SPLIT] v9: the shape a speculative round actually verifies -- k + 1 = 3 rows, all of
        // them outputs. The chain answers this frame with per-row ids *and* the tap block
        // (VERIFY_ROWS | LAYER_INP), so it is the one reply shape with both sections.
        gb.n_tokens = kVerifyRows;
        for (int32_t j = 0; j < kVerifyRows; ++j) {
            gb.token[j]     = (llama_token) kVerifyTokens[j];
            gb.pos[j]       = p + j;
            gb.n_seq_id[j]  = 1;
            gb.seq_id[j][0] = 0;
            gb.logits[j]    = 1;
        }
        if (llama_decode(c, gb) != 0) {
            printf("[FAIL] whole-model verify-shaped frame\n");
            llama_batch_free(gb);
            llama_batch_free(b);
            llama_free(c);
            llama_model_free(m);
            return false;
        }
        r.vf_taps.assign(kVerifyRows, std::vector<float>((size_t) g_taps.size() * n_embd, 0.0f));
        r.vf_ids.resize(kVerifyRows);
        for (int32_t j = 0; j < kVerifyRows; ++j) {
            const float * lg = llama_get_logits_ith(c, j);
            if (lg == nullptr) {
                printf("[FAIL] whole-model verify-shaped frame: row %d has no logits\n", j);
                llama_batch_free(gb);
                llama_batch_free(b);
                llama_free(c);
                llama_model_free(m);
                return false;
            }
            r.vf_ids[j] = argmax_of(lg, n_vocab);
            for (size_t k = 0; k < g_taps.size(); ++k) {
                const float * t = llama_get_embeddings_layer_inp(c, (uint32_t) g_taps[k]);
                if (t == nullptr) {
                    printf("[FAIL] whole-model verify-shaped frame: tap %zu is null\n", k);
                    llama_batch_free(gb);
                    llama_batch_free(b);
                    llama_free(c);
                    llama_model_free(m);
                    return false;
                }
                memcpy(r.vf_taps[j].data() + k * (size_t) n_embd, t + (size_t) j * n_embd,
                       (size_t) n_embd * sizeof(float));
            }
        }
        llama_batch_free(gb);
        printf("[A] greedy continuation: %d tokens, first = %d; verify-shaped frame: %d rows\n",
               kGenSteps, r.gen_tokens[0], kVerifyRows);
    }

    llama_batch_free(b);
    llama_free(c);
    llama_model_free(m);
    return true;
}

static bool run_chain(const std::string & path, const std::vector<llama_token> & toks, int32_t n_embd,
                      int32_t n_vocab, result & r) {
    llama_model_params ph = llama_model_default_params();
    ph.split_layer_lo = 0;
    ph.split_layer_hi = HEAD_END;
    llama_model * m_head = llama_model_load_from_file(path.c_str(), ph);

    llama_model_params pt = llama_model_default_params();
    pt.split_layer_lo = HEAD_END;
    pt.split_layer_hi = 40;
    llama_model * m_tail = llama_model_load_from_file(path.c_str(), pt);

    if (m_head == nullptr || m_tail == nullptr) {
        printf("[FAIL] split model load\n");
        return false;
    }
    printf("[B] chain: head layers=%d, tail layers=%d\n",
           (int) llama_model_n_layer(m_head), (int) llama_model_n_layer(m_tail));

    llama_context_params cp_head = llama_context_default_params();
    cp_head.n_ctx                = 256;
    cp_head.n_batch              = 256;
    cp_head.n_ubatch             = 256;
    cp_head.embeddings           = true;
    cp_head.embeddings_pre_norm  = true;
    cp_head.n_outputs_max        = 0;
    cp_head.n_outputs_max_per_seq = 0;

    llama_context_params cp_tail = llama_context_default_params();
    cp_tail.n_ctx                 = 256;
    cp_tail.n_batch               = 256;
    cp_tail.n_ubatch              = 256;
    cp_tail.n_outputs_max         = 0;
    cp_tail.n_outputs_max_per_seq = 0;

    llama_context * c_head = llama_init_from_model(m_head, cp_head);
    llama_context * c_tail = llama_init_from_model(m_tail, cp_tail);
    if (c_head == nullptr || c_tail == nullptr) {
        printf("[FAIL] split context create\n");
        return false;
    }
    if (llama_split_attach(c_head, &c_tail, 1) != 1) {
        printf("[FAIL] llama_split_attach\n");
        return false;
    }

    // 与 DFlash impl 相同的映射：id < head 的层数 -> head 的 id；否则 tail 的 id - head_n
    const int32_t head_n = (int32_t) llama_model_n_layer(m_head);
    std::vector<std::pair<llama_context *, uint32_t>> taps;
    for (size_t k = 0; k < g_taps.size(); ++k) {
        const int32_t id = g_taps[k];
        llama_context * ctx = id < head_n ? c_head : c_tail;
        const uint32_t  lid = id < head_n ? (uint32_t) id : (uint32_t) (id - head_n);
        printf("    tap %zu: global %2d -> %s local %u\n", k, id, id < head_n ? "head" : "tail", lid);
        taps.emplace_back(ctx, lid);
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    llama_batch b = make_batch(toks);
    if (llama_decode(c_head, b) != 0) {
        printf("[FAIL] chain decode\n");
        return false;
    }

    const int32_t n_tok = (int32_t) toks.size();
    if (!read_taps(c_head, taps, n_tok, n_embd, r.taps)) {
        printf("[FAIL] chain taps\n");
        return false;
    }

    r.n_vocab = n_vocab;
    r.logits.resize((size_t) n_tok * n_vocab);
    for (int32_t i = 0; i < n_tok; ++i) {
        const float * lg = llama_get_logits_ith(c_head, i);
        if (lg == nullptr) {
            printf("[FAIL] chain logits row %d is null\n", i);
            return false;
        }
        memcpy(r.logits.data() + (size_t) i * n_vocab, lg, (size_t) n_vocab * sizeof(float));
    }

    llama_batch_free(b);
    llama_free(c_head);
    llama_free(c_tail);
    llama_model_free(m_head);
    llama_model_free(m_tail);
    return true;
}

// [QLH-SPLIT] v9: 整模结果落盘。远程模式要分两次运行（整模 23.5 GB + stage 进程放不进 31.6 GB），
// 所以第一次跑完整模就把它写到文件里，第二次用 stage 起远程链再比。
static const uint32_t kWholeMagic = 0x54464451; // 'QDFT'

static bool whole_save(const char * path, const result & r, int32_t n_tok, int32_t n_embd) {
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        printf("[FAIL] cannot write %s\n", path);
        return false;
    }
    const uint32_t hdr[5] = { kWholeMagic, (uint32_t) n_tok, (uint32_t) r.n_vocab,
                              (uint32_t) n_embd, (uint32_t) g_taps.size() };
    bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1;
    ok = ok && fwrite(g_taps.data(), sizeof(int32_t), g_taps.size(), f) == g_taps.size();
    for (size_t k = 0; k < r.taps.size() && ok; ++k) {
        ok = fwrite(r.taps[k].data(), sizeof(float), r.taps[k].size(), f) == r.taps[k].size();
    }
    ok = ok && fwrite(r.logits.data(), sizeof(float), r.logits.size(), f) == r.logits.size();
    // [QLH-SPLIT] v9: the greedy continuation -- tokens, their logits rows and their taps, so the
    // remote chain can be compared on positions whose logits it can actually produce (a chain answers
    // one logits row per frame; per-row logits never cross the wire).
    const uint32_t n_gen = (uint32_t) r.gen_tokens.size();
    ok = ok && fwrite(&n_gen, sizeof(n_gen), 1, f) == 1;
    ok = ok && fwrite(r.gen_tokens.data(), sizeof(int32_t), r.gen_tokens.size(), f) == r.gen_tokens.size();
    ok = ok && fwrite(r.gen_logits.data(), sizeof(float), r.gen_logits.size(), f) == r.gen_logits.size();
    for (size_t g = 0; g < r.gen_taps.size() && ok; ++g) {
        ok = fwrite(r.gen_taps[g].data(), sizeof(float), r.gen_taps[g].size(), f) == r.gen_taps[g].size();
    }
    const uint32_t n_vf = (uint32_t) r.vf_ids.size();
    ok = ok && fwrite(&n_vf, sizeof(n_vf), 1, f) == 1;
    ok = ok && fwrite(r.vf_ids.data(), sizeof(int32_t), r.vf_ids.size(), f) == r.vf_ids.size();
    for (size_t j = 0; j < r.vf_taps.size() && ok; ++j) {
        ok = fwrite(r.vf_taps[j].data(), sizeof(float), r.vf_taps[j].size(), f) == r.vf_taps[j].size();
    }
    fclose(f);
    if (!ok) {
        printf("[FAIL] short write to %s\n", path);
        return false;
    }
    printf("[A] saved whole-model taps (%u taps, %d tokens) to %s\n",
           (unsigned) g_taps.size(), n_tok, path);
    return true;
}

static bool whole_load(const char * path, result & r, int32_t & n_tok, int32_t & n_embd) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        printf("[FAIL] cannot read %s\n", path);
        return false;
    }
    uint32_t hdr[5] = {};
    bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == kWholeMagic;
    n_tok   = (int32_t) hdr[1];
    r.n_vocab = (int32_t) hdr[2];
    n_embd  = (int32_t) hdr[3];
    const uint32_t n_taps = hdr[4];
    ok = ok && n_taps > 0 && n_taps <= 64 && n_tok > 0 && n_embd > 0 && r.n_vocab > 0;
    std::vector<int32_t> ids(n_taps);
    ok = ok && fread(ids.data(), sizeof(int32_t), ids.size(), f) == ids.size();
    // 落盘用的是这次的 tap 列表：两边必须一致，否则比的是不同的层
    ok = ok && (size_t) n_taps == g_taps.size() && std::memcmp(ids.data(), g_taps.data(),
                                                              g_taps.size() * sizeof(int32_t)) == 0;
    if (!ok) {
        printf("[FAIL] %s is not a whole-model dump of these %zu taps\n", path, g_taps.size());
        fclose(f);
        return false;
    }
    r.taps.assign(n_taps, std::vector<float>());
    for (uint32_t k = 0; k < n_taps && ok; ++k) {
        r.taps[k].resize((size_t) n_tok * n_embd);
        ok = fread(r.taps[k].data(), sizeof(float), r.taps[k].size(), f) == r.taps[k].size();
    }
    r.logits.resize((size_t) n_tok * r.n_vocab);
    ok = ok && fread(r.logits.data(), sizeof(float), r.logits.size(), f) == r.logits.size();
    uint32_t n_gen = 0;
    ok = ok && fread(&n_gen, sizeof(n_gen), 1, f) == 1 && n_gen > 0 && n_gen <= 64;
    if (ok) {
        r.gen_tokens.resize(n_gen);
        ok = fread(r.gen_tokens.data(), sizeof(int32_t), r.gen_tokens.size(), f) == r.gen_tokens.size();
        r.gen_logits.resize((size_t) n_gen * r.n_vocab);
        ok = ok && fread(r.gen_logits.data(), sizeof(float), r.gen_logits.size(), f) == r.gen_logits.size();
        r.gen_taps.assign(n_gen, std::vector<float>((size_t) n_taps * n_embd, 0.0f));
        for (uint32_t g = 0; g < n_gen && ok; ++g) {
            ok = fread(r.gen_taps[g].data(), sizeof(float), r.gen_taps[g].size(), f) == r.gen_taps[g].size();
        }
    }
    uint32_t n_vf = 0;
    ok = ok && fread(&n_vf, sizeof(n_vf), 1, f) == 1 && n_vf == (uint32_t) kVerifyRows;
    if (ok) {
        r.vf_ids.resize(n_vf);
        ok = fread(r.vf_ids.data(), sizeof(int32_t), r.vf_ids.size(), f) == r.vf_ids.size();
        r.vf_taps.assign(n_vf, std::vector<float>((size_t) n_taps * n_embd, 0.0f));
        for (uint32_t j = 0; j < n_vf && ok; ++j) {
            ok = fread(r.vf_taps[j].data(), sizeof(float), r.vf_taps[j].size(), f) == r.vf_taps[j].size();
        }
    }
    fclose(f);
    if (!ok) {
        printf("[FAIL] %s is truncated\n", path);
        return false;
    }
    return true;
}

// [QLH-SPLIT] v9: 远程链跑一遍，tap 一半从本地 head 读、一半从线上取回来的块里读。
// 这是 OP_LAYER_INP 通道的验收：33/38 这两层在这里没有任何 context，只有 wire 上的字节。
static bool run_chain_remote(const std::string & path, const std::string & endpoint,
                             const std::vector<llama_token> & toks, int32_t n_embd,
                             int32_t n_vocab, result & r) {
    llama_model_params ph = llama_model_default_params();
    ph.split_layer_lo = 0;
    ph.split_layer_hi = HEAD_END;
    llama_model * m_head = llama_model_load_from_file(path.c_str(), ph);
    if (m_head == nullptr) {
        printf("[FAIL] head model load\n");
        return false;
    }
    const int32_t head_n = (int32_t) llama_model_n_layer(m_head);
    printf("[C] remote chain: head layers=%d, tail served by %s\n", head_n, endpoint.c_str());

    llama_context_params cp_head = llama_context_default_params();
    cp_head.n_ctx                 = 256;
    cp_head.n_batch               = 256;
    cp_head.n_ubatch              = 256;
    cp_head.embeddings            = true;
    cp_head.embeddings_pre_norm   = true;
    cp_head.n_outputs_max         = 0;
    cp_head.n_outputs_max_per_seq = 0;

    llama_context * c_head = llama_init_from_model(m_head, cp_head);
    if (c_head == nullptr) {
        printf("[FAIL] head context create\n");
        return false;
    }
    const char * eps[1] = { endpoint.c_str() };
    char err[512] = {0};
    if (llama_split_attach_remote(c_head, eps, 1, err, sizeof(err)) < 0) {
        printf("[FAIL] llama_split_attach_remote: %s\n", err);
        return false;
    }

    int32_t n_remote = 0;
    std::vector<uint32_t> remote_ids;
    for (size_t k = 0; k < g_taps.size(); ++k) {
        const int32_t id = g_taps[k];
        if (id < head_n) {
            llama_set_embeddings_layer_inp(c_head, (uint32_t) id, true);
        } else {
            remote_ids.push_back((uint32_t) id);
            n_remote++;
        }
        printf("    tap %zu: source layer %2d -> %s\n", k, id, id < head_n ? "head (local)" : "remote stage");
    }
    // [QLH-SPLIT] v9: 生产路径里这一句由 DFlash impl 的构造函数发出（common/speculative.cpp），
    // 这里没有草稿模型，所以测试自己把「哪些层要跨线」告诉 head context。
    if (!remote_ids.empty()) {
        llama_split_set_layer_inp(c_head, remote_ids.data(), (uint32_t) remote_ids.size());
    }
    printf("[C] %u taps: %d local, %d over the wire\n", (unsigned) g_taps.size(),
           (int) g_taps.size() - n_remote, n_remote);

    llama_batch b = make_batch(toks);
    const int32_t n_tok = (int32_t) toks.size();
    if (llama_decode(c_head, b) != 0) {
        printf("[FAIL] remote chain decode\n");
        llama_batch_free(b);
        return false;
    }

    // [QLH-SPLIT] v9: 线上取回来的块按源层号索引。找不到就是通道没通——这正是这一组要证明的事，
    // 所以这里不退回整模、直接失败。
    const uint32_t n_blocks = llama_split_take_layer_inp_n(c_head);
    printf("[C] wire answered %u layer-input blocks (%zu asked for from the stages)\n",
           n_blocks, (size_t) n_remote);

    bool ok = (int) n_blocks >= n_remote;
    r.taps.assign(g_taps.size(), std::vector<float>());
    int n_from_wire = 0;
    for (size_t k = 0; k < g_taps.size(); ++k) {
        const int32_t id = g_taps[k];
        r.taps[k].resize((size_t) n_tok * n_embd);
        if (id < head_n) {
            const float * p = llama_get_embeddings_layer_inp(c_head, (uint32_t) id);
            if (p == nullptr) {
                printf("[FAIL] local tap %zu (layer %d) has no rows\n", k, id);
                ok = false;
                continue;
            }
            std::memcpy(r.taps[k].data(), p, (size_t) n_tok * n_embd * sizeof(float));
            continue;
        }
        const float * p = nullptr;
        for (uint32_t blk = 0; blk < n_blocks && p == nullptr; ++blk) {
            uint32_t     blk_id = 0, blk_rows = 0, blk_embd = 0;
            const float * q = llama_split_take_layer_inp(c_head, blk, &blk_id, &blk_rows, &blk_embd);
            if (q != nullptr && (int32_t) blk_id == id) {
                if (blk_rows != (uint32_t) n_tok || blk_embd != (uint32_t) n_embd) {
                    printf("[FAIL] wire tap %d answered %u rows x %u wide, expected %d x %d\n",
                           id, blk_rows, blk_embd, n_tok, n_embd);
                    ok = false;
                    continue;
                }
                p = q;
            }
        }
        if (p == nullptr) {
            printf("[FAIL] remote tap %zu (source layer %d) did not arrive on the wire\n", k, id);
            ok = false;
            continue;
        }
        n_from_wire++;
        std::memcpy(r.taps[k].data(), p, (size_t) n_tok * n_embd * sizeof(float));
    }
    printf("[C] %d of %d taps came from the wire (n_tok=%d, %d floats each)\n",
           n_from_wire, n_remote, n_tok, n_tok * n_embd);

    // [QLH-SPLIT] v9: a remote chain answers *one* logits row per frame -- the last one -- because per-row
    // logits never cross the wire (that is what the v7 verify channel exists to avoid). So the comparison
    // "chain == whole model" is made on every position whose logits the chain can produce: the last row of
    // the prefill, then each step of the same greedy continuation the whole-model pass saved.
    r.n_vocab = n_vocab;
    r.logits.assign((size_t) n_tok * n_vocab, 0.0f);
    {
        const float * lg = llama_get_logits_ith(c_head, n_tok - 1);
        if (lg == nullptr) {
            printf("[FAIL] remote chain answered no logits for the last row of the prefill\n");
            ok = false;
        } else {
            r.last_row_logits.assign(lg, lg + n_vocab);
        }
    }

    // The continuation must start from the same token the whole-model pass continued from, so that is
    // taken from the chain's own answer (which the check above just established is byte-equal).
    r.gen_taps.assign(kGenSteps, std::vector<float>((size_t) g_taps.size() * n_embd, 0.0f));
    r.gen_logits.resize((size_t) kGenSteps * n_vocab);
    {
        // sized for the largest frame this loop decodes: the 1-row steps and the kVerifyRows row
        // verify-shaped frame at the end (llama_batch_init allocates exactly n_tokens entries)
        llama_batch gb = llama_batch_init(kVerifyRows, 0, 1);
        llama_pos p = n_tok;
        llama_token tok = argmax_of(r.last_row_logits.data(), n_vocab);
        for (int32_t st = 0; st < kGenSteps && ok; ++st) {
            r.gen_tokens.push_back((int32_t) tok);
            gb.n_tokens     = 1;
            gb.token[0]     = tok;
            gb.pos[0]       = p;
            gb.n_seq_id[0]  = 1;
            gb.seq_id[0][0] = 0;
            gb.logits[0]    = 1;
            if (llama_decode(c_head, gb) != 0) {
                printf("[FAIL] remote chain generation step %d\n", st);
                ok = false;
                break;
            }
            const float * lg = llama_get_logits_ith(c_head, 0);
            if (lg == nullptr) {
                printf("[FAIL] remote chain generation step %d has no logits\n", st);
                ok = false;
                break;
            }
            std::memcpy(r.gen_logits.data() + (size_t) st * n_vocab, lg, (size_t) n_vocab * sizeof(float));
            // [QLH-SPLIT] v9: read this one-row frame's taps back -- local layers from the head context,
            // remote ones from what the stages answered on the wire.
            const uint32_t n_blocks = llama_split_take_layer_inp_n(c_head);
            for (size_t k = 0; k < g_taps.size(); ++k) {
                const int32_t id = g_taps[k];
                const float * src = nullptr;
                if (id < HEAD_END) {
                    src = llama_get_embeddings_layer_inp(c_head, (uint32_t) id);
                } else {
                    for (uint32_t blk = 0; blk < n_blocks && src == nullptr; ++blk) {
                        uint32_t      blk_id = 0, blk_rows = 0, blk_embd = 0;
                        const float * q = llama_split_take_layer_inp(c_head, blk, &blk_id, &blk_rows, &blk_embd);
                        if (q != nullptr && (int32_t) blk_id == id && blk_rows == 1 &&
                            blk_embd == (uint32_t) n_embd) {
                            src = q;
                        }
                    }
                }
                if (src == nullptr) {
                    printf("[FAIL] remote chain generation step %d: tap %zu (layer %d) is missing\n",
                           st, k, id);
                    ok = false;
                    break;
                }
                std::memcpy(r.gen_taps[st].data() + k * (size_t) n_embd, src,
                            (size_t) n_embd * sizeof(float));
            }
            tok = argmax_of(lg, n_vocab);
            p++;
        }
        // [QLH-SPLIT] v9: the verify-shaped frame, with both channels armed -- per-row ids from the
        // v7 channel and the tap block from this one. This is the reply shape a DFlash round uses on a
        // remote chain, so it gets its own end-to-end check rather than being inferred from the plain
        // frames above.
        llama_split_request_verify_rows(c_head, true);
        gb.n_tokens = kVerifyRows;
        for (int32_t j = 0; j < kVerifyRows; ++j) {
            gb.token[j]     = (llama_token) kVerifyTokens[j];
            gb.pos[j]       = p + j;
            gb.n_seq_id[j]  = 1;
            gb.seq_id[j][0] = 0;
            gb.logits[j]    = 1;
        }
        if (llama_decode(c_head, gb) != 0) {
            printf("[FAIL] remote chain verify-shaped frame\n");
            ok = false;
        } else {
            r.vf_taps.assign(kVerifyRows, std::vector<float>((size_t) g_taps.size() * n_embd, 0.0f));
            r.vf_ids.resize(kVerifyRows);
            const uint32_t n_blocks = llama_split_take_layer_inp_n(c_head);
            for (int32_t j = 0; j < kVerifyRows; ++j) {
                r.vf_ids[j] = (int32_t) llama_split_verified_token(c_head, j);
                if (r.vf_ids[j] < 0) {
                    printf("[FAIL] remote chain verify-shaped frame: row %d answered no id\n", j);
                    ok = false;
                    break;
                }
                for (size_t k = 0; k < g_taps.size(); ++k) {
                    const int32_t id = g_taps[k];
                    const float * src = nullptr;
                    if (id < HEAD_END) {
                        src = llama_get_embeddings_layer_inp(c_head, (uint32_t) id);
                        src = src != nullptr ? src + (size_t) j * n_embd : nullptr;
                    } else {
                        for (uint32_t blk = 0; blk < n_blocks && src == nullptr; ++blk) {
                            uint32_t blk_id = 0, blk_rows = 0, blk_embd = 0;
                            const float * q = llama_split_take_layer_inp(c_head, blk, &blk_id, &blk_rows, &blk_embd);
                            if (q != nullptr && (int32_t) blk_id == id && blk_rows == (uint32_t) kVerifyRows &&
                                blk_embd == (uint32_t) n_embd) {
                                src = q + (size_t) j * n_embd;
                            }
                        }
                    }
                    if (src == nullptr) {
                        printf("[FAIL] remote chain verify-shaped frame: row %d tap %zu (layer %d) missing\n",
                               j, k, id);
                        ok = false;
                        break;
                    }
                    std::memcpy(r.vf_taps[j].data() + k * (size_t) n_embd, src,
                                (size_t) n_embd * sizeof(float));
                }
            }
            printf("[C] verify-shaped frame: %d rows, %u tap blocks\n", kVerifyRows, n_blocks);
        }
        llama_split_request_verify_rows(c_head, false);
        llama_batch_free(gb);
    }
    printf("[C] greedy continuation on the chain: %d tokens\n", (int) r.gen_tokens.size());

    llama_batch_free(b);
    llama_free(c_head);
    llama_model_free(m_head);
    return ok;
}

// 远程模式的比较与报告。返回 true = 全过。
static bool remote_compare(const std::string & path, const char * whole_path,
                           const std::string & endpoint, const std::vector<llama_token> & toks) {
    result whole, chain;
    int32_t n_tok_w = 0;
    int32_t n_embd  = 0;
    if (!whole_load(whole_path, whole, n_tok_w, n_embd)) {
        return false;
    }
    const int32_t n_tok = (int32_t) toks.size();
    if (n_tok != n_tok_w) {
        printf("[FAIL] the dump has %d tokens, this run has %d; the prompt must match\n", n_tok_w, n_tok);
        return false;
    }
    printf("[A] loaded whole-model taps: %d tokens, %d embd, %d vocab, %zu taps\n",
           n_tok_w, n_embd, whole.n_vocab, g_taps.size());

    if (!run_chain_remote(path, endpoint, toks, n_embd, whole.n_vocab, chain)) {
        return false;
    }

    int n_tap_ok = 0;
    int n_wire_ok = 0;
    int n_head_taps = 0;
    for (size_t k = 0; k < g_taps.size(); ++k) {
        const float d = max_abs_diff(whole.taps[k], chain.taps[k]);
        const bool same = d == 0.0f;
        const bool wire = g_taps[k] >= HEAD_END;
        n_tap_ok += same ? 1 : 0;
        n_head_taps += wire ? 0 : 1;
        printf("  tap %zu (source layer %2d, %s): max|chain-whole| = %.9g %s\n",
               k, g_taps[k], wire ? "wire" : "head", d, same ? "逐位相同" : "不同");
        if (wire && same) {
            n_wire_ok++;
        }
    }
    const int n_wire_taps = (int) g_taps.size() - n_head_taps;
    const bool taps_ok = n_tap_ok == (int) g_taps.size();
    printf("[%s] %d 路 tap 全部与整模逐位相同 (%d/%zu；其中 %d 路经 wire)\n",
           taps_ok ? "PASS" : "FAIL", (int) g_taps.size(), n_tap_ok, g_taps.size(), n_wire_taps);
    printf("[%s] wire 上的 %d 路 tap 与整模逐位相同 (%d/%d)\n",
           (n_wire_ok == n_wire_taps) ? "PASS" : "FAIL", n_wire_taps, n_wire_ok, n_wire_taps);

    // 每个「链上能产出 logits」的位置都逐字节比：prefill 的最后一行 + 8 步贪心续写。
    int n_pos_ok = 0;
    int n_pos = 0;
    {
        const bool same = whole.last_row_logits.empty() ? false
                : std::memcmp(chain.last_row_logits.data(), whole.last_row_logits.data(),
                              (size_t) whole.n_vocab * sizeof(float)) == 0;
        // whole.last_row_logits 在落盘里没有单独存一份：它就是 whole.logits 的最后一行。
        const float * wl = whole.logits.data() + (size_t) (n_tok - 1) * whole.n_vocab;
        const bool same2 = chain.last_row_logits.size() == (size_t) whole.n_vocab &&
                std::memcmp(chain.last_row_logits.data(), wl, (size_t) whole.n_vocab * sizeof(float)) == 0;
        (void) same;
        n_pos += 1;
        n_pos_ok += same2 ? 1 : 0;
        printf("  prefill last row (pos %d): %s\n", n_tok - 1, same2 ? "逐位相同" : "不同");
    }
    for (size_t st = 0; st < chain.gen_logits.size() / (size_t) whole.n_vocab; ++st) {
        const float * a = chain.gen_logits.data() + st * (size_t) whole.n_vocab;
        const float * b = whole.gen_logits.data() + st * (size_t) whole.n_vocab;
        const bool same = std::memcmp(a, b, (size_t) whole.n_vocab * sizeof(float)) == 0;
        n_pos += 1;
        n_pos_ok += same ? 1 : 0;
        printf("  gen step %zu (pos %d, token %d): logits %s\n", st, n_tok + (int) st,
               st < chain.gen_tokens.size() ? chain.gen_tokens[st] : -1, same ? "逐位相同" : "不同");
    }
    const bool logits_ok = n_pos_ok == n_pos;
    printf("[%s] 远程链 vs 整模 logits 逐位相同 (%d/%d 个位置)\n",
           logits_ok ? "PASS" : "FAIL", n_pos_ok, n_pos);

    // 续写的 token 序列与整模相同（这是「链上输出 == 整模输出」的直接判据）。
    bool tokens_ok = chain.gen_tokens.size() == whole.gen_tokens.size() && !chain.gen_tokens.empty();
    if (tokens_ok) {
        for (size_t i = 0; i < chain.gen_tokens.size(); ++i) {
            if (chain.gen_tokens[i] != whole.gen_tokens[i]) {
                tokens_ok = false;
                printf("  gen token %zu: chain %d != whole %d\n", i, chain.gen_tokens[i], whole.gen_tokens[i]);
            }
        }
    }
    printf("[%s] 远程链续写 token 与整模逐 token 相同 (%zu/%zu)\n",
           tokens_ok ? "PASS" : "FAIL", chain.gen_tokens.size(), whole.gen_tokens.size());

    // 续写那 8 步的 tap（1 行的帧，最小的形状）也要逐位相同。
    bool gen_taps_ok = chain.gen_taps.size() == whole.gen_taps.size() && !chain.gen_taps.empty();
    if (gen_taps_ok) {
        for (size_t g = 0; g < chain.gen_taps.size() && gen_taps_ok; ++g) {
            const float d = max_abs_diff(whole.gen_taps[g], chain.gen_taps[g]);
            if (d != 0.0f) {
                printf("  gen step %zu taps: max|diff| = %.9g\n", g, d);
                gen_taps_ok = false;
            }
        }
    }
    printf("[%s] 续写每一步的 %zu 路 tap 与整模逐位相同 (%zu 步)\n",
           gen_taps_ok ? "PASS" : "FAIL", g_taps.size(), chain.gen_taps.size());

    // k=2 verify 帧：逐行 id + tap 段（生产里 DFlash 在远程链上真正跑的回复形状）
    bool vf_ok = chain.vf_ids.size() == whole.vf_ids.size() && !chain.vf_ids.empty();
    if (vf_ok) {
        for (size_t j = 0; j < chain.vf_ids.size(); ++j) {
            if (chain.vf_ids[j] != whole.vf_ids[j]) {
                printf("  verify row %zu: chain %d != whole %d\n", j, chain.vf_ids[j], whole.vf_ids[j]);
                vf_ok = false;
            }
        }
    }
    printf("[%s] verify 形状帧（%d 行）逐行 argmax 与整模相同\n",
           vf_ok ? "PASS" : "FAIL", kVerifyRows);
    bool vf_taps_ok = chain.vf_taps.size() == whole.vf_taps.size() && !chain.vf_taps.empty();
    if (vf_taps_ok) {
        for (size_t j = 0; j < chain.vf_taps.size() && vf_taps_ok; ++j) {
            const float d = max_abs_diff(whole.vf_taps[j], chain.vf_taps[j]);
            if (d != 0.0f) {
                printf("  verify row %zu taps: max|diff| = %.9g\n", j, d);
                vf_taps_ok = false;
            }
        }
    }
    printf("[%s] verify 形状帧的 %d 行 x %zu 路 tap 与整模逐位相同\n",
           vf_taps_ok ? "PASS" : "FAIL", kVerifyRows, g_taps.size());

    // [QLH-SPLIT] v9: the transferred bytes, as a hard check and not a report. A tap row is
    // n_embd * 4 bytes and a frame carries *every* row (the draft injects every position of the
    // batch), so the expected section is n_tok * n_wire_taps * n_embd * 4 plus n_wire_taps ids.
    bool bytes_ok = true;
    {
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        const size_t expect    = (size_t) n_tok * n_wire_taps * row_bytes;
        size_t       got       = 0;
        for (size_t k = 0; k < g_taps.size(); ++k) {
            if (g_taps[k] < HEAD_END) {
                continue;
            }
            got += chain.taps[k].size() * sizeof(float);
        }
        bytes_ok = got == expect;
        printf("[%s] tap 传输量：%d 行 x %d 路 x %zu B = %zu B，实测 %zu B（另加 %zu B 的 id 表）\n",
               bytes_ok ? "PASS" : "FAIL", n_tok, n_wire_taps, row_bytes, expect, got,
               (size_t) n_wire_taps * sizeof(int32_t));
        printf("       对照：一行 logits 是 %zu B；k=2 的 verify 帧是 3 行 x %d 路 = %zu B\n",
               (size_t) whole.n_vocab * sizeof(float), n_wire_taps,
               (size_t) 3 * n_wire_taps * row_bytes + (size_t) n_wire_taps * sizeof(int32_t));
        printf("[INFO] prefill 一个 ubatch(512 行) 的 tap 是 %zu B；生成阶段每 token 只有 1 行的 tap\n",
               (size_t) 512 * n_wire_taps * row_bytes);
    }

    const bool pass = taps_ok && logits_ok && n_wire_ok == n_wire_taps && bytes_ok &&
                      tokens_ok && gen_taps_ok && vf_ok && vf_taps_ok;
    printf("\n%s\n", pass ? "ALL PASS" : "FAILED");
    return pass;
}


int main(int argc, char ** argv) {
    const char * model_path = default_model_path(argc, argv);
    if (model_path == nullptr) {
        printf("[SKIP] dflash-taps: no model path given; pass a GGUF as argv[1] or set QLH_SPLIT_MODEL\n");
        return 0;
    }

    // [QLH-SPLIT] v9: the three modes. Nothing after argv[1] keeps the local-chain behaviour the
    // acceptance script already greps for ("ALL PASS", "tail local 1 == whole 33").
    const char * whole_save_path  = nullptr;
    const char * whole_load_path  = nullptr;
    const char * draft_path       = nullptr;
    const char * endpoint         = nullptr;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                printf("[FAIL] %s needs a value\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--whole-save") {
            if ((whole_save_path = next("--whole-save")) == nullptr) { return 1; }
        } else if (a == "--remote-compare") {
            if ((whole_load_path = next("--remote-compare")) == nullptr) { return 1; }
        } else if (a == "--draft") {
            if ((draft_path = next("--draft")) == nullptr) { return 1; }
        } else if (a == "--endpoint") {
            if ((endpoint = next("--endpoint")) == nullptr) { return 1; }
        } else {
            printf("[FAIL] unknown argument '%s'\n", a.c_str());
            return 1;
        }
    }
    if (whole_load_path != nullptr && endpoint == nullptr) {
        printf("[FAIL] --remote-compare needs --endpoint host:port\n");
        return 1;
    }

    llama_backend_init();
    llama_log_set(silent_log, nullptr);
    // Unbuffered, so a crash still leaves every line it printed behind (a segfault lost the whole
    // buffered log once, which is exactly when you need it).
    setvbuf(stdout, nullptr, _IONBF, 0);

    // [QLH-SPLIT] v9: the tap list comes from the draft sidecar when one is given, so DFlash (8 taps)
    // and DSpark (5) use one command and the ids are the ones the implementation actually reads.
    if (draft_path != nullptr) {
        // A plain load, not vocab_only: the layer-input list lives in the arch hparams, and a
        // vocab_only load never reaches them (src/llama-model.cpp, "everything past this point is not
        // vocab-related"). The default load is mmap'ed, so the weights are mapped and never read --
        // only the metadata is touched, which is all this needs.
        llama_model_params dp = llama_model_default_params();
        llama_model * dm = llama_model_load_from_file(draft_path, dp);
        if (dm == nullptr) {
            printf("[FAIL] draft model load: %s\n", draft_path);
            return 1;
        }
        const uint32_t nt = llama_model_target_layer_ids_n(dm);
        const int32_t * ids = llama_model_target_layer_ids(dm);
        if (nt == 0 || ids == nullptr) {
            printf("[FAIL] %s names no dflash.target_layers\n", draft_path);
            llama_model_free(dm);
            return 1;
        }
        g_taps.assign(ids, ids + nt);
        llama_model_free(dm);
        printf("taps from %s: %u\n", draft_path, nt);
        for (size_t k = 0; k < g_taps.size(); ++k) {
            printf("    tap %zu: source layer %d%s\n", k, g_taps[k],
                   g_taps[k] >= HEAD_END ? " (remote)" : "");
        }
    }

    llama_model_params vp = llama_model_default_params();
    vp.vocab_only = true;
    llama_model * vmodel = llama_model_load_from_file(model_path, vp);
    if (vmodel == nullptr) {
        printf("[FAIL] vocab load\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vmodel);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // 代码风格的 prompt，约 20 token —— 必须远小于 n_ubatch，保证单 ubatch
    const std::vector<llama_token> toks =
        tokenize(vocab, "Write a Python function that returns the nth Fibonacci number, using memoization.");
    const int32_t n_tok = (int32_t) toks.size();
    printf("prompt tokens = %d\n", n_tok);

    int32_t n_embd = 0; // 由 run_whole 从完整模型上取（vocab_only 的模型读不出 n_embd）
    printf("n_vocab = %d\n", n_vocab);
    if (n_tok < 3 || n_tok > 256) {
        printf("[FAIL] prompt size out of the single-ubatch range\n");
        return 1;
    }
    llama_model_free(vmodel);

    // [QLH-SPLIT] v9: 远程链的比较。整模那半是另一次运行写下来的，因为两者的峰值内存放不进这台机器。
    if (whole_load_path != nullptr) {
        const bool ok = remote_compare(model_path, whole_load_path, endpoint, toks);
        llama_backend_free();
        return ok ? 0 : 1;
    }

    // [QLH-SPLIT] v9: 整模那半，落盘给远程模式用。
    if (whole_save_path != nullptr) {
        result whole_only;
        // [QLH-SPLIT] v9: the dump is compared against a *remote* chain, which can only produce the
        // last row's logits, so the whole model is asked for the same shape.
        if (!run_whole(model_path, toks, n_embd, n_vocab, whole_only, /*last_row_only*/ true)) {
            llama_backend_free();
            return 1;
        }
        printf("n_embd = %d\n", n_embd);
        const bool ok = whole_save(whole_save_path, whole_only, n_tok, n_embd);
        llama_backend_free();
        return ok ? 0 : 1;
    }

    result whole, chain;

    // 先跑整模并释放（23 GB 的模型不重复占内存），再建链
    if (!run_whole(model_path, toks, n_embd, n_vocab, whole)) {
        return 1;
    }
    printf("n_embd = %d\n", n_embd);
    if (!run_chain(model_path, toks, n_embd, n_vocab, chain)) {
        return 1;
    }

    int n_tap_ok = 0;
    for (size_t k = 0; k < g_taps.size(); ++k) {
        const float d = max_abs_diff(whole.taps[k], chain.taps[k]);
        const bool same = d == 0.0f;
        printf("  tap %zu (source layer %2d): max|chain-whole| = %.9g %s\n",
               k, g_taps[k], d, same ? "逐位相同" : "不同");
        n_tap_ok += same ? 1 : 0;
    }
    printf("[%s] tap 拼接整模 vs 链逐位相同 (%d/%u)\n",
           n_tap_ok == (int) g_taps.size() ? "PASS" : "FAIL", n_tap_ok, (unsigned) g_taps.size());

    // 单独点名两个跨段 tap：这是门 3 的核心
    {
        const int k33 = 6, k38 = 7;
        const float d33 = max_abs_diff(whole.taps[k33], chain.taps[k33]);
        const float d38 = max_abs_diff(whole.taps[k38], chain.taps[k38]);
        printf("[%s] tail local 1 == whole 33 (max|diff|=%.9g), tail local 6 == whole 38 (max|diff|=%.9g)\n",
               (d33 == 0.0f && d38 == 0.0f) ? "PASS" : "FAIL", d33, d38);
    }

    // logits：链上每一行 vs 整模每一行
    bool logits_ok = false;
    {
        int n_ok = 0;
        float max_d = 0.0f;
        for (int32_t i = 0; i < n_tok; ++i) {
            const float * a = chain.logits.data() + (size_t) i * n_vocab;
            const float * b = whole.logits.data() + (size_t) i * n_vocab;
            const bool same = memcmp(a, b, (size_t) n_vocab * sizeof(float)) == 0;
            if (!same) {
                float md = 0.0f;
                for (int32_t j = 0; j < n_vocab; ++j) {
                    const float d = std::fabs(a[j] - b[j]);
                    if (d > md) {
                        md = d;
                    }
                }
                if (md > max_d) {
                    max_d = md;
                }
            }
            n_ok += same ? 1 : 0;
        }
        logits_ok = n_ok == n_tok;
        printf("[%s] 链 vs 整模 logits 逐位相同 (%d/%d 行)\n",
               logits_ok ? "PASS" : "FAIL", n_ok, n_tok);
        if (!logits_ok) {
            printf("     max|diff| = %.9g（检查是否真的单 ubatch；跨 ubatch 是已知浮点差异）\n", max_d);
        }
    }

    const bool pass = n_tap_ok == (int) g_taps.size() && logits_ok;
    printf("\n%s\n", pass ? "ALL PASS" : "FAILED");
    llama_backend_free();
    return pass ? 0 : 1;
}
