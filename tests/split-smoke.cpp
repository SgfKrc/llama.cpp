// N=2 分段实测：两个 model（0:20 与 20:24）、两个 context、llama_split_attach，看
// 拼接后的 logits 是否与整模单 context 一致。
//
// 这是「分段 context」的第一个真正的端到端证据。如果 argmax 不一致，说明 hidden 的
// 传递或位置语义有偏差，后面的跨进程没有意义。
#include "llama.h"
#include "../src/llama-ext.h"   // relay 通道：llama_set_embeddings_nextn / get_embeddings_nextn_ith
#include "../src/llama-split-net.h"    // [H] 假 stage 用的 loopback 监听器
#include "../src/llama-split-proto.h"  // [H] 假 stage 要回 v4 的 OP_HELLO

#include <cmath>
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

// 跑一批 token，返回最后一个输出行的 logits 的 argmax 与前 8 个值
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

    // vocab 从一个 vocab_only 的加载里拿（不占多少内存）
    llama_model_params vp = llama_model_default_params();
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

    // ONLY_SPLIT=1 跳过 baseline/[A]/[B]/[C]，只跑链路：用来验证「前面的实验是否污染了
    // 链路的 embedding 缓冲」。
    const bool only_split = getenv("ONLY_SPLIT") != nullptr;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = 256;
    cp.n_batch = 256;
    // n_ubatch 默认 512，比我们设的 n_batch 还大；shim 也是把它压到 n_batch 以内的。
    cp.n_ubatch = 256;
    // recurrent 层默认不能部分回滚（0 = no rollback），seq_rm 会直接失败。
    // 只在需要裁剪的段上开；n_ubatch 必须装得下快照深度。
    cp.n_rs_seq = 8;

    // --- 基线：整模单 context ---
    int base_argmax = -1;
    float base8[8] = {0};
    // 三段链要逐位对照，所以基线不能只留 argmax 和前 8 个值。
    std::vector<float> base_logits;
    if (!only_split) {
        llama_model_params mp = llama_model_default_params();
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

    // --- 实验 B：整模在层 20 处的输入 hidden（作为 head 段输出的对照）---
    std::vector<float> layer20_input;
    if (!only_split) {
        llama_model_params mp = llama_model_default_params();
        llama_model * m = llama_model_load_from_file(model_path(), mp);
        llama_context_params cpb = cp;
        cpb.embeddings = true;
        cpb.embeddings_pre_norm = true;
        cpb.n_outputs_max = 0;
        cpb.n_outputs_max_per_seq = 0;
        llama_context * c = llama_init_from_model(m, cpb);
        llama_set_embeddings_layer_inp(c, 20, true);   // 要第 20 层的输入 = 第 19 层的输出

        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
        if (llama_decode(c, b) == 0) {
            // 用专用出口，不要再借 get_embeddings_ith —— 那取的是 res->t_embd，
            // 在 pre_norm 打开时是层 23 的输出，不是层 20 的输入。
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

    // --- 实验 A：只有 head 段，一个 context ---
    // 用来区分「embeddings_nextn 的图本身有问题」和「两个 context 共享 backend 冲突」。
    if (!only_split) {
        llama_model_params p1 = llama_model_default_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 20;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);
        if (m1 == nullptr) { printf("A: model load failed\n"); return 1; }

        llama_context_params cp_head = cp;
        cp_head.embeddings          = true;
        cp_head.embeddings_pre_norm = true;
        cp_head.n_outputs_max         = 0;
        cp_head.n_outputs_max_per_seq = 0;

        fprintf(stderr, "[A] creating single head context\n");
        llama_context * c1 = llama_init_from_model(m1, cp_head);
        fprintf(stderr, "[A] created = %p\n", (void *) c1);
        fprintf(stderr, "[A] decoding\n");

        llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
        const int rc = llama_decode(c1, b);
        fprintf(stderr, "[A] decode rc = %d\n", rc);
        if (rc == 0) {
            const float * h = llama_get_embeddings_ith(c1, 0);
            printf("[A] head(0:20) output hidden[0..7] = ");
            if (h) { for (int i = 0; i < 8; ++i) printf("%.5f ", h[i]); }
            printf("\n");
            // 回归口径是「[B] 与 [A] 的 hidden 逐位相同」：只比前 8 个值会漏掉别处的偏差。
            if (h && layer20_input.size() == 2048) {
                printf("[A] vs [B] hidden 2048 维%s\n",
                       memcmp(h, layer20_input.data(), 2048 * sizeof(float)) == 0
                       ? "逐位相同" : "不同");
            }
        }
        llama_free(c1);
        llama_model_free(m1);
    }

    // --- 实验 C：tail 段单独跑，喂 [B] 的正确 hidden ---
    if (!only_split && !layer20_input.empty()) {
        llama_model_params p2 = llama_model_default_params();
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

    // --- 分段：0:20 + 20:24，两个 context 串起来 ---
    int split_argmax = -1;
    float split8[8] = {0};
    {
        llama_model_params p1 = llama_model_default_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 20;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_model_params p2 = llama_model_default_params();
        p2.split_layer_lo = 20;
        p2.split_layer_hi = 24;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);

        printf("head model layers=%d  tail model layers=%d\n",
               m1 ? llama_model_n_layer(m1) : -1, m2 ? llama_model_n_layer(m2) : -1);

        if (m1 == nullptr || m2 == nullptr) {
            printf("split model load failed\n");
            return 1;
        }

        // head 段要在构造时声明「要 embedding 输出、且取 pre-norm」：这两个都是改变图输出
        // 形状的 params 字段，运行时改不动 buffer 规划。shim 里也是成对设置的。
        llama_context_params cp_head = cp;
        cp_head.embeddings          = true;
        cp_head.embeddings_pre_norm = true;
        // 输出行数从 n_batch 推，而不是用默认上限：embeddings 输出缓冲的大小由它算出来，
        // 留默认值会让图要一个比分配到的更大的输出缓冲。shim 里也是这么设的。
        cp_head.n_outputs_max         = 0;  // derive from n_batch
        // 默认是 1：每 seq 只留一行输出。embedding 输出要求每个 token 一行，留着默认值会让
        // 图申请的 embedding 缓冲比分配到的多。shim 里同样把它置 0。
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
            // 跨进程模式：tail 在 split-stage 进程里，本地不建 c2。
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
            // 失败就是失败：以前 attach 是 void，链拼错也照样往下跑，只会在 logits 上偏。
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

    // --- 实验 H：远端 stage 失败之后，链被标记为不可信，但进程不 abort ---
    // 这一组是「一次网络抖动 = 整个 server 死」那条实测问题的回归测试。现场形状是：stage 挂在
    // recv 上 120 s，seq_rm 失败，common.cpp 直接 GGML_ABORT。这里把它拆成可确定复现的三件事：
    //   1. 远端 seq_rm 失败 → 返回 false，链被标记（llama_context_split_failed() == 1），
    //      并且原因里带着是哪个 stage、哪个 endpoint、哪个范围；
    //   2. 之后的 decode 必须被拒（-3），绝不能拿着已经错开的 KV 算 token；
    //   3. 再次 seq_rm 不再碰网络（立即失败），否则每个后续请求都要再等一次 120 s。
    // 用「accept 一次就立刻关闭」的假 stage 代替挂起真 stage：attach 没有握手，所以这就是
    // 「链刚建好，对端就消失」，不用等 120 s 超时，测试是确定的。
    {
        // 整模的对照：没有远端 stage，就永远没有「链不可信」这回事
        {
            llama_model_params pm = llama_model_default_params();
            llama_model * m = llama_model_load_from_file(model_path(), pm);
            llama_context * c = m != nullptr ? llama_init_from_model(m, cp) : nullptr;
            if (c != nullptr) {
                printf("[H] whole model: split_failed=%d failure='%s' (expect 0 and empty)\n",
                       (int) llama_context_split_failed(c), llama_context_split_failure(c));
                llama_free(c);
            }
            if (m != nullptr) llama_model_free(m);
        }

        // 反例输入：整模上让 seq_rm 真的失败，看它落在哪条分支。hybrid 模型在 n_rs_seq=0 时
        // 不允许部分回滚（llama-memory-recurrent.cpp 的 rollback <= n_rs_seq 检查），所以
        // 「decode 3 个 token 再 seq_rm(0, 2, -1)」就是一个必然失败的整模 seq_rm。它返回 false
        // 而 split_failed 仍是 0 —— common_context_seq_rm 里那个 if 因此取原来的 GGML_ABORT
        // 分支，非链场景一个字节都没变。真跑下去会杀掉本进程，所以这里只证明输入与判定。
        {
            llama_context_params cp0 = cp;
            cp0.n_rs_seq = 0; // 不要回滚快照
            llama_model_params pm = llama_model_default_params();
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
            addr.sin_port        = 0; // 让系统挑端口，测试之间不会抢端口
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
            llama_model_params p1 = llama_model_default_params();
            p1.split_layer_lo = 0;
            p1.split_layer_hi = 20;
            llama_model * m1 = llama_model_load_from_file(model_path(), p1);

            llama_context_params cph = cp;
            cph.embeddings            = true;
            cph.embeddings_pre_norm   = true;
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
                // [QLH-SPLIT] v4：attach 成功之前必须先过一次 OP_HELLO。这个假 stage 回一个
                // 「就是 head 期望的那一段」的身份，让测试测到的仍然是后面那次「stage 半路
                // 掉线」，而不是 attach 被握手拒掉。回显期望值是最简写法：这一段本来就该是
                // head 期望的 tail。
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
                        self.layer_hi    = expect.layer_hi; // head 期望的 tail，上界已知
                        self.n_layer_src = expect.n_layer_src;
                        self.n_embd_inp  = expect.n_embd_inp;
                        self.n_embd_out  = expect.n_embd_out;
                        self.n_vocab     = expect.n_vocab;
                        self.role        = qlh_split::ROLE_TAIL;
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

                // 必须拒绝，而不是拿着错开的 KV 继续算
                const std::vector<llama_token> one = { toks[0] };
                llama_batch b = llama_batch_get_one(const_cast<llama_token *>(one.data()), 1);
                const int rc = llama_decode(c1, b);
                printf("[H] decode after the chain died -> %d (expect -3)\n", rc);

                // 再 trim 一次：必须不走网络，否则每个后续请求都要再等 120 s
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
        printf("\n=== baseline 被跳过，split argmax=%d ===\n", split_argmax);
        return 0;
    }
    // --- 实验 D：seq_rm 广播 ---
    // 裁掉 pos>=2 的部分，看 tail 段自己的 memory 是否也少了。只验证「聚合视图返回什么」
    // 不够——错位正是出现在「一段裁了另一段没裁」的时候。
    {
        llama_model_params p1 = llama_model_default_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 20;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_model_params p2 = llama_model_default_params();
        p2.split_layer_lo = 20;
        p2.split_layer_hi = 24;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);

        llama_context_params cph = cp;
        cph.embeddings = true;
        cph.embeddings_pre_norm = true;
        cph.n_outputs_max = 0;
        cph.n_outputs_max_per_seq = 0;

        llama_context * c1 = llama_init_from_model(m1, cph);
        llama_context * c2 = llama_init_from_model(m2, cp);
        llama_context * stages[1] = { c2 };
        const int attached = llama_split_attach(c1, stages, 1);

        // 用多 token 的输入，否则 pos 只有 0，裁 1 以上没有可裁的
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
            printf("[D] %s\n", (h1 < h0 && t1 < t0) ? "广播生效：两段都裁了"
                   : (t1 == t0 ? "广播失败：tail 没裁" : "异常"));
        }
        llama_free(c1);
        llama_free(c2);
        llama_model_free(m1);
        llama_model_free(m2);
    }

    // --- 实验 E：三段链 0:8 + 8:16 + 16:24 ---
    // 前面的 [A]/[B] 只证明单段，两段只证明一次接力。三段要求两个 relay 段都导出 pre-norm
    // hidden，并且 tail 段仍然拿到正确的输入。逐位比较 logits：argmax 相同太容易蒙对。
    {
        const int ranges[3][2] = {{0, 8}, {8, 16}, {16, 24}};
        llama_model   * models[3] = {nullptr, nullptr, nullptr};
        llama_context * ctxs[3]   = {nullptr, nullptr, nullptr};
        bool ok = true;

        for (int s = 0; s < 3 && ok; ++s) {
            llama_model_params p = llama_model_default_params();
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
                // head 与中间 relay 段：要 pre-norm hidden；输出行数从 n_batch 推。
                cps.embeddings = true;
                cps.embeddings_pre_norm = true;
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
                            printf("[E] 首个不同: logits[%d] 链=%.9g baseline=%.9g\n",
                                   i, lg[i], base_logits[i]);
                            bit_equal = false;
                            break;
                        }
                    }
                }
                printf("[E] %s\n", bit_equal ? "三段链与整模 logits 逐位相同"
                       : "三段链与整模 logits 不同");
            }
        }

        // 先放 head，再放段：head 的 split_memory 持有段的 memory，反过来会留下悬空视图。
        llama_free(ctxs[0]);
        llama_free(ctxs[1]);
        llama_free(ctxs[2]);
        for (int s = 0; s < 3; ++s) {
            llama_model_free(models[s]);
        }
    }

    // --- 实验 F：反例——不衔接的层范围必须被 attach 拒绝 ---
    // 0:8 的 head 后接 12:16（中间缺 8:12）；0:8 + 8:16（没盖到模型末尾 24）；以及 8:20 的 head
    // 后接 20:24（tile 了 8..24，衔接连续、也到末层，但前 8 层从没被算过）。三种以前都被静默
    // 接受，然后 decoded 出「看着合理」的垃圾。校验失败必须返回 -1 并保持单段。
    {
        llama_model_params p1 = llama_model_default_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 8;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_model_params p2 = llama_model_default_params();
        p2.split_layer_lo = 12;
        p2.split_layer_hi = 16;
        llama_model * m2 = llama_model_load_from_file(model_path(), p2);

        llama_model_params p3 = llama_model_default_params();
        p3.split_layer_lo = 8;
        p3.split_layer_hi = 16;
        llama_model * m3 = llama_model_load_from_file(model_path(), p3);

        // head 从第 8 层开始：接上 20:24 后 tile 8..24 是连续的、也覆盖到末层，唯一的毛病是
        // 0..7 从来没人算。这是 split_attach 的「head 必须以第 0 层开始」那条不变量的反例。
        llama_model_params p4 = llama_model_default_params();
        p4.split_layer_lo = 8;
        p4.split_layer_hi = 20;
        llama_model * m4 = llama_model_load_from_file(model_path(), p4);

        llama_model_params p5 = llama_model_default_params();
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

    // --- 实验 G / G2 的共同入口：远程链的 endpoint 列表 ---
    // 用 QLH_SPLIT_REMOTE2="host:port,host:port" 触发，未设置时 G/G2 整段跳过。
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

    // --- 实验 G：三段跨进程（本地 head 0:8 + 远程 8:16 + 远程 16:24）---
    // 远程 relay 段以前从来没工作过（角色判定把 relay 跑成 tail，客户端拿到 logits 却当
    // hidden 用，静默算错），所以第一次端到端必须拿强证据：整个 vocab 逐位对比，argmax 相同
    // 太容易蒙对。
    if (!remote_eps.empty() && !base_logits.empty()) {
        const std::vector<std::string> & eps = remote_eps;

        llama_model_params p1 = llama_model_default_params();
        p1.split_layer_lo = 0;
        p1.split_layer_hi = 8;
        llama_model * m1 = llama_model_load_from_file(model_path(), p1);

        llama_context_params cph = cp;
        // head 要把 pre-norm hidden 交给第一段远程 relay，和本地链是同一个约定。
        cph.embeddings          = true;
        cph.embeddings_pre_norm = true;
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
                printf("[G] argmax=%d (baseline %d), logits 全 %d 维: %s\n",
                       g_argmax, base_argmax, nv, bit_equal ? "逐位相同" : "不同");
                if (first_diff >= 0) {
                    printf("[G] 首个不同: logits[%d] 链=%.9g baseline=%.9g\n",
                           first_diff, lg[first_diff], base_logits[first_diff]);
                }

                // [QLH-SPLIT] 远程 KV 真的被裁：这条路径以前不存在（seq_rm 到不了远程段）。
                // 用一个多 token 的输入裁到 p0=2，stage 侧必须打出 OP_SEQ_RM seq=0 p0=2 p1=-1
                // 且 pos_max 从 3 变成 1；head 侧的 pos_max 是同一个数字。
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

    // --- 实验 G2：远程链 checkpoint 往返（LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY）---
    // 服务端的前缀复用靠 checkpoint，而 checkpoint 的 blob 里只有 recurrent state（qwen35 没有
    // SWA，PARTIAL_ONLY 下 KV 部分一行都不写）。这里验证 save → 再 decode → restore → 裁掉多出来
    // 的后缀 → 重新 decode 同一段：logits 必须与第一次逐位相同，并且与整模走同一流程的结果逐位
    // 相同。两个方向都要验，否则「恢复出来一个自洽但错的链」也能过。
    if (!remote_eps.empty()) {
        std::vector<llama_token> seq = tokenize(vocab,
                "Name the capital of France. Name the capital of Japan. "
                "Name the capital of Peru. Name the capital of Norway. ");
        if (seq.size() < 40) {
            seq.resize(40, 1000);
        }
        const int n_tail = 20;
        const int n0     = (int) seq.size() - n_tail;

        // llama_batch_get_one() 会把位置重新从 0 编号，而「save 之后再 decode」必须接在已有
        // 位置后面；用显式位置的 batch，否则两次 decode 会互相覆盖 cells，测到的就不是 checkpoint。
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

        // 返回 restore 之后重新 decode 的 logits；失败返回 false。
        const auto roundtrip = [&](llama_context * c, const char * tag, std::vector<float> & out) -> bool {
            const int nv = (int) llama_vocab_n_tokens(vocab);

            // 同一对 split-stage 进程刚被 [G] 用过，seq 0 上还留着位置；不清掉的话第一个
            // decode 会因为「batch 起始位置 < 已有位置」被 stage 拒绝（M-RoPE 的位置校验）。
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
                   tag, sz, sz / 1048576.0, same ? "逐位相同" : "不同",
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

        { // 整模基线
            llama_model_params pm = llama_model_default_params();
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

        // 本地三段链：同一套往返。远程链的 state 路径是在 aggregate memory 里加的，本地链走的
        // 还是原来那条路，这一步就是「没有把本地链弄坏」的证据（而且比 [E] 强：20 个 token，
        // 位置 0 以外也参与）。
        std::vector<float> logits_local;
        bool ok_local = false;
        {
            const int ranges[3][2] = {{0, 8}, {8, 16}, {16, 24}};
            llama_model   * ms[3] = {nullptr, nullptr, nullptr};
            llama_context * cs[3] = {nullptr, nullptr, nullptr};
            bool ok = true;
            for (int st = 0; st < 3 && ok; ++st) {
                llama_model_params p = llama_model_default_params();
                p.split_layer_lo = ranges[st][0];
                p.split_layer_hi = ranges[st][1];
                ms[st] = llama_model_load_from_file(model_path(), p);
                if (ms[st] == nullptr) { ok = false; break; }
                llama_context_params cps = cp;
                if (st < 2) {
                    cps.embeddings            = true;
                    cps.embeddings_pre_norm   = true;
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
            // head 先放：它的 split_memory 持有段的 memory
            if (cs[0] != nullptr) llama_free(cs[0]);
            if (cs[1] != nullptr) llama_free(cs[1]);
            if (cs[2] != nullptr) llama_free(cs[2]);
            for (int st = 0; st < 3; ++st) {
                if (ms[st] != nullptr) llama_model_free(ms[st]);
            }
        }

        { // 远程链
            llama_model_params p1 = llama_model_default_params();
            p1.split_layer_lo = 0;
            p1.split_layer_hi = 8;
            llama_model * m1 = llama_model_load_from_file(model_path(), p1);

            llama_context_params cph = cp;
            cph.embeddings            = true;
            cph.embeddings_pre_norm   = true;
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

                    // 反例：save 之后把整条链清空（远端 pos_max -> -1），再恢复必须失败。这是
                    // 「stage 被清掉/被重启」那条静默分叉的唯一防线：stage 侧发现自己的位置已经
                    // 落后于快照，在**应用之前**拒绝，head 因此也没有被恢复。
                    llama_memory_t mem = llama_get_memory(c1);
                    llama_batch b0 = make_batch(head_toks, 0);
                    llama_decode(c1, b0);
                    llama_batch_free(b0);

                    const size_t sz = llama_state_seq_get_size_ext(c1, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    std::vector<uint8_t> blob(sz);
                    const size_t got = sz > 0 ? llama_state_seq_get_data_ext(c1, blob.data(), sz, 0,
                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : 0;
                    llama_memory_seq_rm(mem, 0, 0, -1); // 整条链清空
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
                   same ? "逐位相同" : "不同", logits_whole.size(),
                   first_diff >= 0 ? (" first diff at " + std::to_string(first_diff)).c_str() : "");
        };

        compare("local chain", logits_local, ok_local);
        compare("remote chain", logits_chain, ok_chain);
        if (!ok_whole || !ok_local || !ok_chain) {
            printf("[G2] roundtrip incomplete (whole=%d local=%d remote=%d)\n",
                   (int) ok_whole, (int) ok_local, (int) ok_chain);
        }
    }

    printf("\n=== %s ===\n", base_argmax == split_argmax ? "ARGMAX 一致" : "ARGMAX 不一致");
    return base_argmax == split_argmax ? 0 : 2;
}
