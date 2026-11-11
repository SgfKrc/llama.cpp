#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-split.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"
#include "slim-arc-prefetch.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cstddef> // [QLH-SPLIT] offsetof, used by the candidate layout assertions
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_flash_attn_kv_rows_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    /*.name             =*/ "Flash Attention over K/V rows",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && (!llm_arch_supports_rs_rollback(model.arch) || !hparams.causal_attn)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    cparams.embeddings_layer_inp_dev = false;
    cparams.embeddings_nextn_dev     = false;
    cparams.ctx_state_src            = nullptr;
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        } else if (model.arch == LLM_ARCH_DFLASH) {
            // used to read the target's layer input device buffer
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    // the KV cache is shared with ctx_other, so follow its flash attention setting instead of resolving it here:
    // the probe disables it when the layers and the shared KV are on different devices, which breaks a quantized V cache
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        const bool flash_attn_other = cparams.ctx_other->get_cparams().flash_attn;
        if (!cparams.auto_fa && cparams.flash_attn != flash_attn_other) {
            LLAMA_LOG_WARN("%s: flash_attn = %d does not match the context the KV cache is shared with, using %d\n",
                    __func__, cparams.flash_attn, flash_attn_other);
        }
        cparams.flash_attn = flash_attn_other;
        cparams.auto_fa    = false;
    }

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size      = params.moe_cache_size;
    cparams.moe_cache_cpu_share = params.moe_cache_cpu_share;

    // resolved after flash attention, see resolve_fused_ops()
    {
        cparams.fused_kv_rows = false;

        const char * LLAMA_KV_ROWS_DISABLE = getenv("LLAMA_KV_ROWS_DISABLE");
        if (!LLAMA_KV_ROWS_DISABLE) {
            cparams.auto_fkvr = cparams.kv_unified;
        } else {
            LLAMA_LOG_INFO("%s: LLAMA_KV_ROWS_DISABLE set, disabling kv_rows\n", __func__);
            cparams.auto_fkvr = false;
        }
    }
    cparams.embeddings_pre_norm = params.embeddings_pre_norm;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: embeddings_pre_norm   = %s\n",   __func__, cparams.embeddings_pre_norm ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    {
        const int64_t max_nodes = graph_max_nodes();

        gf_res_reserve = std::make_unique<llm_graph_result>(max_nodes);
        for (auto & res : gf_res_prev) {
            res = std::make_unique<llm_graph_result>(max_nodes);
        }
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // tensors of ctx_other (e.g. a shared KV cache) can be on devices that this model does not use
        if (cparams.ctx_other) {
            for (const auto & dev_other : llama_get_model(cparams.ctx_other)->devices) {
                const bool found = std::any_of(model.devices.begin(), model.devices.end(),
                        [&](const llama_device & dev) { return dev.dev == dev_other.dev; });
                if (found) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(dev_other.dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev_other.dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
            /*.no_alloc  =*/ model.no_alloc(),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, model.no_alloc(), backend_ptrs, backend_buft, cparams.moe_cache_size, cparams.moe_cache_cpu_share);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.no_alloc() && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    // the fallback attends to the whole unified cache with a dense mask
    if (cparams.auto_fkvr) {
        cparams.fused_kv_rows = cparams.flash_attn;
        resolve(llm_fused_op_flash_attn_kv_rows_probe, cparams.fused_kv_rows);
        cparams.auto_fkvr = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_new(const sched_params & params) {
    // reusing the scheduler avoids reallocating its buffers (incl. the compute buffers)
    if (sched && params.can_reuse(sched_params_cur)) {
        LLAMA_LOG_DEBUG("%s: reusing scheduler: max_nodes = %zu (current max_nodes = %zu)\n",
                __func__, params.max_nodes, sched_params_cur.max_nodes);
        return;
    }

    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), params.max_nodes, params.parallel, params.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);

    sched_params_cur = params;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes();

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev_active = nullptr;

    gf_res_reserve->set_max_nodes(max_nodes);
    gf_res_reserve->reset();
    for (auto & res : gf_res_prev) {
        res->set_max_nodes(max_nodes);
        res->reset();
    }

    sched_new({
        /*.max_nodes  =*/ max_nodes,
        /*.parallel   =*/ cparams.pipeline_parallel,
        /*.op_offload =*/ cparams.op_offload,
    });

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.no_alloc(), model.no_alloc() ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new({
                    /*.max_nodes  =*/ max_nodes,
                    /*.parallel   =*/ false,
                    /*.op_offload =*/ cparams.op_offload,
                });
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.no_alloc());
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.no_alloc());
                break;
            default:
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.no_alloc());
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.no_alloc()) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    // [QLH-SPLIT] With stages attached, callers must see the whole chain. split_memory wraps
    // head + tails and is what makes llama-server's seq_rm/pos_min/pos_max cover every stage.
    if (split_memory) {
        return split_memory.get();
    }

    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            res->reset();
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    // [QLH-SPLIT] llama.h defines this as get_logits_ith(-1), and in split mode the logits live
    // in another context (or in split_remote_logits), so the plain buffer here is not the answer.
    // Forward for the same reason get_logits_ith() does.
    if (!split_stages.empty() || !split_remotes.empty()) {
        return get_logits_ith(-1);
    }

    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    // [QLH-SPLIT] The chain's logits live in the last stage. Forwarding instead of copying
    // keeps the buffer owned by the context that actually produced it.
    if (!split_remotes.empty()) {
        // A remote tail returns exactly one row of logits: the last token of the batch it was
        // handed (tools/split-stage/main.cpp, OP_DECODE). The caller indexes logits by batch
        // position, and llama-server samples from the last prompt token -- so remember which
        // position that row is and refuse the others instead of handing back the same row for
        // any index. The local chain behaves the same way: asking it for a token that was not
        // marked as an output is an error, not a repeat.
        //
        // [QLH-SPLIT] v7: a per-row verify frame transfers no logits at all, so every index --
        // including the last one -- answers null here. The rows of such a frame are read through
        // llama_split_verified_token() instead; the sampler picks the path (common_sampler_sample).
        if (split_remote_logits.empty() || split_remote_n_vocab == 0) {
            return nullptr;
        }
        if (i >= 0 && i != split_remote_logits_row) {
            return nullptr;
        }
        return split_remote_logits.data();
    }
    if (!split_stages.empty()) {
        return split_stages.back()->get_logits_ith(i);
    }

    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // [QLH-SPLIT] A stage that has to hand hidden states downstream cannot stop producing
    // them. llama-server toggles this per batch from the slot's need_embd(), which is false
    // for ordinary chat requests -- that toggle is what kept emptying the head stage's
    // embedding buffer. Only the last stage of a chain is allowed to turn it off, and a
    // remote chain needs the same protection as a local one: the head is what fills the
    // carrier that gets sent over the wire.
    if ((!split_stages.empty() || !split_remotes.empty()) && !value) {
        // [QLH-SPLIT] Not silent: the caller asked for something this context cannot do, and the
        // price of ignoring it (an embedding-sized output buffer for the whole batch) is worth one
        // line per process. call_once because llama-server toggles this from its slot loop, once
        // per batch -- a line per call would be a line per token.
        static std::once_flag once;
        std::call_once(once, [] {
            LLAMA_LOG_WARN("llama_context::set_embeddings: ignoring value = false on a context that "
                    "drives a split chain -- the head owes pre-norm hidden to the next stage and "
                    "cannot stop producing it\n");
        });
        return;
    }

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;

    // the device rows are batch rows of the unmasked extraction
    if (cparams.embeddings_nextn_dev && (!value || masked)) {
        cparams.embeddings_nextn_dev = false;
        h_nextn_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;

    if (cparams.embeddings_layer_inp_dev) {
        layer_inp_dev_alloc();
    }
}

void llama_context::set_embeddings_layer_inp_dev(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.embeddings_layer_inp_dev != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings_layer_inp_dev = value;

    layer_inp_dev_alloc();
}

bool llama_context::supports_embd_row() const {
    // the DFlash/DSpark decoder graphs read the device buffer
    if (model.arch != LLM_ARCH_DFLASH) {
        return false;
    }

    const ggml_tensor * src = cparams.ctx_other ? cparams.ctx_other->get_layer_inp_dev() : nullptr;
    if (src == nullptr) {
        return false;
    }

    // [QLH-SPLIT] A row of that buffer is the decoder's *whole* encoder input: one n_embd block per
    // extract layer, in ascending layer id order (llama-graph.cpp, build_layer_inp_dev). ctx_other only
    // fills the layers it computes itself, so in a split chain the buffer is narrower than the decoder's
    // fc expects -- part of the extract layers live in the tail context or on another stage, and no
    // device row can hold them. Reading such a row as the full feature vector is not a degraded draft,
    // it is a wrong one, so this predicate is false and the caller keeps host copies (where the same
    // layers are gathered from every context that owns one).
    //
    // This is the single place that decides, and both sides ask it: the batch builder in
    // common/speculative.cpp before it adds embd rows, and graph_params() before it resolves
    // layer_inp_src. The assert in models/dflash.cpp build_inp_target_features() stays exactly as it
    // is -- reaching it means those two disagreed, which is a logic error, not a chain shape to tolerate.
    if (src->ne[0] != (int64_t) model.hparams.n_embd_inp_enc()) {
        return false;
    }

    // one of our backends must be able to read the buffer, e.g. not a draft on another GPU than the target output
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    for (const auto & backend : backends) {
        if (ggml_backend_supports_buft(backend.get(), buft)) {
            return true;
        }
    }

    return false;
}

ggml_backend_t llama_context::backend_from_dev(ggml_backend_dev_t dev) const {
    for (const auto & backend : backends) {
        if (ggml_backend_get_device(backend.get()) == dev) {
            return backend.get();
        }
    }
    return nullptr;
}

void llama_context::layer_inp_dev_alloc() {
    synchronize();

    layer_inp_dev     = nullptr;
    layer_inp_dev_dev = nullptr;
    layer_inp_backend = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();

    if (!cparams.embeddings_layer_inp_dev) {
        return;
    }

    int64_t n_layer_inp = 0;
    for (bool enabled : cparams.embeddings_layer_inp) {
        n_layer_inp += enabled ? 1 : 0;
    }
    if (n_layer_inp == 0) {
        return;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_layer_inp_dev.reset(ggml_init(params));

    layer_inp_dev = ggml_new_tensor_2d(ctx_layer_inp_dev.get(), GGML_TYPE_F32, n_layer_inp*model.hparams.n_embd, cparams.n_batch);
    ggml_set_name(layer_inp_dev, "layer_inp_dev");

    layer_inp_dev_dev = model.dev_output();
    auto * buft = layer_inp_dev_dev ? ggml_backend_dev_buffer_type(layer_inp_dev_dev) : ggml_backend_cpu_buffer_type();

    buf_layer_inp_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_layer_inp_dev.get(), buft));
    if (!buf_layer_inp_dev) {
        // e.g. a large n_batch that does not fit next to the model, keep the host copies
        LLAMA_LOG_WARN("%s: failed to allocate the layer input device buffer, using host copies\n", __func__);
        cparams.embeddings_layer_inp_dev = false;
        layer_inp_dev     = nullptr;
        layer_inp_dev_dev = nullptr;
        ctx_layer_inp_dev.reset();
        return;
    }

    LLAMA_LOG_INFO("%s: %10s layer input buffer size = %8.2f MiB (%d layers)\n", __func__,
            ggml_backend_buffer_name(buf_layer_inp_dev.get()), ggml_backend_buffer_get_size(buf_layer_inp_dev.get())/1024.0/1024.0, (int) n_layer_inp);

    layer_inp_backend = layer_inp_dev_dev ? backend_from_dev(layer_inp_dev_dev) : nullptr;
    if (layer_inp_backend && !ev_layer_inp_written) {
        ev_layer_inp_written.reset(ggml_backend_event_new(layer_inp_dev_dev));
        ev_layer_inp_read.reset(ggml_backend_event_new(layer_inp_dev_dev));
        ev_layer_inp_read_recorded = false;
    }
}

bool llama_context::h_nextn_dev_alloc() {
    synchronize();

    auto reset = [&]() {
        h_nextn_dev     = nullptr;
        h_nextn_dev_dev = nullptr;
        h_nextn_backend = nullptr;
        buf_h_nextn_dev.reset();
        ctx_h_nextn_dev.reset();
        ev_h_nextn_written_recorded = false;
        ev_h_nextn_read_recorded    = false;
    };

    // __func__ inside the lambda would be "operator()"
    const char * func = __func__;
    auto fail = [&](const char * why) {
        LLAMA_LOG_WARN("%s: %s, extracting the h_nextn rows to host memory\n", func, why);
        cparams.embeddings_nextn_dev = false;
        reset();
        return false;
    };

    reset();

    if (!cparams.embeddings_nextn_dev) {
        return true;
    }

    h_nextn_dev_dev = model.dev_output();
    h_nextn_backend = h_nextn_dev_dev ? backend_from_dev(h_nextn_dev_dev) : nullptr;
    if (h_nextn_backend == nullptr) {
        return fail("no device backend for the model output");
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_h_nextn_dev.reset(ggml_init(params));

    h_nextn_dev = ggml_new_tensor_2d(ctx_h_nextn_dev.get(), GGML_TYPE_F32, model.hparams.n_embd_out(), cparams.n_batch + cparams.n_seq_max);
    ggml_set_name(h_nextn_dev, "h_nextn_dev");

    buf_h_nextn_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_h_nextn_dev.get(), ggml_backend_dev_buffer_type(h_nextn_dev_dev)));
    if (!buf_h_nextn_dev) {
        return fail("failed to allocate the h_nextn device buffer");
    }

    if (!ev_h_nextn_written) {
        ev_h_nextn_written.reset(ggml_backend_event_new(h_nextn_dev_dev));
        ev_h_nextn_read.reset(ggml_backend_event_new(h_nextn_dev_dev));
    }
    if (!ev_h_nextn_written || !ev_h_nextn_read) {
        return fail("the device has no events");
    }

    LLAMA_LOG_INFO("%s: %10s h_nextn buffer size = %8.2f MiB\n", __func__,
            ggml_backend_buffer_name(buf_h_nextn_dev.get()), ggml_backend_buffer_get_size(buf_h_nextn_dev.get())/1024.0/1024.0);

    return true;
}

bool llama_context::set_embeddings_nextn_dev(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the device rows are batch rows, so the extraction must not be masked
    if (value && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked)) {
        return false;
    }

    if (cparams.embeddings_nextn_dev != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn_dev = value;

    return h_nextn_dev_alloc() && cparams.embeddings_nextn_dev == value;
}

bool llama_context::get_h_nextn_rows(int64_t row0, int64_t n_rows, float * out) {
    if (h_nextn_dev == nullptr || row0 < 0 || n_rows < 0 || row0 + n_rows > h_nextn_dev->ne[1]) {
        return false;
    }
    if (ev_h_nextn_written_recorded) {
        ggml_backend_event_synchronize(ev_h_nextn_written.get());
    } else {
        synchronize();
    }
    ggml_backend_tensor_get(h_nextn_dev, out, row0*h_nextn_dev->nb[1], n_rows*h_nextn_dev->nb[1]);
    return true;
}

bool llama_context::set_h_nextn_rows(int64_t row0, int64_t n_rows, const float * data) {
    if (h_nextn_dev == nullptr || row0 < 0 || n_rows < 0 || row0 + n_rows > h_nextn_dev->ne[1]) {
        return false;
    }
    // a reader may still read these rows
    if (ev_h_nextn_read_recorded) {
        ggml_backend_event_synchronize(ev_h_nextn_read.get());
    }
    ggml_backend_tensor_set(h_nextn_dev, data, row0*h_nextn_dev->nb[1], n_rows*h_nextn_dev->nb[1]);
    return true;
}

bool llama_context::set_state_src(llama_context * src) {
    if (cparams.ctx_state_src != src) {
        sched_need_reserve = true;
    }
    cparams.ctx_state_src = src;
    return supports_state_row();
}

bool llama_context::supports_state_row() const {
    // the MTP graphs read the rows, see llm_graph_context::build_inp_embd_state()
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        return false;
    }

    // the rows replace the host state rows, so they must have the width the graph takes for those
    const ggml_tensor * src = cparams.ctx_state_src ? cparams.ctx_state_src->get_h_nextn_dev() : nullptr;
    if (src == nullptr || src->ne[0] != (int64_t) llama_batch_ext_select_n_embd_state(cparams.ctx_type, model.hparams)) {
        return false;
    }

    // one of our backends must be able to read the buffer
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src->buffer);
    for (const auto & backend : backends) {
        if (ggml_backend_supports_buft(backend.get(), buft)) {
            return true;
        }
    }

    return false;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph

    cparams.causal_attn = value;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because warmups are usually with small batches

    cparams.warmup = value;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    // [QLH] SLIM-ARC: resolved once per call. The prefetcher is a process-wide singleton and the
    // two uses below -- install the per-node eval callback, then scan the freshly built graph --
    // have to agree on whether it is armed; asking twice is what lets a future edit change one of
    // them and leave the other behind. CPU only: the callback path splits the graph at every
    // observed node and synchronizes there, which would break GPU pipelining.
    slim_arc::prefetch_scheduler * sched_arc =
            model.n_gpu_layers() == 0 ? slim_arc::get_global_prefetch_scheduler() : nullptr;

    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        // [QLH] SLIM-ARC 2026-11-02: when the layer-ahead prefetcher is armed, the "which layer
        // is running" signal comes from the per-node eval callback -- the whole graph is
        // submitted at once, so nothing in graph_compute() can know the current layer. The user
        // callback is chained inside eval_cb(). See the resolution at the top of this function.
        if (sched_arc != nullptr) {
            sched_arc->attach_user_callback(cparams.cb_eval, cparams.cb_eval_user_data);
            ggml_backend_sched_set_eval_callback(sched.get(), slim_arc::prefetch_scheduler::eval_cb, sched_arc);
        } else {
            ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
        }

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // [QLH] SLIM-ARC 2026-11-02: scan the freshly built graph for the first node of every
        // layer. Has to run after build_graph() -- the per-tensor names that carry the layer
        // index are assigned by graph_get_cb() while the graph is being built.
        if (sched_arc != nullptr) {
            sched_arc->prepare_graph(gf);
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    if (!batch_inp.tokens.empty() && batch_inp.tokens[0].embd_row >= 0) {
        LLAMA_LOG_ERROR("%s: embd row batches are not supported by encode\n", __func__);
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we skip position continuity checks and output all tokens
    //       a decoder without memory (see decode()) outputs the marked tokens, unless it extracts embeddings
    if (!balloc->init(batch_inp, model.vocab, memory || cparams.embeddings)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs_all);
        return -2;
    };

    const auto & out_ids = balloc->get_out_ids();
    for (uint32_t i = 0; i < n_outputs_all; ++i) {
        output_ids[out_ids[i]] = i;
    }

    n_outputs = n_outputs_all;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_outputs*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_outputs*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_outputs*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_outputs*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_outputs*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    // Both kinds of stage mean "this context drives a chain": a remote-only chain has an empty
    // split_stages, and checking just that sent the whole request down the plain path with the
    // remote stages never contacted.
    if (!split_stages.empty() || !split_remotes.empty()) {
        return decode_split(batch_inp);
    }
    return decode_plain(batch_inp);
}

// [QLH-SPLIT] Does this architecture have a pre-output-norm exit for the relay channel?
//
// A chain's head and every relay stage are asked for cparams.embeddings_pre_norm hidden, which
// the graph builds by assigning t_embd = cur *before* output_norm. Only qwen35 and qwen35moe
// implement that: their graph reads the flag and moves the assignment (src/models/qwen35.cpp,
// src/models/qwen35moe.cpp). Every other architecture ignores cparams.embeddings_pre_norm and
// fills t_embd from the post-norm tensor, so the next stage normalizes a second time. Nothing
// downstream can see that: the argmax usually still matches and only the values are shifted, so
// the chain serves plausible-but-wrong tokens with no error. The list is exactly the two arch
// constants of those files -- adding an architecture here means implementing the exit there
// first, not relaxing this check.
static bool split_arch_exports_pre_norm(llm_arch arch) {
    return arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE;
}

// [QLH-SPLIT] Attach remote stages by endpoint. Returns the number attached, or -1 when the
// chain already has local stages (see the note on split_remotes) or cannot be carried by the
// stage protocol at all.
int llama_context::split_attach_remote(const std::vector<std::string> & endpoints, std::string & err) {
    if (!split_stages.empty()) {
        err = "cannot mix local and remote stages in one chain";
        return -1;
    }

    // [QLH-SPLIT] A remote chain hands every stage one seq_id for the whole batch
    // (src/llama-split-proto.h), so a context that can decode several sequences at once would
    // silently fold them into one. M-RoPE is checked where it can actually be seen: decode_split
    // rejects a batch whose positions need more than the single scalar the protocol carries.
    // Refusing every M-RoPE model here would also refuse this text-only case, where the batch
    // and the stage both encode the position as [p, 0, 0, 0] and nothing is lost.
    //
    // Do not read the check below as a gap to be closed: a remote chain being single-sequence is
    // a design decision, taken deliberately, and this comment is where it is recorded so the
    // next person finds "chosen" instead of "forgotten".
    //
    // Feasible, about 105 lines: one seq_id per OP_DECODE row in the stage protocol
    // (src/llama-split-proto.h) plus threading it through client and stage. Positions are already
    // flattened per row, and context checkpoints are already per-sequence, so a restore order for
    // several sequences would not have to be rethought.
    //
    // Not done because of what a remote chain is for: "this machine cannot hold the model" or
    // "borrow the other machine's compute" -- not multi-tenancy. llama.cpp's concurrency is
    // continuous batching, not high-QPS serving: -np exists so that requests do not sit idle
    // waiting for each other, it is not how this server scales throughput. When several requests
    // really do have to run at once, N servers each grouping their own chain is both simpler and
    // better at isolating a failure than one chain carrying N sequences. That is a boundary of
    // the design, not an unfinished feature.
    if (cparams.n_seq_max > 1) {
        err = "a remote split chain is single-sequence by design (its stage protocol carries one "
              "seq_id per decode frame, so several sequences would be folded into one), and this "
              "context was built with n_seq_max = " + std::to_string(cparams.n_seq_max) +
              ". For concurrency, run one llama-server per chain -- each groups its own chain -- "
              "or use a local chain (--split-layer-range 0:N --split-tail-range N:M), which does "
              "support -np > 1.";
        return -1;
    }

    // Same capability check as the local path: the head of a remote chain is the stage that
    // exports the pre-norm hidden the wire carries, so a head without that exit hands every
    // remote stage a post-norm vector. This one is about the head itself and cannot be delegated
    // to the v4 identity handshake: that reports what a stage loaded (range, width, depth), not
    // which tensor its graph writes at the end. Refusing here is what turns "wrong chain, no
    // error" into a startup failure for the whole class of models the fork cannot relay.
    if (!split_arch_exports_pre_norm(model.arch)) {
        err = std::string("this architecture (") + llm_arch_name(model.arch) +
              ") has no pre-output-norm exit; a remote split chain needs one, because the head "
              "must hand the next stage the raw hidden (src/models/qwen35.cpp)";
        return -1;
    }

    // [QLH-SPLIT] And the architecture is only half of it: the graph has to be *asked* for that
    // output, through the two llama_context_params fields that select it. split_attach() refuses
    // a local chain whose contexts were built without them (they are read once, at construction,
    // so this is a startup property); the remote head is checkable the same way and for the same
    // reason -- without them the head hands over an empty carrier and the first decode fails
    // with -3, several seconds after startup said the chain was attached.
    if (!cparams.embeddings || !cparams.embeddings_pre_norm) {
        err = "this head was created without embeddings + embeddings_pre_norm; a remote chain "
              "head must export its pre-norm hidden (llama_context_params::embeddings and "
              "embeddings_pre_norm are context-construction flags and cannot be set later)";
        return -1;
    }

    // The head must be a proper prefix of the source model, for the same reason split_attach()
    // checks it for local stages: nothing on the other side of a socket can be verified, so the
    // only part of the chain that can be checked is the one that runs here. A head that already
    // covers the whole model would decode every layer and then hand pre-norm hidden to stages
    // that expect layer 8's input -- wrong logits with no error anywhere.
    const auto & hp = model.hparams;
    if (hp.layer_hi < 0 || hp.n_layer_src <= 0) {
        err = "this context loaded the whole model; a remote chain head must load the first "
              "layer range of the source model (--split-layer-range 0:N)";
        return -1;
    }
    if (hp.layer_lo != 0) {
        err = "a remote chain head must start at layer 0 (it covers [" +
              std::to_string(hp.layer_lo) + ", " + std::to_string(hp.layer_hi) + "))";
        return -1;
    }
    if (hp.layer_hi >= hp.n_layer_src) {
        err = "the head already covers layers [0, " + std::to_string(hp.layer_hi) + ") of " +
              std::to_string(hp.n_layer_src) + "; there is nothing left for the remote stages";
        return -1;
    }

    // [QLH-SPLIT] The hop width, same check as split_attach: the head exports n_embd_out and the
    // stage on the other end consumes n_embd_inp, but the wire carries one width (the reply's
    // n_embd and the next OP_DECODE request's n_embd must match a single number). Only the head
    // can be checked from here; split-stage refuses to start when its own three widths disagree,
    // which is the other half.
    if (hp.n_embd_out() != hp.n_embd_inp()) {
        err = "this head exports " + std::to_string(hp.n_embd_out()) + "-wide hidden but consumes " +
              std::to_string(hp.n_embd_inp()) + "-wide input; a chain hop carries one width";
        return -1;
    }

    // The old view (if any) refers to the stages about to be destroyed. Drop it first: a failed
    // connect below leaves no view, which is the state a failed attach is supposed to leave.
    split_memory.reset();
    split_remotes.clear();

    // [QLH-SPLIT] v4: every connection starts with an identity handshake, and the chain checks it
    // against what its own head says the position must be.
    //
    // What can be pinned here, and why it is enough to catch a wrong stage:
    //  - the head's model fixes n_layer_src, both widths and n_vocab for every stage in the
    //    chain (they all load the same file);
    //  - the endpoints are in chain order, so every stage but the last is a relay and the last is
    //    the tail;
    //  - the layer ranges must tile: stage i starts where the previous one ended (the head for
    //    the first stage), and the last stage ends at the head's n_layer_src. Each stage proves
    //    the first half of that from its own layer_lo; the last one proves the far end.
    // The only number this side cannot know before the stage answers is a middle stage's layer_hi
    // -- it is whatever the next stage starts at. So it is pinned to ID_ANY_I32 and the next
    // iteration pins the next stage's layer_lo to the layer_hi that stage just reported. The
    // result is the same tiling check split_attach() performs for local stages, over the wire.
    const int32_t  n_layer_src = hp.n_layer_src;
    const uint32_t n_embd_hop  = (uint32_t) hp.n_embd_out();
    const uint32_t n_vocab     = (uint32_t) model.vocab.n_tokens();
    int32_t        prev_hi     = hp.layer_hi;

    for (size_t i = 0; i < endpoints.size(); ++i) {
        auto r = std::make_unique<llama_split_remote>();
        if (!r->connect(endpoints[i], err)) {
            split_remotes.clear();
            return -1;
        }

        const bool last = (i + 1 == endpoints.size());
        qlh_split::stage_id expect{};
        expect.layer_lo    = prev_hi;
        expect.layer_hi    = last ? n_layer_src : qlh_split::ID_ANY_I32;
        expect.n_layer_src = n_layer_src;
        expect.n_embd_inp  = n_embd_hop;
        expect.n_embd_out  = n_embd_hop;
        expect.n_vocab     = n_vocab;
        expect.role        = last ? qlh_split::ROLE_TAIL : qlh_split::ROLE_RELAY;

        qlh_split::stage_id got{};
        std::string why;
        if (!r->hello(expect, &got, why)) {
            err = "remote stage " + std::to_string(i + 1) + ": " + why;
            split_remotes.clear();
            return -1;
        }
        prev_hi = got.layer_hi;
        split_remotes.push_back(std::move(r));
    }

    // Install the union view over the head and the remote stages: llama_get_memory() is what
    // llama-server's KV bookkeeping mutates, and for a remote chain the only stage with a local
    // memory object is the head. Without this the age-old "nobody trims the remote KV" bug
    // stays: the head trims, the remote keeps serving from positions the head no longer has.
    std::vector<llama_split_remote *> remote_ptrs;
    remote_ptrs.reserve(split_remotes.size());
    for (auto & r : split_remotes) {
        remote_ptrs.push_back(r.get());
    }
    // The head argument is this context's own memory, not get_memory(): the accessor answers with
    // split_memory once a chain exists, and split_memory.reset() above only happens to make the
    // two equal here. Spelling out `memory` is what makes a future edit of that reset harmless.
    split_memory = std::make_unique<llama_memory_split>(
            memory.get(), std::vector<llama_memory_t>{}, remote_ptrs);

    return (int) split_remotes.size();
}

// [QLH-SPLIT] Attach local stages. Returns the number attached, or -1 with a logged reason.
//
// The contiguity check is the reason this returns a value at all: every stage is loaded on its
// own (possibly in another process), so nothing after the loader can see that the ranges do not
// tile the source model. Before this, a wrong --split-tail-range was accepted and produced
// logits that were merely plausible, with no error anywhere.
int llama_context::split_attach(llama_context ** stages, int32_t n) {
    if (n < 0 || (n > 0 && stages == nullptr)) {
        LLAMA_LOG_ERROR("%s: invalid stage list (stages = %p, n_stages = %d)\n",
                __func__, (const void *) stages, n);
        return -1;
    }

    // Mirror of the check in split_attach_remote: a chain is either all-local or remote-only.
    // Without this, attaching local stages after remote ones leaves both containers populated,
    // decode_split takes the remote branch, and split_memory hands llama_get_memory() a KV chain
    // that this decode never writes.
    if (n > 0 && !split_remotes.empty()) {
        LLAMA_LOG_ERROR("%s: cannot mix remote and local stages in one chain\n", __func__);
        return -1;
    }

    // [QLH-SPLIT] A chain is replaced by detaching it first, never by attaching on top of one.
    //
    // split_memory is what get_memory() answers with, so a second attach would build the new
    // union view around the *old* view as its head (see the head argument below), and the old
    // view is destroyed by that very assignment -- a head pointer into freed memory, touched by
    // the first seq_rm/pos_min the server performs. Refusing here also keeps "one chain per
    // context" a property of the API rather than of the caller's discipline.
    if (n > 0 && !split_stages.empty()) {
        LLAMA_LOG_ERROR("%s: this context already drives a chain of %zu stage(s); detach it "
                "first (llama_split_attach(ctx, nullptr, 0)) before attaching another\n",
                __func__, split_stages.size());
        return -1;
    }

    if (n == 0) {
        // Detach: the union view has to go with the stages, or llama_get_memory() keeps
        // answering with a chain that decode() no longer drives.
        //
        // Remote stages are part of that detachment. Clearing only split_stages left them
        // attached: decode() still saw split_remotes non-empty and took decode_split, while
        // split_memory (the union view the KV bookkeeping mutates) had already fallen back to
        // the head -- the exact "nobody trims the remote KV" shape from before that view
        // existed. The reset comes first because the view borrows pointers into the list; the
        // clients' destructors send BYE, which is the teardown the context destructor performs.
        split_memory.reset();
        split_stages.clear();
        split_remotes.clear();
        return 0;
    }

    if (n > 0) {
        const auto & hp = model.hparams;

        // The head always exports pre-norm hidden, so it must be an architecture with that exit.
        // Checked before the range checks because it is the more fundamental blocker: a non-
        // qwen35 model cannot head a chain whatever range it was loaded with, while a whole-model
        // qwen35 head only needs --split-layer-range. See split_arch_exports_pre_norm() for why
        // silently using the post-norm tensor is worse than refusing to start.
        if (!split_arch_exports_pre_norm(model.arch)) {
            LLAMA_LOG_ERROR("%s: this architecture (%s) has no pre-output-norm exit; a split "
                    "chain head must hand the next stage the raw hidden state, and only qwen35 / "
                    "qwen35moe build that output\n",
                    __func__, llm_arch_name(model.arch));
            return -1;
        }

        // A whole-model load has no source range: there is no boundary for the first stage to
        // continue from, and treating it as "the head covers everything" would accept a chain
        // that replays layers already computed here.
        if (hp.layer_hi < 0 || hp.n_layer_src <= 0) {
            LLAMA_LOG_ERROR("%s: this context loaded the whole model; a chain head must load "
                    "the first layer range of the source model\n", __func__);
            return -1;
        }

        // [QLH-SPLIT] The head is the only stage with nothing before it, so its range has to
        // begin at the source model's layer 0. split_attach_remote() has always refused this for
        // a remote head; without the same check here, a local head loaded with
        // --split-layer-range 8:20 plus a --split-tail-range 20:24 tiles 8..24: contiguous, ends
        // at the last layer, passes every other check -- and then decodes a model whose first 8
        // layers were never computed (it never even loaded the token embedding, because
        // needs_tok_embd() is false for a range that does not start at 0). Measured: it started,
        // answered 200 and produced garbage. Nothing on this side of the attach could see that,
        // which is the same reason the two range checks above exist.
        //
        // Placed next to them on purpose: together they define "the head is a prefix of the
        // source model" -- starts at 0, ends before the last layer -- and the stage loop below
        // only has to prove that the rest tiles from there to n_layer_src.
        if (hp.layer_lo != 0) {
            LLAMA_LOG_ERROR("%s: a local chain head must start at layer 0, but this context "
                    "covers [%d, %d) of %d; layers 0..%d would never be computed, and the head "
                    "would decode from hidden states that were never produced. Load it from the "
                    "first layer (--split-layer-range 0:%d) and continue with the stages that "
                    "follow it, or put a [0, %d) stage in front and make this range a relay\n",
                    __func__, hp.layer_lo, hp.layer_hi, hp.n_layer_src, hp.layer_lo - 1,
                    hp.layer_hi, hp.layer_lo);
            return -1;
        }

        if (hp.layer_hi >= hp.n_layer_src) {
            LLAMA_LOG_ERROR("%s: head covers layers [%d, %d) of %d, so there is nothing left to "
                    "attach; the head cannot already be the tail\n",
                    __func__, hp.layer_lo, hp.layer_hi, hp.n_layer_src);
            return -1;
        }

        int32_t prev_hi = hp.layer_hi;
        for (int32_t i = 0; i < n; ++i) {
            if (stages[i] == nullptr) {
                LLAMA_LOG_ERROR("%s: stage %d is null\n", __func__, i + 1);
                return -1;
            }
            if (stages[i] == this) {
                LLAMA_LOG_ERROR("%s: stage %d is the head context itself\n", __func__, i + 1);
                return -1;
            }
            const auto & shp = stages[i]->model.hparams;
            if (shp.layer_hi < 0) {
                LLAMA_LOG_ERROR("%s: stage %d loaded the whole model, not a layer range\n",
                        __func__, i + 1);
                return -1;
            }
            if (shp.n_layer_src != hp.n_layer_src) {
                LLAMA_LOG_ERROR("%s: stage %d comes from a source model with %d layers, the "
                        "head from one with %d\n", __func__, i + 1, shp.n_layer_src, hp.n_layer_src);
                return -1;
            }
            if (shp.layer_lo != prev_hi) {
                LLAMA_LOG_ERROR("%s: stage %d covers layers [%d, %d) but the previous stage ends "
                        "at %d; the chain would skip or replay layers\n",
                        __func__, i + 1, shp.layer_lo, shp.layer_hi, prev_hi);
                return -1;
            }
            // Every stage except the last one is a relay: it consumes hidden and owes hidden.
            // The last stage only has to produce logits, so its architecture does not matter.
            if (i + 1 < n && !split_arch_exports_pre_norm(stages[i]->model.arch)) {
                LLAMA_LOG_ERROR("%s: stage %d (%s) has no pre-output-norm exit; a relay stage "
                        "must forward the raw hidden state, and only qwen35 / qwen35moe build "
                        "that output\n",
                        __func__, i + 1, llm_arch_name(stages[i]->model.arch));
                return -1;
            }
            prev_hi = shp.layer_hi;
        }
        if (prev_hi != hp.n_layer_src) {
            LLAMA_LOG_ERROR("%s: the chain ends at layer %d but the source model has %d layers; "
                    "a tail stage is missing or too short\n", __func__, prev_hi, hp.n_layer_src);
            return -1;
        }

        // [QLH-SPLIT] The hidden rows a hop carries have one width on each side, and the chain
        // names it with three accessors: the exporter writes n_embd_out (output_reserve sizes
        // t_embd that way), the consumer validates the batch row against n_embd_inp
        // (llama_batch_ext::set_token_embd), and the carrier buffer is a single width for every
        // hop between them. They happen to be equal for both qwen35 models this fork is used
        // with, but a model where they differ would copy the wrong number of floats (gather()
        // memcpy) and then be refused by the consumer -- or, worse, silently read past the row.
        // Refuse the chain at attach time instead of building on that coincidence.
        const auto widths_ok = [](const llama_context * c) {
            return c->model.hparams.n_embd_out() == c->model.hparams.n_embd_inp();
        };
        if (!widths_ok(this)) {
            LLAMA_LOG_ERROR("%s: this head exports %u-wide hidden but consumes %u-wide input; a "
                    "chain hop carries one width, so a chain cannot be built from it\n", __func__,
                    this->model.hparams.n_embd_out(), this->model.hparams.n_embd_inp());
            return -1;
        }
        for (int32_t i = 0; i < n; ++i) {
            if (!widths_ok(stages[i])) {
                LLAMA_LOG_ERROR("%s: stage %d exports %u-wide hidden but consumes %u-wide input; "
                        "a chain hop carries one width, so it cannot join one\n", __func__, i + 1,
                        stages[i]->model.hparams.n_embd_out(),
                        stages[i]->model.hparams.n_embd_inp());
                return -1;
            }
        }

        // [QLH-SPLIT] A chain context has to be built to export its hidden instead of logits, and
        // it has to be the *pre-norm* hidden: taking the normed one makes the next stage
        // normalize twice, and the result is plausible-but-shifted (argmax often still matches,
        // so compare values). Both flags are llama_context_params fields read once when the
        // context was created, so this is a startup property and checkable here.
        //
        // This uses the ordinary embedding path (cparams.embeddings + embeddings_pre_norm),
        // NOT set_embeddings_nextn(). The latter is the MTP speculative-decoding channel
        // (`t_h_nextn`); enabling it on a context built without nextn layers leaves the graph
        // asking for a buffer that was never planned -- measured: GGML_ASSERT(buffer) inside
        // the head's own decode, even with a single context.
        //
        // It used to be a WARN, which made this the worst kind of check: the chain started, the
        // first decode could not read the hidden and returned -3, and the only explanation was a
        // line from startup saying the flags were missing. Refuse, like the architecture check.
        const auto relays = [](const llama_context * c) {
            if (c->cparams.embeddings && c->cparams.embeddings_pre_norm) {
                return true;
            }
            LLAMA_LOG_ERROR("%s: a chain context must be created with embeddings + "
                    "embeddings_pre_norm (it owes pre-norm hidden to the next stage); this one "
                    "has embeddings=%d embeddings_pre_norm=%d\n", __func__,
                    (int) c->cparams.embeddings, (int) c->cparams.embeddings_pre_norm);
            return false;
        };
        if (!relays(this)) {
            return -1;
        }
        for (int32_t i = 0; i + 1 < n; ++i) { // the last stage owes logits, not hidden
            if (!relays(stages[i])) {
                LLAMA_LOG_ERROR("%s: stage %d is a relay (it feeds stage %d)\n", __func__, i + 1, i + 2);
                return -1;
            }
        }
    }

    split_stages.assign(stages, stages + n);

    // Install the union view over every stage's KV.
    std::vector<llama_memory_t> tail_mem;
    tail_mem.reserve(split_stages.size());
    for (auto * c : split_stages) {
        tail_mem.push_back(c->get_memory());
    }
    // [QLH-SPLIT] The head of the view is this context's own memory, spelled out rather than
    // through get_memory(): that accessor answers with split_memory itself once a chain exists,
    // so it would nest the new view around the old one (and the assignment destroys the old one).
    // The guard above makes that unreachable today; this is what keeps it unreachable if the
    // guard is ever relaxed.
    split_memory = std::make_unique<llama_memory_split>(
            memory.get(), tail_mem, std::vector<llama_split_remote *>{});

    return n;
}

// [QLH-SPLIT] Drive the chain: this context is the head, split_stages / split_remotes hold
// the rest.
//
// The head always runs the ordinary whole-graph path, asked for pre-norm hidden instead of
// logits (see split_attach). Every later stage is fed that hidden as embeddings, so a stage
// needs no token ids and no access to the model's front. Only the last stage produces logits;
// the ones before it produce hidden again. The empty-chain case stays out of decode() so the
// "no stages" path can never drift from the whole-model path.
// [QLH-SPLIT] The position a batch entry actually contributes to the graph.
//
// llama_batch_allocr expands a *token* entry's single position to [p, p, p, 0] for M-RoPE
// (src/llama-batch.cpp, "expand [p] to [p, p, p, 0]"), while an *embedding* entry keeps whatever
// its caller put in it. A continuation stage is fed embeddings, which are never expanded, so it
// has to be handed the expanded values explicitly -- otherwise every stage after position 0
// rotates a different M-RoPE encoding than the head, which is a silent numerical divergence that
// only shows up as "the chain's logits are close but not identical".
//
// [QLH-SPLIT] The body below is that upstream rule, and this is the only place in this file that
// encodes it: everything that needs a row's graph position goes through this function. It cannot
// be shared with llama_batch.cpp from here (the rule is a loop inside llama_batch_allocr::init),
// so the two have to be changed together -- and the divergence is silent, which is why the
// assertion below is the one thing that will not be: a section count the upstream rule does not
// describe aborts instead of writing past `out`.
static void split_expand_pos(const llama_batch_ext::token & tok, uint32_t n_pos_per_embd, llama_pos * out) {
    GGML_ASSERT(n_pos_per_embd <= GGML_MROPE_SECTIONS);
    const bool expand = tok.id != LLAMA_TOKEN_NULL;
    for (uint32_t j = 0; j < n_pos_per_embd; ++j) {
        out[j] = expand ? (j < 3 ? tok.pos[0] : 0) : tok.pos[j];
    }
}

// [QLH-SPLIT] Does this batch carry more than one sequence?
//
// llama-server's continuous batching (can_batch_with, tools/server/server-context.cpp) submits
// two generating slots in one decode, so a chain has to accept that batch. The only thing that
// cannot carry it is the stage protocol: one OP_DECODE frame holds one seq_id for the whole
// batch (src/llama-split-proto.h), and split_attach_remote already refuses a head whose
// n_seq_max > 1 for exactly that reason. This predicate is therefore used on the remote branch
// only -- as a guard for the day that startup check is relaxed, since a remote chain that
// silently folded two slots into one seq would serve plausible wrong tokens.
static bool split_batch_mixes_seqs(const llama_batch_ext & batch_inp) {
    llama_seq_id only = -2; // -2 = no entry examined yet; real seq ids are >= 0
    for (const auto & tok : batch_inp.tokens) {
        if (tok.seq_ids.size() > 1) {
            return true;
        }
        const llama_seq_id seq = tok.seq_ids.empty() ? 0 : *tok.seq_ids.begin();
        if (only == -2) {
            only = seq;
        } else if (seq != only) {
            return true;
        }
    }
    return false;
}

// [QLH-SPLIT] A context whose output is safe to read.
//
// decode() is async, and the internal getters (get_embeddings_ith / get_logits_ith) do not
// synchronize -- only the public wrappers do, and a caller's wrapper waits on the head's own
// scheduler, never on a stage's. Constructing one of these is the only way to reach the readers
// below, so "synchronize before reading" is a rule the type enforces instead of a comment
// someone has to notice.
struct split_synced {
    llama_context * ctx;

    explicit split_synced(llama_context * from) : ctx(from) { ctx->synchronize(); }
};

// Copy n_tok hidden rows of width n_embd out of a synchronized context into the carrier.
static bool split_gather_hidden(const split_synced & from, std::vector<float> & carrier,
                                int32_t n_tok, int32_t n_embd) {
    for (int32_t i = 0; i < n_tok; ++i) {
        const float * row = from.ctx->get_embeddings_ith(i);
        if (row == nullptr) {
            return false;
        }
        std::memcpy(carrier.data() + (size_t) i * n_embd, row, sizeof(float) * n_embd);
    }
    return true;
}

// [QLH-SPLIT] Which rows the tail owes logits for.
//
// Built once, from the caller's own per-row flag, because two things must never disagree: the
// request the hop builder makes and the check that the answer arrived. The rows the tail was
// actually asked for are also what the "did it produce logits" check looks at -- a speculative
// block marks its sampled row plus the drafts, so the last batch row is not necessarily the one
// that matters.
//
// A batch that flags no output row at all would produce no logits and the caller's
// get_logits_ith() would fail; that case only ever meant "the caller does not care", so the old
// "the last row is the answer" contract is folded in here. That is the point: one function decides
// both, so no later stage has to know the fallback exists or remember to apply it.
struct split_logit_plan {
    std::vector<int32_t> rows; // ascending indices into the batch

    bool asks(int32_t row, bool last) const {
        if (!last) {
            return false;
        }
        for (const int32_t r : rows) {
            if (r == row) {
                return true;
            }
        }
        return false;
    }
};

static split_logit_plan split_plan_logits(const llama_batch_ext & batch_inp, int32_t n_tok) {
    bool any_output = false;
    for (const auto & tok : batch_inp.tokens) {
        if (tok.output) {
            any_output = true;
            break;
        }
    }

    split_logit_plan plan;
    for (int32_t i = 0; i < n_tok; ++i) {
        if (batch_inp.tokens[i].output || (!any_output && i + 1 == n_tok)) {
            plan.rows.push_back(i);
        }
    }
    return plan;
}

// The shape of one hop: every stage of a chain sees the same token count and the same hidden
// width, and the same number of position sections per token (4 for M-RoPE, 1 otherwise).
struct split_hop_shape {
    int32_t  n_tok;
    int32_t  n_embd;
    uint32_t n_pos;
};

// [QLH-SPLIT] One row of one hop: whose hidden it is, which sequence(s) own it, where it sits.
//
// Positions come straight from the incoming batch, so every stage agrees on them by construction.
// There is no n_past to keep in sync: position IS the state, and the KV of each stage is indexed
// by it. A row's pos is the row's own history position, never an index into this batch -- with
// two slots in one batch their positions do not even advance in lockstep.
static int split_add_hop_row(llama_batch_ext * hop, const llama_batch_ext & batch_inp, int32_t i,
                             const std::vector<float> & carrier, const split_hop_shape & shape,
                             size_t stage_no) {
    const auto & tok = batch_inp.tokens[i];
    const llama_embd row = { carrier.data() + (size_t) i * shape.n_embd, 1, (size_t) shape.n_embd };

    // [QLH-SPLIT] Carry every seq_id of this row, not just the first one.
    //
    // Deciding "which sequence owns this hidden row" is the head's job and it already did it: the
    // row is row i of a batch the head decoded with these exact seq_ids. Dropping all but the
    // first is what made -np > 1 chains fold two slots into seq 0 (silently wrong tokens before,
    // a hard -3 after the batch-level refusal was added).
    //
    // add_embd() takes one seq_id, so the rest go through add_seq(); it is one row with several
    // seq_ids, which is what llama_batch_ext::token::seq_ids holds anyway. A *new* API variant is
    // not needed for it.
    llama_seq_id seq = 0;
    if (!tok.seq_ids.empty()) {
        seq = *tok.seq_ids.begin();
    }

    // [QLH-SPLIT] A continuation stage is fed embeddings, and an embedding entry is not expanded
    // by llama_batch_allocr the way a token entry is -- so hand it the position the head's graph
    // actually used, not the raw one.
    llama_pos pos[GGML_MROPE_SECTIONS] = {0, 0, 0, 0};
    split_expand_pos(tok, shape.n_pos, pos);

    // [QLH-SPLIT] A failed add means the stage's batch is smaller than this hop
    // (n_tokens_max comes from its -b / -ubatch-size). Every call below indexes by i, so
    // continuing would set positions and output flags on an entry that does not exist and then
    // decode a batch that silently lacks this token: the stage's positions and KV drift from the
    // head with nothing logged.
    //
    // -3 matches what the remote path reports for exactly this condition: the stage answers
    // ST_NO_SPACE, and the remote branch turns any non-negative rc from
    // llama_split_remote::decode (1 = "retry with a smaller batch") into -3, because retrying
    // cannot grow a capacity fixed at startup. The client server already knows -3 as a failed
    // request.
    const int32_t added = llama_batch_ext_add_embd(hop, seq, row);
    if (added < 0) {
        LLAMA_LOG_ERROR("%s: stage %zu cannot hold this hop (%d tokens); start it with "
                "-b / -ubatch-size at least the driving server's -ubatch-size\n",
                __func__, stage_no, shape.n_tok);
        return -3;
    }

    // add_seq() before the other setters (include/llama.h). A seq id the stage does not know
    // cannot be dropped here: the row would land on the wrong sequence's KV and the stage would
    // still return well-formed output.
    for (const llama_seq_id seq_extra : tok.seq_ids) {
        if (seq_extra == seq) {
            continue;
        }
        if (!llama_batch_ext_add_seq(hop, added, seq_extra)) {
            LLAMA_LOG_ERROR("%s: stage %zu does not accept seq_id %d (its n_seq_max is "
                    "smaller than the driving server's -np); the row would be written into the "
                    "wrong sequence's KV\n", __func__, stage_no, seq_extra);
            return -3;
        }
    }

    llama_batch_ext_set_pos(hop, i, pos);
    return 0;
}

// [QLH-SPLIT] What this row asks the stage to produce.
//
// A stage that hands hidden downstream needs a row for every token; the last stage owes logits
// only for the rows the caller actually samples from, and the plan carries that distinction.
// Marking every token of the last hop as an output (what this used to do) asked the tail to
// materialise n_batch vocab-sized rows in its output buffer -- 485 MiB at -b 512, 1.7 GiB at
// -ub 2048 for this model -- and then sampled one of them. The tail's n_outputs_max is sized for
// the real row count (common.cpp, common_context_params_to_llama), so over-asking is not just
// waste, it now fails the request in output_reserve.
//
// [QLH-SPLIT] Exactly one setter per row, because both of them write the same flag:
// llama_batch_ext_set_output_embd() and llama_batch_ext_set_output_logits() are both
// llama_batch_ext::set_output() (src/llama-batch.cpp), and token::output carries no "embd or
// logits" distinction ("TODO: have dedicated output flags", src/llama-batch.h). Calling both
// meant the second call decided, so a relay row was marked output only because it ended up as
// ask_logits=false -- the same value -- while the *reason* it worked is elsewhere: a stage built
// with cparams.embeddings marks every token as output regardless (decode_plain, output_all). A
// relay stage built without it would then have handed the gather no rows at all, and the chain
// would have failed its first decode with -3 instead of saying why. split_attach() now refuses
// that chain, but the silent overwrite was wrong on its own.
//
// So: the relay asks for a hidden row on every token it forwards, and the tail asks for logits
// only on the rows the caller samples from.
static void split_set_hop_row_output(llama_batch_ext * hop, int32_t i,
                                     const split_logit_plan & plan, bool last) {
    if (last) {
        llama_batch_ext_set_output_logits(hop, i, plan.asks(i, last));
    } else {
        llama_batch_ext_set_output_embd(hop, i, true);
    }
}

// One hop batch for local stage `stage_no`: one row per incoming token, in the incoming order.
static int split_build_hop(llama_context * stage, const llama_batch_ext & batch_inp,
                           const std::vector<float> & carrier, const split_logit_plan & plan,
                           const split_hop_shape & shape, bool last, size_t stage_no,
                           std::unique_ptr<llama_batch_ext> & hop) {
    hop.reset(llama_batch_ext_init(stage));
    if (!hop) {
        return -3;
    }
    for (int32_t i = 0; i < shape.n_tok; ++i) {
        const int rc = split_add_hop_row(hop.get(), batch_inp, i, carrier, shape, stage_no);
        if (rc != 0) {
            return rc;
        }
        split_set_hop_row_output(hop.get(), i, plan, last);
    }
    return 0;
}

// Decode one built hop and check what came back: hidden for a relay, logits for the tail.
static int split_run_hop(llama_context * stage, llama_batch_ext & hop,
                         const split_hop_shape & shape, const split_logit_plan & plan,
                         std::vector<float> & carrier, bool last, size_t stage_no) {
    const int rc = stage->decode(hop);
    if (rc != 0) {
        // [QLH-SPLIT] rc == 1 means "could not find a KV slot for this batch -- retry with a
        // smaller one" (llama.h), and the contract behind it is that the memory state was not
        // touched. Neither half holds here: the head committed this batch before the hop was
        // built, so handing 1 back would make the caller decode the same rows into the head a
        // second time, and a stage cannot grow its KV anyway (it was fixed at startup). Every
        // positive status is fatal for this request, exactly as the remote branch treats it.
        return rc < 0 ? rc : -3;
    }

    if (!last) {
        if (!split_gather_hidden(split_synced(stage), carrier, shape.n_tok, shape.n_embd)) {
            LLAMA_LOG_ERROR("%s: stage %zu produced no hidden state\n", __func__, stage_no);
            return -3;
        }
        return 0;
    }

    // A tail context started with embeddings_pre_norm (the relay shape) builds no t_logits at
    // all: the hop decodes successfully, every output row stays null, and the caller's
    // get_logits_ith() fails one layer up with no hint why. --role is hand-typed for the remote
    // case and the params for the local one, so this is the only place the mismatch is visible.
    // split_synced first: the internal getter does not synchronize, and the public one the caller
    // later uses only waits on this context's own head scheduler, never on this stage.
    const split_synced synced(stage);
    for (const int32_t r : plan.rows) {
        if (synced.ctx->get_logits_ith(r) == nullptr) {
            LLAMA_LOG_ERROR("%s: the last stage produced no logits for row %d; it must be "
                    "the tail, not a relay context (embeddings_pre_norm)\n", __func__, r);
            return -3;
        }
    }
    return 0;
}

// [QLH-SPLIT] Does every sequence in this batch start at position 0?
//
// Only a batch that starts at position 0 may be decoded into an empty (rebuilt) chain: letting
// it continue from a cached prefix would write positions 117.. into a KV with no 0..116 in it.
// For a remote chain the batch carries one sequence by construction (split_attach_remote refuses
// n_seq_max > 1), but the check is written per sequence so it stays correct if that ever changes.
static bool split_starts_at_zero(const llama_batch_ext & batch_inp) {
    std::vector<std::pair<llama_seq_id, llama_pos>> mins; // first position per sequence
    const auto observe = [&mins](llama_seq_id seq, llama_pos pos) {
        for (auto & it : mins) {
            if (it.first == seq) {
                if (pos < it.second) {
                    it.second = pos;
                }
                return;
            }
        }
        mins.emplace_back(seq, pos);
    };
    for (const auto & tok : batch_inp.tokens) {
        if (tok.seq_ids.empty()) {
            observe(0, tok.pos[0]);
            continue;
        }
        for (const llama_seq_id seq : tok.seq_ids) {
            observe(seq, tok.pos[0]);
        }
    }
    for (const auto & it : mins) {
        if (it.second != 0) {
            return false;
        }
    }
    return true;
}

// [QLH-SPLIT] Admit this batch into the chain, or refuse it before anything is decoded.
int llama_context::split_admit_batch(const llama_batch_ext & batch_inp) {
    // [QLH-SPLIT] A local chain carries every row's own seq_ids downstream (split_build_hop),
    // so a batch may mix sequences: each stage writes each row into that row's own KV stream and
    // position, exactly like the head's own decode_plain does. Verified: two seqs, one embedding
    // row each, through a tail context -- KV/state land on their own seq, each row keeps its own
    // pos, and the per-seq logits are bit-identical to a single-seq run.
    //
    // n_seq_max is deliberately NOT rejected in split_attach(): -np 2 / -np 4 chains need it,
    // and both the head and the stages run the ordinary multi-sequence paths.
    //
    // The remote branch is the exception; see split_batch_mixes_seqs().
    if (!split_remotes.empty() && split_batch_mixes_seqs(batch_inp)) {
        LLAMA_LOG_ERROR("%s: a remote chain carries one sequence per batch (the stage protocol "
                "has one seq_id per OP_DECODE), and this batch mixes %d tokens of several "
                "sequences; the remote head is built with n_seq_max > 1\n",
                __func__, (int) batch_inp.tokens.size());
        return -3;
    }
    return split_self_heal(batch_inp);
}

// [QLH-SPLIT] Re-attach a chain that a failed remote stage left untrustworthy, if this batch can
// be decoded into the empty chain a rebuild produces.
int llama_context::split_self_heal(const llama_batch_ext & batch_inp) {
    // A remote stage that failed an operation with no error channel (clear, or an op the stage
    // protocol does not carry) leaves the chain holding different state on different machines.
    //
    // [QLH-SPLIT] Self-heal: a stage that came back is re-attached here instead of requiring a
    // server restart. Measured before this existed: restarting only the stage left every request
    // at 500 forever, because the flag is only ever cleared by a successful clear() of the whole
    // chain, and nothing performed one.
    //
    // The decision belongs to the memory layer, but the *safety interlock* belongs here, because
    // only the batch knows whether it can be decoded into an empty chain: after a rebuild every
    // stage holds nothing, so only a batch that starts at position 0 may be decoded. The
    // alternative -- rebuilding and letting the batch continue from a cached prefix -- would
    // write positions 117.. into a KV with no 0..116 in it: the exact "empty KV, positions keep
    // counting" state this whole layer exists to make impossible.
    //
    // When the batch does not qualify, the -3 below is not a dead end: llama-server's decode
    // error path drops the slot's cached prompt (server-context.cpp, server_context::decode), so
    // the retried request starts at 0 and this branch gets a batch it can rebuild for.
    if (!split_memory || split_memory->remote_failure.empty()) {
        return 0;
    }

    std::string why;
    if (!split_memory->try_rebuild(split_starts_at_zero(batch_inp), why)) {
        LLAMA_LOG_ERROR("%s: refusing to decode: %s\n", __func__, why.c_str());
        return -3;
    }
    return 0;
}

// [QLH-SPLIT] The wire carries every position section, not one scalar per token: section-major,
// exactly the layout the head's own batch uses.
static std::vector<int32_t> split_remote_positions(const llama_batch_ext & batch_inp,
                                                   int32_t n_tok, uint32_t n_pos) {
    std::vector<int32_t> positions((size_t) n_tok * n_pos);
    for (int32_t i = 0; i < n_tok; ++i) {
        llama_pos pos[GGML_MROPE_SECTIONS] = {0, 0, 0, 0};
        split_expand_pos(batch_inp.tokens[i], n_pos, pos);
        for (uint32_t j = 0; j < n_pos; ++j) {
            positions[(size_t) j * n_tok + i] = (int32_t) pos[j];
        }
    }
    return positions;
}

// [QLH-SPLIT] The five split_remote_* answer slots are written only through these three
// functions, and both stores start by resetting: "clear before fill" is therefore a property of
// writing at all, not a step the caller has to remember. split_remote_reset() on its own is the
// frame-start clear -- the one that must happen before anything can fail, so a chain that ends up
// returning hidden (or a stage that errors out) answers null from get_logits_ith() instead of
// silently handing back logits from an earlier request.
void llama_context::split_remote_reset() {
    split_remote_logits.clear();
    split_remote_n_vocab    = 0;
    split_remote_logits_row = -1;
    split_remote_cands.clear();
    split_remote_cands_row  = -1;
    split_remote_verify.clear();
    split_remote_nextn.clear();
    split_remote_nextn_embd = 0;
    split_remote_nextn_row0 = -1;
    split_remote_tap.clear();
    split_remote_tap_embd = 0;
    split_remote_tap_rows = 0;
}

void llama_context::split_remote_store_cands(std::vector<qlh_split::token_cand> && cands, int32_t row) {
    split_remote_reset();
    split_remote_cands     = std::move(cands);
    split_remote_cands_row = row;
}

void llama_context::split_remote_store_logits(std::vector<float> && logits, uint32_t n_vocab, int32_t row) {
    split_remote_reset();
    split_remote_logits     = std::move(logits);
    split_remote_n_vocab    = n_vocab;
    split_remote_logits_row = row;
}

// A middle remote stage must answer with hidden; a tail in the middle of the chain answers with
// logits instead, and the carrier would keep the previous stage's hidden -- the next stage then
// computes on the wrong input with no error anywhere. --role is typed by hand, so this is the
// only place it can be checked.
static int split_remote_take_hidden(std::vector<float> & carrier,
                                    const std::vector<float> & out_hidden, bool last,
                                    size_t stage_no) {
    if (last) {
        return 0;
    }
    if (out_hidden.empty()) {
        LLAMA_LOG_ERROR("%s: remote stage %zu returned logits instead of hidden; every stage "
                "before the last must be started with --role relay\n", __func__, stage_no);
        return -3;
    }
    carrier = out_hidden; // next stage consumes this
    return 0;
}

// [QLH-SPLIT] Record that the chain diverged because a hop failed.
//
// decode_plain() has already committed this batch to the head's KV, and so has every stage before
// the failing one: their positions run ahead of the stages that never received the rows. Nothing
// in the chain can be rewound to repair that -- a recurrent stage only keeps n_rs_seq snapshots,
// and a partial seq_rm on a hybrid model is precisely the operation that has aborted in this tree
// before -- so this is recorded instead of rolled back.
//
// "Recorded" is the memory layer's own dead-chain flag (llama_memory_split::remote_failure), which
// decode_split consults before computing anything: the next request answers -3 with this reason
// instead of continuing from a KV that no longer agrees with itself. A remote chain repairs itself
// on the next batch that starts at position 0 (llama_memory_split::try_rebuild); a local chain
// stays refused until it is re-attached, which is the honest state -- its stages are borrowed
// contexts that this side cannot rebuild, and a refused request is strictly better than a token
// computed from rows the tail never saw.
static void split_note_diverged(llama_memory_split * mem, size_t stage_no, int rc) {
    if (mem == nullptr || !mem->remote_failure.empty()) {
        return;
    }
    mem->remote_failure = "stage " + std::to_string(stage_no) + " failed the hop (rc=" +
            std::to_string(rc) + "): the head and every stage before it already committed this "
            "batch, so their KV holds positions this stage never received";
    LLAMA_LOG_ERROR("%s: %s\n", __func__, mem->remote_failure.c_str());
}

// [QLH-SPLIT] Remote chain: the head ran locally, hand the rows to each remote stage in turn and
// adopt the last stage's answer.
int llama_context::split_drive_remote(const llama_batch_ext & batch_inp, std::vector<float> & carrier,
                                      int32_t n_tok, int32_t n_embd) {
    split_remote_reset();

    const uint32_t             n_pos     = model.hparams.n_pos_per_embd();
    const std::vector<int32_t> positions = split_remote_positions(batch_inp, n_tok, n_pos);
    const int32_t seq = !batch_inp.tokens.empty() && !batch_inp.tokens[0].seq_ids.empty()
                      ? *batch_inp.tokens[0].seq_ids.begin() : 0;

    std::vector<float> out_hidden;
    std::vector<float> out_logits;
    uint32_t           out_n_vocab = 0;
    std::vector<qlh_split::token_cand> out_cands;
    std::vector<int32_t>               out_verify;
    std::vector<float>                 out_nextn;
    uint32_t                           out_nextn_embd = 0;
    std::string        err;

    // [QLH-SPLIT] Every failure exit below has the same shape: the head (and every stage before
    // s) already committed this batch, so the chain is diverged and has to be recorded as such --
    // see split_note_diverged. One lambda keeps the exits from drifting apart.
    const auto fail = [this](int rc, size_t stage_no) {
        split_note_diverged(split_memory.get(), stage_no, rc);
        return rc < 0 ? rc : -3;
    };

    // [QLH-SPLIT] v9: the layer-input tap request goes to *every* remote stage, not just the last one.
    // A tap layer can land on any stage of the chain -- a three-stage chain puts some of DFlash's 8
    // taps on the relay -- and each stage answers the ids inside its own range (the protocol makes the
    // ids source-model ids for exactly this reason). The width is this model's own trunk width, which
    // is what a layer input is.
    llama_split_tap_req tap_req;
    tap_req.ids    = split_tap_ids.empty() ? nullptr : &split_tap_ids;
    tap_req.n_embd = (uint32_t) model.hparams.n_embd;
    llama_split_tap_out out_tap;

    // [QLH-SPLIT] v9: the taps accumulate across stages here and are stored last, for the same reason
    // the h_nextn rows are (see store_nextn below): the answer stores reset every per-frame slot.
    std::vector<split_tap_block> tap_blocks;
    uint32_t                     tap_embd = 0;
    uint32_t                     tap_rows = 0;

    // [QLH-SPLIT] v5: only the last stage can answer with candidates (a relay answers hidden
    // by construction), and only when the driving side asked for this frame. Taken and
    // cleared here -- before any of the failure exits below -- so a failed frame cannot leave
    // the next one asking for something the sampler did not plan for.
    const bool want_cands  = split_cand_wanted;
    const bool want_verify = split_verify_wanted;
    const bool want_nextn  = split_nextn_wanted;
    split_cand_wanted   = false;
    split_verify_wanted = false;
    split_nextn_wanted  = false;

    // [QLH-SPLIT] v7: a verify frame has to fit in the tail's output-row cap, and MAX_VERIFY_ROWS
    // is the number both ends agree on (llama-split-proto.h). A frame over it falls back to the
    // whole-row path rather than being answered partly: the caller that armed this channel reads
    // token ids, not logits, so a partial answer would leave the extra rows unreadable. Falling
    // back here is safe -- it is the path that existed before the channel did -- and the row
    // count comes from the batch, before anything is sent, so the fallback is never discovered
    // after the tail already answered.
    bool verify_rows = want_verify && n_tok <= (int32_t) qlh_split::MAX_VERIFY_ROWS;
    if (want_verify && !verify_rows && !split_verify_row_warned) {
        split_verify_row_warned = true;
        LLAMA_LOG_WARN("%s: a verify frame with %d rows does not fit MAX_VERIFY_ROWS (%u); this "
                "decode takes the whole-row path (the draft block is capped to %u rows by the "
                "server, so this means the batch was assembled elsewhere)\n",
                __func__, n_tok, qlh_split::MAX_VERIFY_ROWS, qlh_split::MAX_VERIFY_ROWS - 1);
    }

    for (size_t s = 0; s < split_remotes.size(); ++s) {
        const bool last = (s + 1 == split_remotes.size());
        const qlh_split::cand_req * cand = (last && want_cands) ? &split_cand_req : nullptr;
        out_tap.ids.clear();
        out_tap.rows.clear();
        out_tap.n_embd = 0;
        const int rc = split_remotes[s]->decode(carrier, positions, n_pos, (uint32_t) n_embd, seq,
                                                cand, split_cand_bias, split_cand_hist,
                                                last && verify_rows, last && want_nextn,
                                                tap_req,
                                                out_hidden, out_logits, out_n_vocab, out_cands,
                                                out_verify, out_nextn, out_nextn_embd, out_tap, err);
        if (rc != 0) {
            LLAMA_LOG_ERROR("%s: remote stage %zu failed: %s (rc=%d)\n",
                            __func__, s + 1, err.c_str(), rc);
            return fail(rc, s + 1);
        }
        const int hrc = split_remote_take_hidden(carrier, out_hidden, last, s + 1);
        if (hrc != 0) {
            return fail(hrc, s + 1);
        }
        // [QLH-SPLIT] v9: the taps are per stage and the stages own disjoint ranges, so the blocks
        // just accumulate. A stage that owned none contributes nothing (its flags say so), which is
        // not an error: the head decides what a missing tap means, and it is the only side that knows
        // which layers the draft model needs.
        for (size_t t = 0; t < out_tap.ids.size(); ++t) {
            split_tap_block block;
            block.id = out_tap.ids[t];
            const size_t stride = (size_t) out_tap.n_embd;
            block.rows.resize((size_t) n_tok * stride);
            // The stage answered [row][tap][n_embd]; a block keeps [row][n_embd] for one tap, so the
            // copy walks rows out of a strided block.
            for (int32_t r = 0; r < n_tok; ++r) {
                memcpy(block.rows.data() + (size_t) r * stride,
                       out_tap.rows.data() + (((size_t) r * out_tap.ids.size()) + t) * stride,
                       stride * sizeof(float));
            }
            tap_blocks.push_back(std::move(block));
        }
        if (!out_tap.ids.empty()) {
            tap_embd = out_tap.n_embd;
            tap_rows = (uint32_t) n_tok;
        }
    }

    // [QLH-SPLIT] v8: the h_nextn rows, when they were asked for and the stage answered them.
    //
    // Stored *after* the answer slots below, not before: those stores start by resetting every
    // per-frame slot (that is what makes "clear before fill" a property of writing at all), and
    // they reset this one too. Doing the rows first silently dropped them -- measured: the stage
    // answered 2048 floats and the draft was told there were none.
    //
    // A stage that has no NextN block leaves them empty, and that is not an error: the draft
    // driver reads "no rows" as "do not draft", while the frame's own answer still stands.
    const auto store_nextn = [&]() {
        if (!want_nextn) {
            return;
        }
        if (!out_nextn.empty() && out_nextn_embd == (uint32_t) n_embd) {
            split_remote_nextn      = std::move(out_nextn);
            split_remote_nextn_embd = out_nextn_embd;
            // A verify frame answers every row of the frame, a plain decode only its last row.
            const uint32_t n_rows = (uint32_t) (split_remote_nextn.size() / out_nextn_embd);
            split_remote_nextn_row0 = (int32_t) (n_tok - (int32_t) n_rows);
        } else if (!split_nextn_warned && out_nextn_embd == 0 && split_mtp_ready()) {
            // Only worth one line per process: a chain whose tail cannot answer them would log
            // this for every frame, and a log line per token is not a diagnosis.
            split_nextn_warned = true;
            LLAMA_LOG_WARN("%s: the remote tail answered no h_nextn rows; MTP drafting is "
                           "unavailable on this chain (the stage reported a draft context but "
                           "sends no rows)\n", __func__);
        }
    };

    // [QLH-SPLIT] v9: the layer-input taps the stages answered. Stored after the answer slots for the
    // same reason as the h_nextn rows above (the two stores below reset every per-frame slot), and
    // empty when the stages owned none of the asked layers -- which only the draft driver can judge,
    // so it is carried as an empty answer rather than as a failure of the frame.
    const auto store_tap = [&]() {
        split_remote_tap      = std::move(tap_blocks);
        split_remote_tap_embd = tap_embd;
        split_remote_tap_rows = tap_rows;
    };

    // The request was for this frame only; it was consumed at the top of this block.
    if (!out_verify.empty()) {
        // [QLH-SPLIT] v7: one token id per row. No row of logits crossed the wire, so
        // get_logits_ith() answers null here by construction -- the caller was told to read
        // tokens (llama_split_verified_token) and common_sampler_sample does exactly that.
        split_remote_verify = std::move(out_verify);
        store_nextn();
        store_tap();
        return 0;
    }
    if (!out_cands.empty()) {
        // Candidate answer: there is no whole row, and get_logits_ith() must say so instead
        // of serving an empty buffer. The sampler plans around this (common_sampling.h).
        split_remote_store_cands(std::move(out_cands), n_tok - 1);
        store_nextn();
        store_tap();
        return 0;
    }
    if (!out_logits.empty() && out_n_vocab > 0) {
        split_remote_store_logits(std::move(out_logits), out_n_vocab, n_tok - 1);
        store_nextn();
        store_tap();
        return 0;
    }

    // No logits and no candidates. A last stage started as --role relay answers with hidden
    // instead, and that used to be accepted silently: get_logits_ith() answered null, and
    // the caller failed one layer up with nothing pointing at the misconfigured stage.
    //
    // The v4 handshake now rejects this at attach time (the last position expects a tail),
    // and so does a rebuild. This check stays as the runtime backstop for the path the
    // handshake cannot see: a stage that was already attached with the right role argument
    // but answers hidden because its own decode path is wrong. It costs one branch and it is
    // the only thing between that and serving null logits as a success.
    LLAMA_LOG_ERROR("%s: the last remote stage returned no logits; it must be started "
            "with --role tail (a relay stage exports hidden and cannot end the chain)\n",
            __func__);
    // Not recorded as a divergence: every stage decoded this batch successfully, so the chain's
    // KV still agrees with itself -- what is wrong is the answer's type, which the handshake at
    // attach time already refuses. Killing the chain here would turn a misconfigured --role into
    // a local chain that can never serve again, for no gain.
    return -3;
}

// [QLH-SPLIT] Local chain: drive every borrowed stage in turn, one hop each.
int llama_context::split_drive_local(const llama_batch_ext & batch_inp, std::vector<float> & carrier,
                                     int32_t n_tok, int32_t n_embd) {
    const split_hop_shape  shape = { n_tok, n_embd, model.hparams.n_pos_per_embd() };
    const split_logit_plan plan  = split_plan_logits(batch_inp, n_tok);

    for (size_t s = 0; s < split_stages.size(); ++s) {
        const bool last = (s + 1 == split_stages.size());

        std::unique_ptr<llama_batch_ext> hop;
        int rc = split_build_hop(split_stages[s], batch_inp, carrier, plan, shape, last, s + 1, hop);
        if (rc == 0) {
            rc = split_run_hop(split_stages[s], *hop, shape, plan, carrier, last, s + 1);
        }
        if (rc != 0) {
            // [QLH-SPLIT] Same rule as the remote branch: the head already committed this batch,
            // so a failed hop leaves the chain diverged. Recorded, not rolled back -- see
            // split_note_diverged for why a rewind is not an option here.
            split_note_diverged(split_memory.get(), s + 1, rc);
            return rc < 0 ? rc : -3;
        }
    }
    return 0;
}

// [QLH-SPLIT] decode() with a split chain: this context is stage 0 and runs the whole graph, then
// the hidden rows travel hop by hop -- over the wire for a remote chain, stage by stage in this
// process for a local one.
int llama_context::decode_split(const llama_batch_ext & batch_inp) {
    int rc = split_admit_batch(batch_inp);
    if (rc != 0) {
        return rc;
    }

    // Stage 0 is this context and runs the ordinary whole-graph path; it is asked for
    // pre-norm hidden instead of logits (see split_attach).
    rc = decode_plain(batch_inp);
    if (rc != 0) {
        return rc;
    }

    const int32_t n_tok  = (int32_t) batch_inp.tokens.size();
    // [QLH-SPLIT] The width of a hidden row is the *output* width of the stage that produced it
    // (output_reserve sizes t_embd as n_embd_out), and the next stage reads it back at its own
    // n_embd_inp. Split_attach() refuses a chain where those differ, so one number is correct
    // for every hop; using n_embd here only worked because this model has all three equal.
    const int32_t n_embd = (int32_t) model.hparams.n_embd_out();

    // The carrier is the same buffer for every hop: each stage overwrites it with its own
    // output, so a chain of N stages costs one buffer, not N.
    std::vector<float> carrier((size_t) n_tok * (size_t) n_embd);

    if (!split_gather_hidden(split_synced(this), carrier, n_tok, n_embd)) {
        LLAMA_LOG_ERROR("%s: head stage produced no hidden state\n", __func__);
        return -3;
    }

    if (!split_remotes.empty()) {
        return split_drive_remote(batch_inp, carrier, n_tok, n_embd);
    }
    return split_drive_local(batch_inp, carrier, n_tok, n_embd);
}

int llama_context::decode_plain(const llama_batch_ext & batch_inp) {
    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const bool has_embd_row = !batch_inp.tokens.empty() && batch_inp.tokens[0].embd_row >= 0;
    if (has_embd_row && !supports_embd_row()) {
        LLAMA_LOG_ERROR("%s: this context cannot read embd rows (needs a DFlash/DSpark decoder and a layer input device buffer in ctx_other)\n", __func__);
        return -1;
    }

    const bool has_state_row = !batch_inp.tokens.empty() && batch_inp.tokens[0].state_row >= 0;
    if (has_state_row && !supports_state_row()) {
        LLAMA_LOG_ERROR("%s: this context cannot read state rows (needs an MTP context and an h_nextn device buffer in ctx_state_src)\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    // ordering for the layer input device buffer
    llama_context * inject_src     = has_embd_row ? cparams.ctx_other : nullptr;
    ggml_backend_t  inject_backend = nullptr;
    if (inject_src) {
        inject_backend = inject_src->ev_layer_inp_written ? backend_from_dev(inject_src->layer_inp_dev_dev) : nullptr;
        if (inject_backend) {
            // wait on the host: a device-side wait would stall our queue for the whole target batch,
            // which trips the Metal command buffer timeout on long prompts
            ggml_backend_event_synchronize(inject_src->ev_layer_inp_written.get());
        } else {
            inject_src->synchronize();
        }
    }
    if (layer_inp_dev && ev_layer_inp_read && ev_layer_inp_read_recorded) {
        // do not overwrite rows the reader has not consumed yet
        ggml_backend_event_wait(layer_inp_backend, ev_layer_inp_read.get());
    }

    // ordering for the h_nextn device buffer, as for the layer input buffer
    llama_context * state_src     = has_state_row ? cparams.ctx_state_src : nullptr;
    ggml_backend_t  state_backend = nullptr;
    if (state_src) {
        state_backend = backend_from_dev(state_src->h_nextn_dev_dev);
        if (state_src->ev_h_nextn_written_recorded) {
            ggml_backend_event_synchronize(state_src->ev_h_nextn_written.get());
        } else {
            state_src->synchronize();
        }
    }
    const bool write_h_nextn = cparams.embeddings_nextn_dev && h_nextn_dev != nullptr;
    if (write_h_nextn && ev_h_nextn_read_recorded) {
        ggml_backend_event_wait(h_nextn_backend, ev_h_nextn_read.get());
    }

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            // the device buffer replaces the host extraction, see build_h_nextn_dev()
            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE && !write_h_nextn) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    if (layer_inp_dev && ev_layer_inp_written) {
        ggml_backend_event_record(ev_layer_inp_written.get(), layer_inp_backend);
    }
    if (write_h_nextn) {
        ggml_backend_event_record(ev_h_nextn_written.get(), h_nextn_backend);
        ev_h_nextn_written_recorded = true;
    }
    if (state_src) {
        if (state_backend) {
            ggml_backend_event_record(state_src->ev_h_nextn_read.get(), state_backend);
            state_src->ev_h_nextn_read_recorded = true;
        } else {
            synchronize();
        }
    }
    if (inject_src) {
        if (inject_backend) {
            ggml_backend_event_record(inject_src->ev_layer_inp_read.get(), inject_backend);
            inject_src->ev_layer_inp_read_recorded = true;
        } else {
            synchronize();
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);

        // [QLH-SPLIT] Say how big this landed and why: the tail of a chain used to allocate
        // n_vocab * n_batch here (~1.2 GiB at -b 2048), and nothing in the logs said so.
        //
        // Only on growth (the branch above), so this is not a per-request line: the buffer is
        // sized for the largest n_outputs_max seen so far and reused from then on. It is also
        // INFO on the library side, which maps to LOG_LEVEL_TRACE and therefore does not print
        // at llama-server's default -lv 3 at all -- ask for it with -lv 4 when you need the
        // number behind an out-of-memory report.
        LLAMA_LOG_INFO("%s: output buffer reallocated to %.02f MiB (n_outputs_max = %lld, n_vocab = %lld, logits = %s, embd = %s)\n",
                __func__, new_size / (1024.0 * 1024.0),
                (long long) n_outputs_max, (long long) n_vocab,
                has_logits ? "yes" : "no", has_embd ? "yes" : "no");
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il] && !cparams.embeddings_layer_inp_dev) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    if (cparams.embeddings_layer_inp_dev) {
        return extracted;
    }
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes() const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_KIMI_K31 || model.arch == LLM_ARCH_GLM5_NEXT ||
        model.arch == LLM_ARCH_HRM_TEXT) {
        res = std::max<uint32_t>(1024u, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_METIS16 ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_CLOVER_MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        model.arch == LLM_ARCH_XING4_0 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(1024u, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    return gf_res_prev[n_outputs > 0].get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        res->reset();
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.model       =*/ &model,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.layer_inp_dev =*/ layer_inp_dev,
        /*.layer_inp_src =*/ supports_embd_row() ? cparams.ctx_other->get_layer_inp_dev() : nullptr,
        /*.h_nextn_dev   =*/ cparams.embeddings_nextn_dev ? h_nextn_dev : nullptr,
        /*.h_nextn_src   =*/ supports_state_row() ? cparams.ctx_state_src->get_h_nextn_dev() : nullptr,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // the slot maps of the MoE cache
    if (lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph)) {
        return true;
    }

    auto & st = lctx->copy_experts;

    // the ids must be computed before the split starts, so only the first node of the split is considered
    if (ggml_graph_n_nodes(graph) == 0) {
        return false;
    }
    const ggml_tensor * node = ggml_graph_node(graph, 0);
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != dst) {
        return false;
    }

    const ggml_tensor * ids = node->src[2];
    if (ggml_nelements(ids) == 0) {
        return true;
    }

    const int64_t n_expert    = src->ne[2];
    const size_t  expert_size = src->nb[2];

    if (ids != st.ids || (int64_t) st.used.size() != n_expert) {
        st.ids_data.resize(ggml_nbytes(ids)/sizeof(int32_t));
        ggml_backend_tensor_get_async(backend, ids, st.ids_data.data(), 0, ggml_nbytes(ids));
        ggml_backend_synchronize(backend);

        st.used.assign(n_expert, false);
        for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
            for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                const int32_t id = st.ids_data[i1*ids->nb[1]/sizeof(int32_t) + i0*ids->nb[0]/sizeof(int32_t)];
                GGML_ASSERT(id >= 0 && id < n_expert);
                st.used[id] = true;
            }
        }

        st.ids = ids;
    }

    // group consecutive experts and copy them together
    for (int64_t first = 0; first < n_expert; ) {
        if (!st.used[first]) {
            first++;
            continue;
        }
        int64_t last = first;
        while (last + 1 < n_expert && st.used[last + 1]) {
            last++;
        }

        // the experts in the MoE cache are copied from device memory, the others are uploaded
        int64_t next = first;
        for (int64_t e = first; e <= last && lctx->moe_cache; ) {
            const int64_t n = lctx->moe_cache->copy_experts(backend, src, dst, e, last);
            if (n == 0) {
                e++;
                continue;
            }
            if (next < e) {
                ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + next*expert_size, next*expert_size, (e - next)*expert_size);
            }
            e   += n;
            next = e;
        }

        // copy a bit extra to ensure there are no NaNs in the padding of the last expert, this is necessary for MMQ in the CUDA backend
        const size_t offset  = next*expert_size;
        const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;
        const size_t size    = (last + 1 - next)*expert_size + padding;
        if (size > 0) {
            ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + offset, offset, size);
        }

        first = last + 1;
    }

    return true;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        if (winfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(winfos.size());
        offsets.reserve(winfos.size());
        sizes.reserve(winfos.size());
        data_offsets.reserve(winfos.size());

        for (const auto & winfo : winfos) {
            tensors.push_back(winfo.tensor);
            offsets.push_back(winfo.offset);
            sizes.push_back(winfo.size);
            data_offsets.push_back(winfo.data_offset);
        }

        std::vector<const ggml_tensor *> const_tensors(tensors.begin(), tensors.end());
        ggml_backend_tensor_get_n(const_tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), winfos.size());
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        if (rinfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(rinfos.size());
        offsets.reserve(rinfos.size());
        sizes.reserve(rinfos.size());
        data_offsets.reserve(rinfos.size());

        for (const auto & rinfo : rinfos) {
            tensors.push_back(rinfo.tensor);
            offsets.push_back(rinfo.offset);
            sizes.push_back(rinfo.size);
            data_offsets.push_back(rinfo.data_offset);
        }

        ggml_backend_tensor_set_n(tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), rinfos.size());
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    const uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            GGML_ASSERT(winfo.size % ggml_type_size(winfo.tensor->type) == 0);
            const int64_t n = winfo.size/ggml_type_size(winfo.tensor->type)*ggml_blck_size(winfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            std::vector<const ggml_tensor *> srcs(mbuf_cur.org.begin(), mbuf_cur.org.end());
            std::vector<ggml_tensor *> dsts(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
            ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.org.size());
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            GGML_ASSERT(rinfo.size % ggml_type_size(rinfo.tensor->type) == 0);
            const int64_t n = rinfo.size/ggml_type_size(rinfo.tensor->type)*ggml_blck_size(rinfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    std::vector<const ggml_tensor *> srcs(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
                    std::vector<ggml_tensor *> dsts(mbuf.org.begin(), mbuf.org.end());
                    ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.cpy.size());
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            std::vector<const ggml_tensor *> chunk_srcs;
            std::vector<ggml_tensor *> chunk_dsts;
            chunk_srcs.reserve(mbuf_cur.cpy.size() + mbuf.org.size());
            chunk_dsts.reserve(mbuf_cur.cpy.size() + mbuf.org.size());

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                GGML_ASSERT(n_copy % ggml_type_size(src_t->type) == 0);
                const int64_t n_el = n_copy / ggml_type_size(src_t->type) * ggml_blck_size(src_t->type);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                chunk_srcs.push_back(src_v);
                chunk_dsts.push_back(dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_backend_tensor_copy_n(chunk_srcs.data(), chunk_dsts.data(), chunk_srcs.size());

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

// [QLH-SPLIT] True when this context drives at least one stage whose state it cannot export.
//
// A chain with remote stages can only export the *checkpoint* flavour of its state
// (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY): that flavour is the recurrent state alone, a constant
// few MiB per stage, and it is what makes llama-server's prefix reuse work on a hybrid model.
// Anything else -- a whole-context state, or the full sequence state with the KV cells in it --
// would have to move the attention cache too, which grows with the prompt (22 MiB per stage at
// 4K measured) to save a re-prefill; that is deliberately not implemented.
//
// Callers must read 0 as "no state", not as "an empty state that restored successfully": both
// the server's prompt cache and its context checkpoints do the latter, which is why they are
// taught to skip an empty state (tools/server/server-context.cpp).
static bool split_state_unavailable(
        const std::vector<std::unique_ptr<llama_split_remote>> & remotes,
        llama_state_seq_flags flags) {
    if (remotes.empty()) {
        return false;
    }
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        return false;
    }

    // Once per process, not once per request: the server asks for a checkpoint on every prompt
    // chunk, and a per-call warning would drown everything else. call_once because the server
    // can ask from more than one thread.
    static std::once_flag once;
    std::call_once(once, [] {
        LLAMA_LOG_WARN("%s: this chain has remote stages, so only the checkpoint flavour of its "
                "sequence state (recurrent only) is exported; full states stay unavailable and "
                "the caller has to re-process the prompt instead\n", __func__);
    });
    return true;
}

size_t llama_context::state_get_size() {
    // Whole-context state, used by --slot-save-path. Same rule as the sequence state below: a
    // state that covers only the head is not this context's state, and restoring it would leave
    // the tails where they were. A remote chain cannot export it at all (no flags: this is the
    // full state).
    if (split_state_unavailable(split_remotes, 0)) {
        return 0;
    }

    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    // [QLH-SPLIT] A remote chain exports only the checkpoint flavour (see split_state_unavailable).
    // This is the answer llama-server's prompt cache and context checkpoints key on.
    if (split_state_unavailable(split_remotes, flags)) {
        return 0;
    }

    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        // A remote chain cannot be measured by writing it to a dummy io: state_write() would
        // have to fetch every stage's snapshot just to count the bytes. The split memory asks
        // the stages for their sizes instead, and the local part is still measured by counting.
        if (split_memory && !split_remotes.empty()) {
            return io.n_bytes() + split_memory->state_size(seq_id, flags);
        }

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    // [QLH-SPLIT] A remote chain has no whole-context state to write (see
    // split_state_unavailable). Writing the head alone would produce a file that looks like a
    // session and restores into a head whose tails never moved.
    if (split_state_unavailable(split_remotes, 0)) {
        return 0;
    }

    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    // [QLH-SPLIT] Same rule as the sequence state below: a context's state is the state of the
    // memory it exposes, and for a chain that is the union view (head + tails). `memory` alone
    // would save and restore only the head. No split_memory -- a whole-model context -- leaves
    // this exactly as it was.
    if (auto * mem = get_memory()) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        mem->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (auto * mem = get_memory()) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        mem->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // [QLH-SPLIT] The unsupported flavours still have to answer zero here as well as in
    // state_seq_get_size(): the server sizes the buffer from the getter, and a write that
    // silently emitted part of the chain would make the two disagree.
    if (split_state_unavailable(split_remotes, flags)) {
        return 0;
    }

    // [QLH-SPLIT] The sequence state of a chain is the state of every stage: a tail holds the
    // recurrent state that cannot be rewound, and a checkpoint of the head alone restores a
    // head whose positions disagree with the tails'. Use the union view, not `memory` (the
    // head), so this and get_memory() stay the same answer. A whole-model context has no
    // split_memory, so get_memory() is `memory` and nothing changes for it.
    auto * mem = get_memory();
    if (mem) {
        mem->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // [QLH-SPLIT] Mirror of state_seq_write_data: restore every stage, or the tails keep the
    // positions they had before the checkpoint and the next truncating seq_rm() fails there.
    auto * mem = get_memory();
    if (mem) {
        mem->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.no_alloc()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.moe_cache_cpu_share         =*/ 0.0f,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.embeddings_pre_norm         =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4 || model->arch == LLM_ARCH_DEEPSEEK41) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    const bool fp8_cache = params.type_k == GGML_TYPE_F8_E4M3 || params.type_v == GGML_TYPE_F8_E4M3;
    // Direct cache consumers do not use the shared attention scaling helpers.
    if (fp8_cache && (model->hparams.is_mla() ||
            model->arch == LLM_ARCH_DEEPSEEK32 ||
            model->arch == LLM_ARCH_DEEPSEEK4 ||
            model->arch == LLM_ARCH_DEEPSEEK41 ||
            model->arch == LLM_ARCH_GLM_DSA ||
            model->arch == LLM_ARCH_GLM5_NEXT ||
            model->arch == LLM_ARCH_DOTS3NOTE ||
            model->arch == LLM_ARCH_MISTRAL_LARGE4 ||
            model->arch == LLM_ARCH_HY_V4 ||
            model->arch == LLM_ARCH_MINIMAX_M3 ||
            model->arch == LLM_ARCH_QWEN4EXP ||
            model->arch == LLM_ARCH_DFLASH)) {
        LLAMA_LOG_ERROR("%s: FP8 cache is not supported for architecture '%s'\n", __func__, llm_arch_name(model->arch));
        return nullptr;
    }

    if (fp8_cache) {
        for (uint32_t il = 0; il < model->hparams.n_layer_all; ++il) {
            if (!model->hparams.has_kv(il)) {
                continue;
            }

            for (ggml_tensor * scale : { model->layers[il].k_cache_s, model->layers[il].v_cache_s }) {
                if (!scale || !scale->data) {
                    continue;
                }

                float value;
                ggml_backend_tensor_get(scale, &value, 0, sizeof(value));
                if (!std::isfinite(value) || value == 0.0f) {
                    LLAMA_LOG_ERROR("%s: FP8 cache scale '%s' must be finite and nonzero\n", __func__, ggml_get_name(scale));
                    return nullptr;
                }
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

// [QLH-SPLIT] C6 (docs/OPEN-ISSUES.md): no `extern "C"` here -- llama-ext.h already declares these
// with C linkage (and LLAMA_API), and a definition after such a declaration inherits it.
void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

bool llama_set_embeddings_nextn_dev(llama_context * ctx, bool value) {
    return ctx->set_embeddings_nextn_dev(value);
}

bool llama_get_embeddings_nextn_rows(llama_context * ctx, int64_t row0, int64_t n_rows, float * out) {
    return ctx->get_h_nextn_rows(row0, n_rows, out);
}

bool llama_set_embeddings_nextn_rows(llama_context * ctx, int64_t row0, int64_t n_rows, const float * data) {
    return ctx->set_h_nextn_rows(row0, n_rows, data);
}

bool llama_set_embd_state_src(llama_context * ctx, llama_context * src) {
    return ctx->set_state_src(src);
}

void llama_set_embeddings_layer_inp_dev(llama_context * ctx, bool value) {
    ctx->set_embeddings_layer_inp_dev(value);
}

bool llama_supports_embd_row(const llama_context * ctx) {
    return ctx->supports_embd_row();
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_split_attach(llama_context * ctx, llama_context ** stages, int32_t n_stages) {
    if (ctx == nullptr) {
        return -1;
    }
    return ctx->split_attach(stages, n_stages);
}

int32_t llama_split_n_stages(const llama_context * ctx) {
    return ctx == nullptr ? 0 : ctx->split_n_stages();
}

int32_t llama_split_attach_remote(llama_context * ctx, const char * const * endpoints,
                                  int32_t n_endpoints, char * err_out, size_t err_len) {
    if (ctx == nullptr || n_endpoints <= 0 || endpoints == nullptr) {
        return -1;
    }
    std::vector<std::string> eps;
    eps.reserve((size_t) n_endpoints);
    for (int32_t i = 0; i < n_endpoints; ++i) {
        if (endpoints[i] == nullptr) {
            return -1;
        }
        eps.emplace_back(endpoints[i]);
    }
    std::string err;
    const int n = ctx->split_attach_remote(eps, err);
    if (n < 0 && err_out != nullptr && err_len > 0) {
        snprintf(err_out, err_len, "%s", err.c_str());
    }
    return n;
}

// [QLH-SPLIT] The chain's health, for callers outside the library (common/common.cpp decides
// between abort and a failed request with it; the server reports it). Both answer from the view
// installed by split_attach_remote(), so a whole-model context and a local chain are always
// "fine": they have no remote stage whose state could have been left behind.
bool llama_context::split_failed() const {
    return split_memory && !split_memory->remote_failure.empty();
}

const char * llama_context::split_failure() const {
    return split_memory ? split_memory->remote_failure.c_str() : "";
}

// [QLH-SPLIT] v6: separate from split_failed() on purpose. A stale snapshot is not a chain
// failure: the stage refused the restore before touching its memory and the head is restored
// last, so every stage is still where it was. The caller (llama-server) recovers by re-processing
// the prompt, which is the same path a server without checkpoints takes. See llama-ext.h.
bool llama_context::split_state_stale() const {
    return split_memory && split_memory->state_stale;
}

bool llama_context_split_failed(const llama_context * ctx) {
    return ctx != nullptr && ctx->split_failed();
}

const char * llama_context_split_failure(const llama_context * ctx) {
    return ctx == nullptr ? "" : ctx->split_failure();
}

bool llama_context_split_state_stale(const llama_context * ctx) {
    return ctx != nullptr && ctx->split_state_stale();
}

// [QLH-SPLIT] v5 candidate channel, driving side. See llama-ext.h for what the channel is for and
// common/sampling.h for the rule that decides whether it may be used at all.
bool llama_context_split_has_remote(const llama_context * ctx) {
    return ctx != nullptr && ctx->split_has_remote();
}

void llama_context::split_request_candidates(const qlh_split::cand_req * req,
                                             const llama_logit_bias * biases, uint32_t n_bias,
                                             const llama_token * hist, uint32_t n_hist) {
    if (req == nullptr || split_remotes.empty()) {
        // Not an error: a local chain and a whole model both keep answering with the row, so a
        // caller that does not distinguish the chain kinds can call this unconditionally.
        split_cand_wanted = false;
        return;
    }

    split_cand_req = *req;
    // Filled from the vector sizes at send time (llama_split_remote::decode), so the count in the
    // plan cannot disagree with the bytes that follow it.
    split_cand_req.n_bias = 0;
    split_cand_req.n_hist = 0;

    split_cand_bias.clear();
    if (biases != nullptr && n_bias > 0) {
        split_cand_bias.assign(biases, biases + n_bias);
    }

    split_cand_hist.clear();
    if (hist != nullptr && n_hist > 0) {
        split_cand_hist.assign(hist, hist + n_hist);
    }

    split_cand_wanted = true;
}

uint32_t llama_context::split_take_candidates(const llama_split_token_cand ** cands, int32_t * row) const {
    if (cands != nullptr) {
        *cands = nullptr;
    }
    if (row != nullptr) {
        *row = -1;
    }
    if (split_remote_cands.empty()) {
        return 0;
    }
    // The wire structure and the exported one are the same two fields in the same order; the
    // static_assert is what keeps a future edit on one side from silently reinterpreting the
    // other.
    static_assert(sizeof(llama_split_token_cand) == sizeof(qlh_split::token_cand),
                  "llama_split_token_cand must match the wire layout of qlh_split::token_cand");
    static_assert(offsetof(llama_split_token_cand, id)    == offsetof(qlh_split::token_cand, id) &&
                  offsetof(llama_split_token_cand, logit) == offsetof(qlh_split::token_cand, logit),
                  "llama_split_token_cand must match the wire layout of qlh_split::token_cand");

    if (cands != nullptr) {
        *cands = reinterpret_cast<const llama_split_token_cand *>(split_remote_cands.data());
    }
    if (row != nullptr) {
        *row = split_remote_cands_row;
    }
    return (uint32_t) split_remote_cands.size();
}

void llama_split_request_candidates(
        llama_context * ctx,
        const qlh_split::cand_req * req,
        const llama_logit_bias * biases, uint32_t n_bias,
        const llama_token * hist, uint32_t n_hist) {
    if (ctx == nullptr) {
        return;
    }
    ctx->split_request_candidates(req, biases, n_bias, hist, n_hist);
}

uint32_t llama_split_take_candidates(
        const llama_context * ctx,
        const llama_split_token_cand ** cands,
        int32_t * row) {
    if (ctx == nullptr) {
        if (cands != nullptr) {
            *cands = nullptr;
        }
        if (row != nullptr) {
            *row = -1;
        }
        return 0;
    }
    return ctx->split_take_candidates(cands, row);
}

// [QLH-SPLIT] v7: per-row verify channel, driving side. See llama-ext.h for what it answers and
// common/sampling.h for the rule that decides whether it may be asked for at all.
void llama_context::split_request_verify_rows(bool on) {
    // Not an error for a local chain or a whole model: both keep answering with the row, and a
    // caller that does not distinguish the chain kinds must be able to call this unconditionally.
    split_verify_wanted = on && !split_remotes.empty();
}

llama_token llama_context::split_verified_token(int32_t idx) const {
    if (split_remote_verify.empty() || idx < 0 || (size_t) idx >= split_remote_verify.size()) {
        return LLAMA_TOKEN_NULL;
    }
    const int32_t id = split_remote_verify[(size_t) idx];
    // A negative id cannot be an argmax over a vocabulary, so it means the slot was never written
    // -- return it as "no answer" instead of letting the caller index a token table with -1.
    return id < 0 ? LLAMA_TOKEN_NULL : (llama_token) id;
}

void llama_split_request_verify_rows(llama_context * ctx, bool on) {
    if (ctx == nullptr) {
        return;
    }
    ctx->split_request_verify_rows(on);
}

llama_token llama_split_verified_token(const llama_context * ctx, int32_t idx) {
    return ctx == nullptr ? LLAMA_TOKEN_NULL : ctx->split_verified_token(idx);
}

// [QLH-SPLIT] v8: the remote MTP draft, driving side.
//
// The functions below are the whole head half of the feature. They exist as context methods
// because the state they touch (the armed flag, the rows of the last decode, the connections) is
// private; the exported C entry points after them are what common/speculative.cpp calls.
//
// Two shape decisions are worth stating once here, since every branch below depends on them:
//
//  - "ready" is asked of the stage that answered the handshake, never of the model in this process.
//    A remote chain's head model has no NextN layers at all (the loader keeps them for the tail
//    range only), so the only honest source is what OP_HELLO reported.
//  - every failure route ends in "no drafts". None of them fails a request and none of them marks
//    the chain dead: a draft is an optimization, and the answer without it is what the same request
//    would have produced without speculation at all.
bool llama_context::split_mtp_ready() const {
    if (split_remotes.empty()) {
        return false;
    }
    // One seq per batch on a remote chain, so the draft always belongs to the last stage; there is
    // no "which tail" question to answer here.
    return split_remotes.back()->mtp_ready();
}

void llama_context::split_request_nextn_rows(bool on) {
    // A local chain reads h_nextn rows in this process and a whole model has no tail to ask; both
    // ignore this the way they ignore the two channels above.
    split_nextn_wanted = on && !split_remotes.empty();
}

uint32_t llama_context::split_take_nextn(const float ** rows, int32_t * row0, uint32_t * n_embd) const {
    if (rows != nullptr) {
        *rows = nullptr;
    }
    if (row0 != nullptr) {
        *row0 = -1;
    }
    if (n_embd != nullptr) {
        *n_embd = 0;
    }
    if (split_remote_nextn.empty() || split_remote_nextn_embd == 0) {
        return 0;
    }
    if (rows != nullptr) {
        *rows = split_remote_nextn.data();
    }
    if (row0 != nullptr) {
        *row0 = split_remote_nextn_row0;
    }
    if (n_embd != nullptr) {
        *n_embd = split_remote_nextn_embd;
    }
    return (uint32_t) (split_remote_nextn.size() / split_remote_nextn_embd);
}

// [QLH-SPLIT] v9: the layer-input tap channel, driving side.
//
// The set is sticky (see llama-context.h): it names source-model layers, and every decode asks the
// remote stages for them until it is cleared. A whole model and a local chain clear it, because for
// them there is nothing to ask -- a local chain reads the same rows out of the context that owns the
// layers, which is what the head-side tap accessor in common/speculative.cpp is for.
void llama_context::split_set_layer_inp(const uint32_t * ids, uint32_t n) {
    split_tap_ids.clear();
    split_tap_warned = false;
    if (split_remotes.empty() || ids == nullptr || n == 0) {
        return;
    }
    if (n > qlh_split::MAX_LAYER_INP_TAPS) {
        // Not a refusal: the cap is a wire bound, and a draft model with more taps than the protocol
        // carries is a build mismatch. Ask for what fits and say so, so the drafts still work with
        // the taps that arrived instead of a request failing at the first decode.
        LLAMA_LOG_WARN("%s: %u layer-input taps requested, the protocol carries at most %u; the "
                       "draft will run with the first %u\n", __func__, n,
                       (unsigned) qlh_split::MAX_LAYER_INP_TAPS,
                       (unsigned) qlh_split::MAX_LAYER_INP_TAPS);
        n = qlh_split::MAX_LAYER_INP_TAPS;
    }
    split_tap_ids.assign(ids, ids + n);
    LLAMA_LOG_INFO("%s: asking the remote stages for %u layer-input taps\n", __func__, n);
}

const float * llama_context::split_take_layer_inp(uint32_t i, uint32_t * id, uint32_t * n_rows,
                                                  uint32_t * n_embd) const {
    if (id != nullptr) {
        *id = 0;
    }
    if (n_rows != nullptr) {
        *n_rows = 0;
    }
    if (n_embd != nullptr) {
        *n_embd = 0;
    }
    if (i >= split_remote_tap.size() || split_remote_tap_embd == 0 || split_remote_tap_rows == 0) {
        return nullptr;
    }
    const split_tap_block & block = split_remote_tap[i];
    if (id != nullptr) {
        *id = block.id;
    }
    if (n_rows != nullptr) {
        *n_rows = split_remote_tap_rows;
    }
    if (n_embd != nullptr) {
        *n_embd = split_remote_tap_embd;
    }
    return block.rows.data();
}

int llama_context::split_mtp_draft(const float * h, int32_t token, int32_t pos, uint32_t k,
                                   int32_t seq_id, int32_t * out, uint32_t * n_out) {
    if (n_out != nullptr) {
        *n_out = 0;
    }
    if (split_remotes.empty() || h == nullptr || out == nullptr) {
        return 0;
    }
    const uint32_t n_embd = split_remote_nextn_embd != 0 ? split_remote_nextn_embd
                                                        : (uint32_t) model.hparams.n_embd_out();
    std::vector<float> row(h, h + n_embd);
    std::vector<int32_t> ids;
    std::string err;
    const int n = split_remotes.back()->mtp_draft(row, token, pos, k, seq_id, ids, err);
    if (n <= 0) {
        if (!err.empty()) {
            LLAMA_LOG_WARN("%s: the tail could not draft (%s); this request continues without "
                           "speculation\n", __func__, err.c_str());
        }
        return n < 0 ? -1 : 0;
    }
    const uint32_t take = std::min<uint32_t>((uint32_t) n, k);
    std::memcpy(out, ids.data(), (size_t) take * sizeof(int32_t));
    if (n_out != nullptr) {
        *n_out = take;
    }
    return (int) take;
}

void llama_split_request_nextn_rows(llama_context * ctx, bool on) {
    if (ctx == nullptr) {
        return;
    }
    ctx->split_request_nextn_rows(on);
}

uint32_t llama_split_take_nextn_rows(
        const llama_context * ctx,
        const float ** rows,
        int32_t * row0,
        uint32_t * n_embd) {
    if (ctx == nullptr) {
        if (rows != nullptr) {
            *rows = nullptr;
        }
        if (row0 != nullptr) {
            *row0 = -1;
        }
        if (n_embd != nullptr) {
            *n_embd = 0;
        }
        return 0;
    }
    return ctx->split_take_nextn(rows, row0, n_embd);
}

bool llama_context_split_mtp_ready(const llama_context * ctx) {
    return ctx != nullptr && ctx->split_mtp_ready();
}

// [QLH-SPLIT] v9: the layer-input tap channel, exported. See llama-ext.h for what these answer.
void llama_split_set_layer_inp(llama_context * ctx, const uint32_t * ids, uint32_t n) {
    if (ctx == nullptr) {
        return;
    }
    ctx->split_set_layer_inp(ids, n);
}

uint32_t llama_split_take_layer_inp_n(const llama_context * ctx) {
    return ctx == nullptr ? 0 : ctx->split_take_layer_inp_n();
}

const float * llama_split_take_layer_inp(
        const llama_context * ctx,
        uint32_t i,
        uint32_t * id,
        uint32_t * n_rows,
        uint32_t * n_embd) {
    if (ctx == nullptr) {
        if (id != nullptr) {
            *id = 0;
        }
        if (n_rows != nullptr) {
            *n_rows = 0;
        }
        if (n_embd != nullptr) {
            *n_embd = 0;
        }
        return nullptr;
    }
    return ctx->split_take_layer_inp(i, id, n_rows, n_embd);
}

int32_t llama_context_split_mtp_draft(
        llama_context * ctx,
        const float * h,
        int32_t token,
        int32_t pos,
        uint32_t k,
        int32_t seq_id,
        llama_token * out,
        uint32_t * n_out) {
    if (ctx == nullptr || out == nullptr) {
        if (n_out != nullptr) {
            *n_out = 0;
        }
        return 0;
    }
    // llama_token is int32_t; the round trip through a vector keeps the wire type (int32) and the
    // caller's type apart, so a future change to either has to be made on purpose.
    std::vector<int32_t> ids(k);
    uint32_t n = 0;
    const int rc = ctx->split_mtp_draft(h, token, pos, k, seq_id, ids.data(), &n);
    if (rc > 0) {
        std::memcpy(out, ids.data(), (size_t) n * sizeof(llama_token));
    }
    if (n_out != nullptr) {
        *n_out = n;
    }
    return rc;
}

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
