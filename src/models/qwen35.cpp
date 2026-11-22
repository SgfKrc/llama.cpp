#include "models.h"
#include "llama-chip-emb.h"
#include "llama-memory-recurrent.h"

#include "gguf.h"

#include <cinttypes>

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr_layer(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            // ml.layer_lo, not hparams.layer_lo: this runs in load_arch_hparams(), which is before
            // llama-model.cpp copies the loader's range into hparams -- hparams.layer_lo is
            // still 0 here, so a range starting on a non-multiple of full_attn_interval
            // would classify every layer from 0 instead of from the segment's first source
            // layer (measured: --layer-range 33:40 asked for blk.2.ssm_conv1d.weight).
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1 + (uint32_t) ml.layer_lo) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    // A checkpoint may keep only the text rows of the token embedding and describe the
    // circuit-token rows with a small kv blob (see llama-chip-emb.h). The row count comes from the
    // file for the same reason the output head's does below: the file is the authority on how many
    // rows exist, and a full (or tied) checkpoint still creates the usual { n_embd, n_vocab }
    // tensor here, so its load path is unchanged.
    //
    // The chip rows are never materialised: the decoder rebuilds the rows a batch asks for (see
    // llama-graph.cpp, llm_graph_input_embd::set_input), which is what takes the 3.38 GiB out of
    // RAM/VRAM and not merely out of the file.
    int64_t n_vocab_embd = n_vocab;
    if (const struct ggml_tensor * t = ml.get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str())) {
        n_vocab_embd = t->ne[1];
    }
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab_embd }, 0);

    if (n_vocab_embd < n_vocab) {
        const struct gguf_context * kv = ml.metadata;

        auto kv_lookup = [&](const char * key) -> int64_t {
            return gguf_find_key(kv, key);
        };
        auto kv_u32 = [&](const char * key, bool required, uint32_t & out) -> bool {
            const int64_t kid = kv_lookup(key);
            if (kid < 0) {
                if (required) {
                    throw std::runtime_error(format("qwen35: missing key '%s' for a partial token embedding", key));
                }
                return false;
            }
            if (gguf_get_kv_type(kv, kid) != GGUF_TYPE_UINT32) {
                throw std::runtime_error(format("qwen35: key '%s' is not a uint32", key));
            }
            out = gguf_get_val_u32(kv, kid);
            return true;
        };
        auto kv_str = [&](const char * key, bool required, std::string & out) -> bool {
            const int64_t kid = kv_lookup(key);
            if (kid < 0) {
                if (required) {
                    throw std::runtime_error(format("qwen35: missing key '%s' for a partial token embedding", key));
                }
                return false;
            }
            if (gguf_get_kv_type(kv, kid) != GGUF_TYPE_STRING) {
                throw std::runtime_error(format("qwen35: key '%s' is not a string", key));
            }
            out = gguf_get_val_str(kv, kid);
            return true;
        };
        auto kv_u8_array = [&](const char * key, bool required) -> std::vector<uint8_t> {
            const int64_t kid = kv_lookup(key);
            if (kid < 0) {
                if (required) {
                    throw std::runtime_error(format("qwen35: missing key '%s' for a partial token embedding", key));
                }
                return {};
            }
            if (gguf_get_kv_type(kv, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(kv, kid) != GGUF_TYPE_UINT8) {
                throw std::runtime_error(format("qwen35: key '%s' is not an array of uint8", key));
            }
            const size_t n = gguf_get_arr_n(kv, kid);
            const void * data = gguf_get_arr_data(kv, kid);
            if (n > 0 && data == nullptr) {
                throw std::runtime_error(format("qwen35: key '%s' has no data", key));
            }
            const uint8_t * p = (const uint8_t *) data;
            return std::vector<uint8_t>(p, p + n);
        };

        std::string format_name;
        kv_str("qwen35.chip_emb_format", true, format_name);
        if (format_name != "rv1-chip-emb-v1") {
            throw std::runtime_error(format("qwen35: unsupported chip embedding format '%s'", format_name.c_str()));
        }

        uint32_t embedded_off = 0, n_chip = 0, n_kernel_rows = 0, n_pad_rows = 0;
        kv_u32("qwen35.chip_emb_offset",      true,  embedded_off);
        kv_u32("qwen35.chip_emb_rows",        true,  n_chip);
        kv_u32("qwen35.chip_emb_kernel_rows", false, n_kernel_rows);
        kv_u32("qwen35.chip_emb_pad_rows",    false, n_pad_rows);

        if ((int64_t) embedded_off != n_vocab_embd) {
            throw std::runtime_error(format(
                    "qwen35: token_embd has %" PRId64 " rows but the chip rows start at %u",
                    n_vocab_embd, embedded_off));
        }
        if (n_vocab_embd + n_chip + n_kernel_rows + n_pad_rows != (int64_t) n_vocab) {
            throw std::runtime_error(format(
                    "qwen35: %" PRId64 " stored + %u chip + %u kernel + %u pad rows do not add up to the %" PRId64 " tokens of the vocabulary",
                    n_vocab_embd, n_chip, n_kernel_rows, n_pad_rows, n_vocab));
        }

        // The rebuild writes bf16 bit patterns, so a quantised table cannot take this path.
        if (tok_embd->type != GGML_TYPE_BF16) {
            throw std::runtime_error(format(
                    "qwen35: a partial token embedding needs bf16 rows, this table is %s",
                    ggml_type_name(tok_embd->type)));
        }

        const std::vector<uint8_t> blob   = kv_u8_array("qwen35.chip_emb_meta",   true);
        const std::vector<uint8_t> kernel = kv_u8_array("qwen35.chip_emb_kernel", false);

        auto chip = std::make_unique<llama_chip_emb>();
        std::string err;
        if (!chip->parse(blob, kernel, n_vocab_embd, n_vocab, n_embd, err)) {
            throw std::runtime_error("qwen35: " + err);
        }
        LLAMA_LOG_INFO("%s: token_embd has %" PRId64 " rows in the file; %" PRId64
                " circuit rows rebuilt on demand from a %zu-byte blob (+ %" PRId64 " kernel, %" PRId64
                " pad rows)\n", __func__, chip->n_stored, chip->n_chip, blob.size(), chip->n_kernel, chip->n_pad);
        chip_emb = std::move(chip);
    }

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    {
        // A split head stores only the rows the model can actually emit. The reference
        // implementation samples text positions with argmax(logits[:v0]), v0 = 248320 of 642496,
        // so the 394176 circuit-token rows are unreachable from a text position -- storing them
        // costs a 5.52 GiB bf16 matvec read on every token and buys nothing. The row count comes
        // from the file, not from n_vocab: a tied (or full) checkpoint still creates the usual
        // { n_embd, n_vocab } tensor here, so this load path is unchanged for it.
        //
        // The graph pads the narrower result back to n_vocab (llama_model_qwen35::graph::graph),
        // because the context and the samplers index logits by n_vocab regardless.
        int64_t n_vocab_out = n_vocab;
        if (const struct ggml_tensor * t = ml.get_tensor_meta(tn(LLM_TENSOR_OUTPUT, "weight").str().c_str())) {
            n_vocab_out = t->ne[1];
        }
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab_out }, TENSOR_NOT_REQUIRED);
    }

    // optional projection of the embeddings output
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_embd_out() }, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   { hparams.n_embd_out() },         TENSOR_NOT_REQUIRED);

    // k0v4: the chip layers' sparse store, five flat 1-D tensors (K0V4-SPARSE-FORMAT.md §2).
    //
    // They have to be *created*, not merely read: llama_model_loader counts what it created, and a
    // full (non-partial) load throws "wrong number of tensors; expected N, got N-5" for any tensor
    // in the file that no create_tensor() claimed. That is also why they cannot be smuggled in as
    // kv: kv has no shape and no backend buffer, and this path is what gives the kernel a device
    // pointer (`-ot 'chip_sparse\.=CUDA0'`).
    //
    // The lengths come from the file, like token_embd's and output's row counts above: mat_base,
    // row_ptr, col and w are sized by the circuit layout (78 matrices / 520000 rows / 1611555
    // non-zeros today), so hard-coding them here would make a data regeneration a loader rebuild.
    // TENSOR_NOT_REQUIRED keeps every checkpoint without the store on the unchanged path: the
    // pointers stay null and nothing in the dense graph reads them.
    //
    // Nothing in the graph consumes them yet -- the sparse matvec (GGML_OP_CHIP_SPARSE_MM) and the
    // five build_lora_mm substitutions are a separate change (_s4a/REPORT.md §7) -- so this load
    // leaves the dense chip weights in place and changes no number.
    {
        static const llm_tensor chip_tensor_ids[5] = {
            LLM_TENSOR_CHIP_SPARSE_MAT_BASE, LLM_TENSOR_CHIP_SPARSE_MAT_COLS,
            LLM_TENSOR_CHIP_SPARSE_ROW_PTR,  LLM_TENSOR_CHIP_SPARSE_COL,
            LLM_TENSOR_CHIP_SPARSE_W,
        };
        // Fill locals first and only publish all five at the end: a half-filled store must never be
        // visible to a consumer, and a member that was never written must never be mistaken for
        // "present" (the first version tested the members directly and rejected every checkpoint
        // without a store, because the test could not tell "not created" from "not there").
        ggml_tensor * chip_got[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
        int64_t chip_bytes = 0;
        for (int i = 0; i < 5; ++i) {
            const auto tn_chip = tn(chip_tensor_ids[i]);
            // a 1-D tensor: ne[0] is its length, and the file is the authority on that
            const struct ggml_tensor * meta = ml.get_tensor_meta(tn_chip.str().c_str());
            if (meta == nullptr) {
                continue;   // not a k0v4 checkpoint -- leave the slot null
            }
            if (meta->ne[1] != 1) {
                throw std::runtime_error(format(
                        "qwen35: %s must be a flat 1-D tensor, the file stores %" PRId64
                        " rows", tn_chip.str().c_str(), meta->ne[1]));
            }
            chip_got[i] = create_tensor(ml, tn_chip, { meta->ne[0] }, TENSOR_NOT_REQUIRED);
            if (chip_got[i] != nullptr) {
                chip_bytes += ggml_nbytes(chip_got[i]);
            }
        }
        int n_chip = 0;
        for (int i = 0; i < 5; ++i) {
            n_chip += chip_got[i] != nullptr;
        }
        if (n_chip == 5) {
            chip_sparse_mat_base = chip_got[0];
            chip_sparse_mat_cols = chip_got[1];
            chip_sparse_row_ptr  = chip_got[2];
            chip_sparse_col      = chip_got[3];
            chip_sparse_w        = chip_got[4];
            // 78 matrices; mat_base has n_mat + 1 entries (the last one is the last matrix' first
            // row, so row_ptr[mat_base.back()] == nnz), mat_cols has n_mat.
            LLAMA_LOG_INFO("%s: chip sparse store: %" PRId64 " matrices, %" PRId64
                    " rows + 1 pointers, %" PRId64 " non-zeros, %" PRId64 " B = %.3f MiB "
                    "(loaded, not consumed by the dense graph)\n",
                    __func__, chip_sparse_mat_cols->ne[0], chip_sparse_row_ptr->ne[0] - 1,
                    chip_sparse_col->ne[0], chip_bytes, chip_bytes / (double) (1 << 20));
        } else if (n_chip != 0) {
            // Half a store is worse than none: a consumer would read col/w through a row_ptr (or
            // mat_cols) that describes some other layout, and the CSR indices are the one thing a
            // wrong guess cannot survive.
            std::string got;
            for (int i = 0; i < 5; ++i) {
                got += std::string(chip_got[i] ? " +" : " -") + tn(chip_tensor_ids[i]).str();
            }
            throw std::runtime_error(format(
                    "qwen35: the chip sparse store is incomplete (mat_base/mat_cols/row_ptr/col/w "
                    "must all be present):%s", got.c_str()));
        }
    }

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        if (chip_emb) {
            // A tied head is the token embedding itself, so it needs every row of the vocabulary;
            // this checkpoint dropped the rows the tied head would have to emit.
            throw std::runtime_error(
                    "qwen35: a checkpoint with programmatic chip rows must also store a separate "
                    "output.weight (the tied head is the full-vocabulary table)");
        }
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            const int separate_flags = flags & TENSOR_SKIP ? flags | TENSOR_NOT_REQUIRED : flags;
            layer.wqkv_z = create_tensor(tn(LLM_TENSOR_ATTN_QKV_GATE, "weight", il), { n_embd, key_dim * 2 + value_dim * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wqkv_z) {
                layer.wqkv      = create_tensor(tn(LLM_TENSOR_ATTN_QKV,  "weight", il), { n_embd, key_dim * 2 + value_dim }, separate_flags);
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, separate_flags);
            }
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta_alpha_cat = create_tensor(tn(LLM_TENSOR_SSM_BETA_ALPHA_CAT, "weight", il), { n_embd, n_v_heads * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.ssm_beta_alpha_cat) {
                layer.ssm_beta  = create_tensor(tn(LLM_TENSOR_SSM_BETA,  "weight", il), { n_embd, n_v_heads }, separate_flags);
                layer.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, n_v_heads }, separate_flags);
            }
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

llama_model_qwen35::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // The hidden exit (t_layer_inp[n_layer]) needs every token row, but
    // crop_before_nextn trims the last layer down to n_outputs rows. A caller that uses
    // the layer_inp channel only takes the hidden and no logits, so n_outputs == 0 and
    // the last layer is cropped to a [n_embd, 0] empty tensor -- which makes
    // extract_layer_inputs' row_floats == n_embd assert fail right away. Skip the crop only
    // when the slot is armed: skipping it unconditionally would leave inp_out_ids unused and
    // its buffer unallocated, so llm_graph_input_out_ids::set_input would hit
    // GGML_ASSERT(buffer) instead.
    const bool qlh_keep_full_rows =
        n_layer < (int) cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer];

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids) && !qlh_keep_full_rows) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Register the last layer's output as t_layer_inp[n_layer].
    // The loop above only fills t_layer_inp[0..n_layer-1] (each entry is that layer's
    // *input*). A relay segment's "layer n_layer input" is precisely this tensor, and
    // DFlash's whole-head path needs it: without this entry, layer_inp[n_layer] stays
    // null and llama_set_embeddings_layer_inp(n_layer) aborts in extract_layer_inputs.
    // Falling back to the embeddings channel instead is NOT equivalent -- measured
    // max|diff| 0.227 against layer_inp[32] on the same prompt, which is enough to
    // poison the draft's eight-way concat.
    res->t_layer_inp[n_layer] = cur;

    // The crop was skipped so the hidden exits keep every token row. inp_out_ids
    // needs a consumer or its buffer is never allocated, and on the relay path nothing below
    // expands the norm chain, so expand the crop itself.
    if (qlh_keep_full_rows && crop_before_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        ggml_build_forward_expand(gf, cur);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    // A relay segment has no use for logits; leaving t_logits null keeps the
    // output projection (and its ~515 MB of weights) out of the graph entirely. A non-tail range
    // never loads output.weight (llama_model_loader::needs_output), so only a range that reaches
    // the source model's last layer can build the head.
    //
    // This is the fourth spelling of "is this range a relay" and the only one left outside
    // qlh_layer_range.h: it is exactly qlh_layer_range::needs_output(layer_lo, layer_hi,
    // n_layer_src), i.e. llama_hparams::needs_output(). Spelled out here only because a graph
    // builder has no business calling a loader/attach helper; if the three predicates there ever
    // change, this line changes with them.
    const bool needs_logits = !(hparams.layer_hi >= 0 && hparams.layer_hi < hparams.n_layer_src);
    if (needs_logits) {
        // The crop above was skipped so the hidden exit keeps every row; the
        // LM head still only needs the output rows.
        if (qlh_keep_full_rows && inp_out_ids != nullptr && cur->ne[1] != inp_out_ids->ne[0]) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        cur = build_lora_mm(model.output, cur, model.output_s);

        // A split head (see load_arch_tensors) is narrower than the vocabulary: it holds only the
        // rows a text position can emit. Everything downstream -- output_reserve(), get_logits_ith(),
        // the samplers -- strides logits by hparams.n_vocab, so pad the missing tail rows with 0
        // here. The padding rows are the rows the split dropped, so no reachable logit changes; a
        // full head does not pad at all (ne[1] == n_vocab), which keeps the tied path bit-identical.
        if (model.output->ne[1] < model.vocab.n_tokens()) {
            cur = ggml_pad(ctx0, cur, model.vocab.n_tokens(), cur->ne[1], cur->ne[2], cur->ne[3]);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_z) {
        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    if (model.layers[il].ssm_beta_alpha_cat) {
        ggml_tensor * ba = build_lora_mm(model.layers[il].ssm_beta_alpha_cat, cur);
        cb(ba, "beta_alpha", il);

        beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);

        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        if (model.chip_emb && tok_embd_w == model.tok_embd) {
            // The main table stores only the text rows: the MTP block's token rows are produced on
            // the host like the main graph's (see llm_graph_input_embd::embd_tok). A dedicated
            // nextn.embed_tokens table is a full one and keeps the gather path.
            inp->embd_tok = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
            ggml_set_input(inp->embd_tok);
            ggml_set_name(inp->embd_tok, "mtp_tok_embd_rows");

            inp->tok_embd_w = tok_embd_w;
            inp->chip_emb   = model.chip_emb.get();

            tok_embd = inp->embd_tok;
        } else {
            tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
        }

        // a Hadamard-latent table stores rotated rows; restore the primal basis
        if (hdmd) {
            if (const auto it = hdmd->inv.find(tok_embd_w); it != hdmd->inv.end()) {
                tok_embd = llama_mul_mat_hadamard(ctx0, tok_embd, it->second.rot);
                if (it->second.signs) {
                    tok_embd = ggml_mul(ctx0, tok_embd, it->second.signs);
                }
            }
        }
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
