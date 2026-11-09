#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;
    static const int TENSOR_READ_LAZY       = 1 << 5; // read rows on demand instead of loading whole tensor; requires mmap for now

    // [C-scheme] load source layers [layer_lo, layer_hi) and expose them as local layers 0..N-1.
    // Local blk.i is served from source blk.(i + layer_lo); block_count is overridden to hi-lo
    // and nextn_predict_layers to 0 at metadata level (see the constructor).
    int32_t layer_lo = 0;
    int32_t layer_hi = -1;

    // [C-scheme] the source model's trunk depth, read before block_count is overridden.
    // Needed to tell a tail segment (hi == total) from a middle one.
    //
    // Named n_layer_src, NOT n_layer_all: llama_hparams has an n_layer_all of its own and it
    // means the opposite thing for a range load -- the *local* block count (hi - lo, plus the
    // MTP block when this is the tail). Two fields with the same name and opposite meanings is
    // how "does this range reach the end of the model" gets answered with the wrong number.
    int32_t n_layer_src = 0;

    bool layer_range_active() const { return layer_hi > 0; }

    // [C-scheme] which globals this range actually reads; non-range loads keep both.
    bool needs_tok_embd() const { return !layer_range_active() || layer_lo == 0; }
    bool needs_output()   const { return !layer_range_active() || layer_hi >= n_layer_src; }

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;
    bool load_mtp;

    // [C-scheme] set when the active layer range ends at the model's last layer and the file
    // carries a NextN/MTP head. The range override forces load_mtp=false upstream, so the tail
    // range re-enables it here rather than relying on a constructor argument it cannot change.
    bool tail_keeps_mtp = false;

    // handle TENSOR_READ_LAZY
    // use case: keep PLE / engrams embd tensors on disk, read them on demand
    struct lazy_read {
        // set by the caller before the create_tensor() calls
        enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;

        // [C-scheme] ignore the AUTO size threshold in add(). Set when a layer range makes a
        // global tensor provably unread; those are ~0.5 GiB, well under the 4 GiB default, so
        // the size heuristic would otherwise refuse to make them lazy.
        bool force = false;

        // decide whether this tensor is read lazily
        // pass w to also record it, or nullptr to only ask
        bool add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w);

        bool any() const {
            return !ranges.empty();
        }

        bool has(const ggml_tensor * t) const {
            return tensors.count(ggml_get_name(t)) > 0;
        }

        const llama_mmap::ranges & for_file(uint32_t idx) const {
            static const llama_mmap::ranges none;

            const auto it = ranges.find(idx);
            return it == ranges.end() ? none : it->second;
        }

        // lazy tensors are gathered on the host, so no offload setting applies to them
        static ggml_backend_buffer_type_t buft();

    private:
        std::map<uint32_t, llama_mmap::ranges> ranges;
        std::set<std::string>                  tensors;
    } lazy;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;

    // [C-scheme] The file tensors this load takes, keyed by the weight they resolve to, for the
    // "skipped" log. size_done cannot stand in for them: it counts *created* tensors, and a tied
    // model creates token_embd twice (the embedding and its output alias) from one file range, so
    // subtracting it from the file total would subtract that range twice. And a tensor can be
    // created yet never read: under a layer range the globals this stage cannot use are mapped
    // lazily, which costs address space but no IO and no resident pages, so it belongs in neither
    // "loaded" nor "skipped".
    std::set<const llama_tensor_weight *> weights_loaded; // read into a buffer, or mapped and used
    std::set<const llama_tensor_weight *> weights_lazy;   // mapped but deliberately never faulted

    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    // lazy tensors need dedicated context
    struct ctx_key {
        ggml_backend_buffer_type_t buft;
        bool lazy;
    };

    struct ctx_key_comparator {
        bool operator()(const ctx_key & lhs, const ctx_key & rhs) const {
            if (lhs.lazy != rhs.lazy) {
                return lhs.lazy < rhs.lazy;
            }
            return strcmp(ggml_backend_buft_name(lhs.buft), ggml_backend_buft_name(rhs.buft)) < 0;
        }
    };

    std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
        int32_t param_split_layer_lo = -1,
        int32_t param_split_layer_hi = -1);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    // byte range within a mapped file
    struct mapping_range {
        size_t first; // inclusive
        size_t last;  // exclusive
    };

    // sorted, non-overlapping ranges of file 'idx' that back the tensors of 'ctx'
    // ranges owned by other contexts are excluded, so that a buffer wrapping them won't pin
    // data that this context never reads (lazy tensors, -cmoe tensors, ...)
    std::vector<mapping_range> get_mapping_ranges(uint32_t idx, ggml_context * ctx) const;

    // release a weight's mmap pages
    void unmap_weight(const llama_tensor_weight & w) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            std::vector<ggml_backend_buffer_t> & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
