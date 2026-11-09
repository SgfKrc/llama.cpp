#pragma once

#include "slim-arc-expert-residency.h"
#include "slim-arc-expert-transition.h"
#include "slim-arc-page-range.h"

#include "ggml-backend.h"

// SLIM-ARC: Tensor-level asynchronous prefetch scheduler
//
// This module implements layer-ahead prefetch on top of upstream llama.cpp's
// mmap infrastructure. When computing layer N, it asynchronously issues
// posix_madvise(WILLNEED) for tensors in layers N+1..N+window, allowing the
// kernel to overlap I/O with computation.

#include <atomic>
#include <climits>  // SLIM-ARC FIX 2026-08-05: guard against cascading INT_MAX errors
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace slim_arc {

// SLIM-ARC FIX 2026-08-05: compute-phase enum, consumed by set_phase() in graph_compute.
// When unified_io_scheduler is not enabled, prefetch_scheduler uses this phase info on
// its own.
enum class compute_phase {
    PREFILL,
    DECODE,
};

struct tensor_prefetch_info {
    void *   addr;      // mmap address of tensor data
    size_t   size;      // tensor data size in bytes
    int      layer;     // layer index (or -1 for non-layer tensors)
    uint64_t signature; // monotonic counter to detect graph changes
};

// SLIM-ARC FIX 2026-08-05: MoE expert tensor registration info (merged 3D tensor
// *_exps). addr points at the whole 3D tensor data, n_experts = ne[2], and each expert
// is size/n_experts bytes.
struct expert_tensor_info {
    void * addr;        // mmap address of the merged 3D expert tensor
    size_t size;        // total bytes of the merged tensor
    int    n_experts;   // number of experts (ne[2])
    int    file_id{-1};
    uint64_t file_offset{0};
};

struct expert_file_advice_stats {
    uint64_t calls{0};
    uint64_t issued_bytes{0};
    uint64_t failures{0};
    uint64_t invalid_ranges{0};
};

struct prefetch_budget_stats {
    uint64_t requested_bytes{0};
    uint64_t issued_bytes{0};
    uint64_t skipped_bytes{0};
    uint64_t rounds_throttled{0};
    uint64_t madvise_failures{0};
    uint64_t advice_requests{0};
    uint64_t coalesced_ranges{0};
    uint64_t covered_bytes{0};
    uint64_t invalid_ranges{0};
    uint64_t stale_requests{0};
    uint64_t stale_bytes{0};
    uint64_t inflight_peak_bytes{0};
};

struct expert_reclaim_stats {
    uint64_t candidate_experts{0};
    uint64_t calls{0};
    uint64_t reclaimed_bytes{0};
    uint64_t skipped_bytes{0};
    uint64_t madvise_failures{0};
    uint64_t invalid_layouts{0};
    uint64_t invalid_ids{0};
};

struct expert_residency_runtime_stats {
    uint64_t samples{0};
    uint64_t admitted_experts{0};
    uint64_t admitted_bytes{0};
    uint64_t skipped_bytes{0};
    uint64_t fallbacks{0};
    uint64_t pressure_missing{0};
    uint64_t pressure_normal{0};
    uint64_t pressure_high{0};
    uint64_t pressure_critical{0};
};

struct expert_runtime_metrics {
    uint64_t samples{0};
    uint64_t issued_bytes{0};
    uint64_t hit_bytes{0};
    uint64_t waste_bytes{0};
    uint64_t advice_requests{0};
    uint64_t coalesced_ranges{0};
    uint64_t covered_bytes{0};
    uint64_t advice_failures{0};
    uint64_t invalid_ranges{0};
};

struct expert_hot_cache_stats {
    uint64_t budget_bytes{0};
    uint64_t locked_bytes{0};
    uint64_t admissions{0};
    uint64_t hits{0};
    uint64_t evictions{0};
    uint64_t budget_rejections{0};
    uint64_t nonresident_bytes{0};
    uint64_t lock_failures{0};
    uint64_t entries{0};
    uint64_t admission_skips{0};
    uint64_t admission_threshold{1};
};

std::vector<size_t> select_prefetch_items(
    const std::vector<size_t> & item_sizes,
    uint64_t budget_bytes,
    uint64_t * requested_bytes,
    uint64_t * skipped_bytes);

class prefetch_scheduler {
  public:
    using advice_fn = std::function<int(void *, size_t, int)>;
    using page_size_query_fn = std::function<long()>;
    using file_advice_fn = std::function<int(int, uint64_t, size_t)>;

    explicit prefetch_scheduler(
        int n_threads = 2,
        int window = 3,
        std::function<void()> request_claim_hook = {},
        advice_fn advice = {},
        page_size_query_fn page_size_query = {},
        file_advice_fn file_advice = {});
    ~prefetch_scheduler();
    prefetch_scheduler(const prefetch_scheduler &) = delete;
    prefetch_scheduler & operator=(const prefetch_scheduler &) = delete;
    void shutdown() noexcept;

    // Register a tensor for potential prefetch. Called during model load.
    void register_tensor(const char * name, void * addr, size_t size, int layer);
    bool register_mapping(void * addr, size_t size, int file_id = -1);

    // Notify that we are about to compute layer `current_layer`.
    // This triggers async madvise(WILLNEED) for layers
    // [current_layer+1, current_layer+window].
    void notify_layer_compute(int current_layer);

    // ---- SLIM-ARC 2026-11-02: real per-layer signalling ----
    // The graph is submitted to ggml_backend_sched as a whole (~3.8k nodes covering all layers),
    // so llama_context::graph_compute has no "we are now computing layer N" moment. The signal
    // is taken from the per-node eval callback instead: prepare_graph() scans the built graph
    // once and remembers the first node of every layer, eval_cb() then fires
    // notify_layer_compute() for the layers it observes.
    //
    // Call after the graph is built (the layer index lives in the tensor name, which
    // llama_context::graph_get_cb() assigns while building). Idempotent for the same graph.
    void prepare_graph(const ggml_cgraph * gf);

    // Chain the user callback (llama_context_params::cb_eval) so that installing eval_cb does
    // not disable it. Must be called before the callback is handed to the scheduler.
    void attach_user_callback(ggml_backend_sched_eval_callback cb, void * user_data);

    // ggml_backend_sched_eval_callback body. user_data must be a prefetch_scheduler *.
    static bool eval_cb(ggml_tensor * t, bool ask, void * user_data);

    // Disable prefetch (e.g., when memory budget exceeded).
    void set_enabled(bool enabled) { enabled_.store(enabled); }

    // Collect statistics
    size_t total_prefetched_bytes() const { return total_bytes_.load(); }
    int    total_prefetch_calls()   const { return total_calls_.load(); }

    // ---- SLIM-ARC FIX 2026-08-05: interfaces the Pi5-side fix needs ----
    // Set the compute phase (prefill / decode).
    // SLIM-ARC FIX 2026-08-07: triggers the dynamic MADV switch (PREFILL->SEQUENTIAL /
    // DECODE->RANDOM).
    void set_phase(compute_phase phase);
    // Current prefetch window size (used by graph_compute to compute the layer range to
    // prefetch).
    int  effective_window() const { return window_; }
    // Memory budget (used by unified_io_scheduler to apportion the weight-prefetch quota).
    void set_memory_budget(size_t bytes) { memory_budget_.store(bytes); }
    prefetch_budget_stats budget_stats() const;

    // Phase 2a: register MoE expert tensors (merged 3D tensor *_exps)
    void register_expert_tensor(const char * name, void * addr, size_t size, int layer, int n_experts);
    // Cache the expert IDs the router selected for this layer (for cross-layer prediction)
    void cache_router_experts(int layer, const int * expert_ids, int n);
    // Settle the given prefetch generation; generation 0 only updates the predictor and
    // settles no metrics.
    void cache_router_experts(int layer, const int * expert_ids, int n, uint64_t generation);
    // Abort the given prefetch generation and count its successfully issued bytes as
    // waste in one go.
    void cancel_expert_prefetch(int layer, uint64_t generation);
    // Get an independent copy of the router expert IDs most recently cached for a layer.
    std::vector<int> cached_experts_snapshot(int layer) const;
    // Issue WILLNEED prefetch for the given experts of the specified layer and return a
    // generation that must not be reused. 0 means there was no successful prefetch to
    // settle.
    uint64_t prefetch_experts(int layer, const int * expert_ids, int n);

    // ---- SLIM-ARC FIX 2026-08-09: expert-prefetch observability metrics (improvement 1:
    // make it observable before optimizing) ----
    // Print the hit-rate / wasted-bytes summary (called from the destructor)
    void dump_metrics() const;
    size_t expert_prefetch_bytes() const { return expert_prefetch_bytes_.load(); }
    size_t expert_hit_bytes()     const { return expert_hit_bytes_.load(); }
    size_t expert_waste_bytes()   const { return expert_waste_bytes_.load(); }
    size_t pending_expert_records(int layer) const;
    expert_reclaim_stats expert_reclaim_statistics() const;
    bool expert_residency_enabled() const noexcept { return expert_residency_enabled_; }
    void set_expert_residency_pressure(expert_pressure_state pressure, size_t budget_bytes);
    expert_pressure_state current_expert_pressure() const;
    expert_residency_runtime_stats expert_residency_statistics() const noexcept;
    expert_runtime_metrics expert_runtime_statistics() const noexcept;
    expert_hot_cache_stats expert_hot_cache_statistics() const noexcept;
    expert_file_advice_stats expert_file_advice_statistics() const noexcept;
    bool expert_hot_lfru_enabled() const noexcept { return expert_hot_lfru_enabled_; }
    std::vector<uint32_t> expert_popularity_snapshot(int layer) const;
    uint64_t popularity_decay_count() const noexcept { return popularity_decay_count_.load(); }
    uint32_t expert_waste_ewma_milli() const;
    uint64_t expert_waste_sample_count() const;
    bool expert_waste_restricted() const;
    size_t pending_request_count() const;
    uint64_t dropped_request_count() const { return dropped_requests_.load(); }
    bool confidence_gating_enabled() const { return conf_gating_; }
    int popularity_k() const { return pop_k_; }
    bool cross_layer_transition_enabled() const noexcept { return cross_layer_transition_enabled_; }
    int cross_layer_transition_topk() const noexcept { return cross_layer_transition_topk_; }
    void observe_expert_transition(
        int layer,
        const std::vector<int> & source,
        const std::vector<int> & target);
    std::vector<int> predict_expert_transition(int layer, const std::vector<int> & source);
    void record_expert_transition_result(
        const std::vector<int> & predicted,
        const std::vector<int> & actual);
    expert_transition_stats expert_transition_statistics() const noexcept;
    // Unified I/O budget issue and per-step reset (improvement 3, called from
    // unified_io_scheduler::tick)
    void set_expert_budget(size_t bytes) {
        expert_budget_.store(bytes);
        expert_budget_enabled_.store(true);
    }
    void reset_expert_budget_usage() { expert_budget_used_.store(0); }

  private:
    void worker_loop();
    uint64_t issue_expert_willneed(int layer, const int * expert_ids, int n);
    void reclaim_wrong_expert_pages(
        const std::vector<int> & prefetched,
        const std::vector<int> & selected,
        const std::vector<expert_tensor_info> & tensors);
    void update_expert_hot_cache(
        int layer,
        const std::vector<int> & stable_experts,
        const std::vector<expert_tensor_info> & tensors);
    void advise_selected_expert_files(
        const std::vector<int> & selected,
        const std::vector<expert_tensor_info> & tensors);

    const bool slow_storage_enabled_;
    const bool router_prefetch_enabled_;
    const bool router_mlock_enabled_;
    const bool shared_mlock_enabled_;
    const bool small_mlock_enabled_;
    const size_t expert_hot_budget_bytes_;
    const bool expert_hot_lru_enabled_;
    const bool expert_hot_lfru_enabled_;
    const uint32_t expert_hot_admission_threshold_;
    const bool expert_prefetch_disabled_;
    const bool expert_file_advice_enabled_;
    const bool expert_random_madv_enabled_;
    const bool expert_normal_madv_enabled_;
    const int cross_layer_transition_topk_;
    const bool cross_layer_transition_enabled_;
    int n_threads_;
    int window_;
    std::atomic<compute_phase> phase_{compute_phase::DECODE};
    // [QLH] unified_io_scheduler (the module that used to hand this scheduler its byte budget)
    // is not ported, so nothing calls set_memory_budget() and the upstream default of 0 truncated
    // every request to zero bytes. The constructor now derives a ceiling from the free physical
    // memory, and plan_request() additionally caps each request at one layer of weights.
    // SLIM_ARC_BUDGET_MB overrides the ceiling; set_memory_budget() still wins if a caller shows up.
    std::atomic<size_t> memory_budget_{0};
    std::atomic<bool>       enabled_{true};
    std::atomic<bool>       stop_{false};
    std::atomic<size_t>     total_bytes_{0};
    std::atomic<int>        total_calls_{0};
    std::atomic<uint64_t>   budget_requested_bytes_{0};
    std::atomic<uint64_t>   budget_issued_bytes_{0};
    std::atomic<uint64_t>   budget_skipped_bytes_{0};
    std::atomic<uint64_t>   budget_rounds_throttled_{0};
    std::atomic<uint64_t>   budget_madvise_failures_{0};
    std::atomic<uint64_t>   budget_advice_requests_{0};
    std::atomic<uint64_t>   budget_coalesced_ranges_{0};
    std::atomic<uint64_t>   budget_covered_bytes_{0};
    std::atomic<uint64_t>   budget_invalid_ranges_{0};
    std::atomic<uint64_t>   budget_stale_requests_{0};
    std::atomic<uint64_t>   budget_stale_bytes_{0};
    std::atomic<uint64_t>   budget_inflight_bytes_{0};
    std::atomic<uint64_t>   budget_inflight_peak_bytes_{0};

    // ---- SLIM-ARC FIX 2026-08-09: expert-prefetch observability metrics (improvement 1) ----
    std::atomic<size_t>     expert_prefetch_bytes_{0};  // bytes actually issued as WILLNEED
    std::atomic<size_t>     expert_hit_bytes_{0};       // prefetched bytes used by the next token
    std::atomic<size_t>     expert_waste_bytes_{0};     // prefetched bytes never used
    std::atomic<uint64_t>   router_samples_{0};         // number of samples
    std::atomic<uint64_t>   expert_advice_requests_{0};
    std::atomic<uint64_t>   expert_coalesced_ranges_{0};
    std::atomic<uint64_t>   expert_covered_bytes_{0};
    std::atomic<uint64_t>   expert_advice_failures_{0};
    std::atomic<uint64_t>   expert_invalid_ranges_{0};

    std::vector<std::thread>          workers_;
    std::function<void()>             request_claim_hook_;
    const advice_fn                   advice_;
    const page_size_query_fn          page_size_query_;
    const file_advice_fn              file_advice_;
    const bool                        reclaim_waste_enabled_;
    const bool                        expert_residency_enabled_;
    std::mutex                        shutdown_mtx_;
    mutable std::mutex                mtx_;
    std::condition_variable           cv_;
    struct prefetch_request {
        uint64_t generation{0};
        int layer{-1};
        uint64_t memory_budget{0};
        uint64_t requested_bytes{0};
        uint64_t advice_requests{0};
        uint64_t invalid_ranges{0};
        uint64_t covered_bytes{0};
        size_t page_size{0};
        std::vector<page_range> ranges;
    };
    prefetch_request plan_request(int current_layer);
    std::deque<prefetch_request>      pending_requests_;
    uint64_t                          next_request_generation_{0};
    static constexpr size_t           max_pending_requests{64};
    std::atomic<uint64_t>             dropped_requests_{0};

    // tensor registry indexed by layer
    std::vector<std::vector<tensor_prefetch_info>> tensors_by_layer_;
    // Small, always-used MoE router path. Kept separate from speculative weights.
    std::vector<tensor_prefetch_info>              router_tensors_;
    std::vector<page_range>                        router_locked_ranges_;
    std::atomic<uint64_t>                          router_locked_bytes_{0};
    std::atomic<uint64_t>                          router_lock_failures_{0};
    std::vector<page_range>                        shared_locked_ranges_;
    std::atomic<uint64_t>                          shared_locked_bytes_{0};
    std::atomic<uint64_t>                          shared_lock_failures_{0};
    std::vector<page_range>                        small_locked_ranges_;
    std::atomic<uint64_t>                          small_locked_bytes_{0};
    std::atomic<uint64_t>                          small_lock_failures_{0};
    struct expert_hot_entry {
        int layer{-1};
        int expert_id{-1};
        std::vector<page_range> ranges;
        uint64_t locked_bytes{0};
        uint64_t last_touch{0};
        uint64_t frequency{1};
    };
    mutable std::mutex                              expert_hot_mtx_;
    std::vector<expert_hot_entry>                  expert_hot_entries_;
    uint64_t                                        expert_hot_locked_bytes_{0};
    uint64_t                                        expert_hot_admissions_{0};
    uint64_t                                        expert_hot_hits_{0};
    uint64_t                                        expert_hot_evictions_{0};
    uint64_t                                        expert_hot_budget_rejections_{0};
    uint64_t                                        expert_hot_nonresident_bytes_{0};
    uint64_t                                        expert_hot_lock_failures_{0};
    uint64_t                                        expert_hot_touch_clock_{0};
    uint64_t                                        expert_hot_admission_skips_{0};
    std::vector<std::vector<uint32_t>>              expert_hot_admission_counts_;
    struct mmap_region_info {
        void * addr;
        size_t size;
        int file_id;
    };
    std::vector<mmap_region_info>                  mmap_regions_;
    std::vector<page_range>                        expert_madv_ranges_;
    std::atomic<uint64_t>                          expert_madv_advice_calls_{0};
    std::atomic<uint64_t>                          expert_madv_advice_bytes_{0};
    std::atomic<uint64_t>                          expert_madv_advice_failures_{0};
    std::atomic<uint64_t>                          expert_file_advice_calls_{0};
    std::atomic<uint64_t>                          expert_file_advice_bytes_{0};
    std::atomic<uint64_t>                          expert_file_advice_failures_{0};
    std::atomic<uint64_t>                          expert_file_advice_invalid_ranges_{0};
    // MoE expert registry indexed by layer
    std::vector<std::vector<expert_tensor_info>>   experts_by_layer_;
    // Guards the expert registry and router predictor/accounting state.
    mutable std::mutex                              expert_state_mtx_;
    mutable std::mutex                              expert_transition_mtx_;
    expert_transition_table                        expert_transition_table_;
    // Router-selected expert IDs per layer (for next-layer prediction)
    std::vector<std::vector<int>>                  cached_router_experts_;
    struct expert_prefetch_record {
        uint64_t generation;
        std::vector<int> expert_ids;
        std::vector<size_t> issued_bytes;
    };
    // Per layer, bounded unsettled prefetch records keyed by generation (a placeholder
    // until the advice is issued).
    std::vector<std::vector<expert_prefetch_record>> pending_expert_prefetches_;
    uint64_t                                        next_expert_generation_{1};
    static constexpr size_t                         max_pending_expert_records_{64};
    std::atomic<uint64_t>                           expert_pending_rejected_generations_{0};
    std::atomic<uint64_t>                           expert_unmatched_generations_{0};
    std::atomic<uint64_t>                           reclaim_candidate_experts_{0};
    std::atomic<uint64_t>                           reclaim_calls_{0};
    std::atomic<uint64_t>                           reclaim_reclaimed_bytes_{0};
    std::atomic<uint64_t>                           reclaim_skipped_bytes_{0};
    std::atomic<uint64_t>                           reclaim_madvise_failures_{0};
    std::atomic<uint64_t>                           reclaim_invalid_layouts_{0};
    std::atomic<uint64_t>                           reclaim_invalid_ids_{0};
    // ---- SLIM-ARC FIX 2026-08-09: confidence gating (improvement 2) ----
    // Routing of the previous step (t-2), used for the "stable across two consecutive
    // tokens" high-confidence filter
    std::vector<std::vector<int>>                  prev_router_experts_;
    // Enable the 2-token stability gate when SLIM_ARC_EXPERT_CONF=1
    bool                                           conf_gating_ = false;

    // ---- SLIM-ARC FIX 2026-08-09: unified I/O budget limiting (improvement 3, following
    // admission control) ----
    // Once set_expert_budget() has been called the budget is enforced strictly; 0
    // disables expert prefetch for this step.
    std::atomic<size_t>                            expert_budget_{0};
    std::atomic<bool>                              expert_budget_enabled_{false};
    // Expert budget used in this step (reset by the unified tick on every graph_compute)
    std::atomic<size_t>                            expert_budget_used_{0};

    // ---- SLIM-ARC FIX 2026-08-09: combined predictor (improvement 4, following
    // ReMoE/DALI locality) ----
    // SLIM_ARC_EXPERT_POP=K: prefetch the temporal set unioned with the top-K hot experts
    int                                            pop_k_ = 0;
    // Per-layer expert activation-frequency counts (recent window)
    std::vector<std::vector<uint32_t>>             expert_pop_counts_;
    uint8_t                                         popularity_samples_since_decay_{0};
    std::atomic<uint64_t>                           popularity_decay_count_{0};
    uint32_t                                        expert_waste_ewma_milli_{0};
    bool                                            expert_waste_ewma_initialized_{false};
    uint64_t                                        expert_waste_samples_{0};
    expert_pressure_state                           expert_pressure_{expert_pressure_state::missing};
    size_t                                          expert_residency_budget_{0};
    bool                                            expert_residency_snapshot_set_{false};
    expert_waste_controller                         expert_waste_controller_;
    bool                                            expert_waste_restricted_{false};
    std::atomic<uint64_t>                           residency_samples_{0};
    std::atomic<uint64_t>                           residency_admitted_experts_{0};
    std::atomic<uint64_t>                           residency_admitted_bytes_{0};
    std::atomic<uint64_t>                           residency_skipped_bytes_{0};
    std::atomic<uint64_t>                           residency_fallbacks_{0};
    std::atomic<uint64_t>                           residency_pressure_missing_{0};
    std::atomic<uint64_t>                           residency_pressure_normal_{0};
    std::atomic<uint64_t>                           residency_pressure_high_{0};
    std::atomic<uint64_t>                           residency_pressure_critical_{0};

    // ---- SLIM-ARC 2026-11-02: per-layer signalling state ----
    void on_layer_boundary(int32_t layer);
    // first node of every layer, sorted by pointer value -- searched with binary_search
    std::vector<const ggml_tensor *>                observe_;
    // layer index of the matching observe_ entry (not sorted: it follows observe_)
    std::vector<int32_t>                            observe_layer_;
    // last layer we issued a request for; reset when the layer counter wraps (= new token)
    int32_t                                         last_notified_ = -1;
    // graph the tables above were built from, for idempotency across the graph-reuse path
    const ggml_cgraph *                             prepared_gf_ = nullptr;
    // set when the scan could not recover layer boundaries; eval_cb then degrades to the user cb
    bool                                            graph_signalling_disabled_ = false;
    ggml_backend_sched_eval_callback                user_cb_ = nullptr;
    void *                                          user_cb_data_ = nullptr;
};

// Helper: extract layer index from tensor name (blk.%d.*)
int tensor_layer_from_name(const char * name);

// [QLH port] Process-global scheduler handle. The upstream integration patch calls these two
// accessors, but the published sources do not define them -- the loader owns the instance and
// the context reads it back during graph_compute. Kept as a pair of free functions so the two
// translation units do not have to share a header-level static.
prefetch_scheduler * get_global_prefetch_scheduler();
void set_global_prefetch_scheduler(prefetch_scheduler * scheduler);

} // namespace slim_arc
