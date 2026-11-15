#pragma once

#include "slim-arc-page-range.h"

#include "ggml-backend.h"

// SLIM-ARC: Tensor-level asynchronous prefetch scheduler
//
// This module implements layer-ahead prefetch on top of upstream llama.cpp's
// mmap infrastructure. When computing layer N, it asynchronously issues
// posix_madvise(WILLNEED) for tensors in layers N+1..N+window, allowing the
// kernel to overlap I/O with computation.

#include <atomic>
#include <climits>  // Guard against cascading INT_MAX errors
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

struct tensor_prefetch_info {
    void *   addr;      // mmap address of tensor data
    size_t   size;      // tensor data size in bytes
    int      layer;     // layer index (or -1 for non-layer tensors)
    uint64_t signature; // monotonic counter to detect graph changes
};

class prefetch_scheduler {
  public:
    using advice_fn = std::function<int(void *, size_t, int)>;
    using page_size_query_fn = std::function<long()>;

    explicit prefetch_scheduler(
        int n_threads = 2,
        int window = 3,
        std::function<void()> request_claim_hook = {},
        advice_fn advice = {},
        page_size_query_fn page_size_query = {});
    ~prefetch_scheduler();
    prefetch_scheduler(const prefetch_scheduler &) = delete;
    prefetch_scheduler & operator=(const prefetch_scheduler &) = delete;
    void shutdown() noexcept;

    // Register a tensor for potential prefetch. Called during model load.
    void register_tensor(const char * name, void * addr, size_t size, int layer);

    // Notify that we are about to compute layer `current_layer`.
    // This triggers async madvise(WILLNEED) for layers
    // [current_layer+1, current_layer+window].
    void notify_layer_compute(int current_layer);

    // ---- real per-layer signalling ----
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
    // Print the per-run summary (called from the destructor).
    void dump_metrics() const;

    // ---- Interfaces the Pi5-side fix needs ----
    // Current prefetch window size (used by graph_compute to compute the layer range to
    // prefetch).
    int  effective_window() const { return window_; }
    // Memory budget (used by unified_io_scheduler to apportion the weight-prefetch quota).
    void set_memory_budget(size_t bytes) { memory_budget_.store(bytes); }

  private:
    void worker_loop();
    const bool slow_storage_enabled_;
    const bool router_prefetch_enabled_;
    const bool router_mlock_enabled_;
    const bool shared_mlock_enabled_;
    const bool small_mlock_enabled_;
    int n_threads_;
    int window_;
    // unified_io_scheduler (the module that used to hand this scheduler its byte budget)
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

    std::vector<std::thread>          workers_;
    std::function<void()>             request_claim_hook_;
    const advice_fn                   advice_;
    const page_size_query_fn          page_size_query_;
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

    // ---- per-layer signalling state ----
    void on_layer_boundary(int32_t layer);
    // first node of every layer, sorted by pointer value -- searched with binary_search
    std::vector<const ggml_tensor *>                observe_;
    // layer index of the matching observe_ entry (not sorted: it follows observe_)
    std::vector<int32_t>                            observe_layer_;
    // last layer we issued a request for; reset when the layer counter wraps (= new token).
    // Atomic: written by on_layer_boundary() on the compute thread and reset by
    // prepare_graph() on the context thread (the other two tables above are protected by mtx_,
    // this one is a single word and does not need the lock on the hot path).
    std::atomic<int32_t>                            last_notified_{-1};
    // graph the tables above were built from, for idempotency across the graph-reuse path
    const ggml_cgraph *                             prepared_gf_ = nullptr;
    // set when the scan could not recover layer boundaries; eval_cb then degrades to the user cb
    bool                                            graph_signalling_disabled_ = false;
    ggml_backend_sched_eval_callback                user_cb_ = nullptr;
    void *                                          user_cb_data_ = nullptr;
};

// Helper: extract layer index from tensor name (blk.%d.*)
int tensor_layer_from_name(const char * name);

// Process-global scheduler handle. The loader owns the instance and the context reads it back
// during graph_compute, so these are two free functions rather than a header-level static shared
// by the two translation units.
prefetch_scheduler * get_global_prefetch_scheduler();
void set_global_prefetch_scheduler(prefetch_scheduler * scheduler);

} // namespace slim_arc
