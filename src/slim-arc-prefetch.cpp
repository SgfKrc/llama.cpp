// SLIM-ARC: Tensor-level asynchronous prefetch scheduler
//
// Implements layer-ahead prefetch on top of upstream llama.cpp's mmap.
// When computing layer N, async madvise(WILLNEED) for layers N+1..N+window.

#include "slim-arc-prefetch.h"
#include "llama-impl.h"   // LLAMA_LOG_*: the library logger, so --log-file and verbosity apply

#include "slim-arc-page-range.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <system_error>
// The original implementation targets Linux. Windows has no madvise/sysconf, so the
// two call sites below get a win32 implementation:
//   MADV_WILLNEED  -> PrefetchVirtualMemory (Win8+), which is the same hint to the memory
//                     manager: bring these pages resident now.
//   MADV_DONTNEED  -> DiscardVirtualMemory (Win8.1+), the closest equivalent (contents become
//                     undefined and the pages are dropped).
//   MADV_* others  -> no-op; they only steer the kernel readahead heuristic, which Windows
//                     does not expose.
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX   // windows.h defines max/min as macros and breaks std::max / ::max()
#  endif
#  include <windows.h>
#  include <memoryapi.h>
#  include <sysinfoapi.h>
// The scheduler passes POSIX_MADV_* through its advice callback; define the same names with
// distinct values so the win32 branch can switch on them.
enum : int {
    POSIX_MADV_NORMAL     = 0,
    POSIX_MADV_RANDOM     = 1,
    POSIX_MADV_SEQUENTIAL = 2,
    POSIX_MADV_WILLNEED   = 3,
    POSIX_MADV_DONTNEED   = 4,
};
#else
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace slim_arc {

namespace {
// mlock/munlock wrap VirtualLock/VirtualUnlock. The scheduler locks the router and
// small-tensor ranges so the memory manager keeps them resident; the win32 call takes the
// same (address, length) pair and fails the same way, so the call sites need no changes.
#if defined(_WIN32)
int mlock(const void * addr, size_t length) {
    return VirtualLock(const_cast<void *>(addr), length) ? 0 : -1;
}

int munlock(const void * addr, size_t length) {
    return VirtualUnlock(const_cast<void *>(addr), length) ? 0 : -1;
}
#endif

template <typename T>
T saturating_add(T left, T right) noexcept {
    const T maximum = std::numeric_limits<T>::max();
    return right > maximum - left ? maximum : left + right;
}

template <typename T>
void atomic_saturating_add(std::atomic<T> & target, T increment) noexcept {
    T current = target.load();
    while (!target.compare_exchange_weak(current, saturating_add(current, increment))) {
    }
}

// The SLIM_ARC_* switches, parsed in one place.
//
// Six call sites used to spell out the same steps (getenv, from_chars with a full-string check, a
// range check, an fprintf, a default), and two of them had already drifted in what they did with an
// empty value. Both helpers keep the behaviour every site had: an unset variable is silent and
// returns the default, a malformed one is reported on stderr and returns the default.
//
// env_bool only treats "1" as true. The switches have always been exact-match, so "true" / "yes"
// stay off; that is deliberate, not an oversight to be fixed by making this smarter.
//
// An empty value is malformed and reports one stderr line, like every other malformed value here.
bool env_bool(const char * name, bool def) noexcept {
    const char * const raw = std::getenv(name);
    if (raw == nullptr) return def;
    return std::strcmp(raw, "1") == 0;
}

int64_t env_int(const char * name, int64_t def, int64_t lo, int64_t hi) noexcept {
    const char * const raw = std::getenv(name);
    if (raw == nullptr) return def;
    const std::string_view value{raw};
    int64_t parsed{0};
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (value.empty() || result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed < lo || parsed > hi) {
        LLAMA_LOG_WARN("SLIM-ARC: invalid %s; expected an integer in [%lld,%lld]\n", name,
                       static_cast<long long>(lo), static_cast<long long>(hi));
        return def;
    }
    return parsed;
}

int system_advice(void * address, size_t length, int advice) {
#if defined(_WIN32)
    if (address == nullptr || length == 0) {
        return EINVAL;
    }
    if (advice == POSIX_MADV_WILLNEED) {
        WIN32_MEMORY_RANGE_ENTRY entry{ address, length };
        if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0)) {
            return static_cast<int>(GetLastError());
        }
        return 0;
    }
    if (advice == POSIX_MADV_DONTNEED) {
        // DiscardVirtualMemory wants a page-aligned range; the scheduler already hands us
        // tensor extents, which land on page boundaries inside the mapping.
        const DWORD rc = DiscardVirtualMemory(address, length);
        return rc == ERROR_SUCCESS ? 0 : static_cast<int>(rc);
    }
    // NORMAL / RANDOM / SEQUENTIAL only steer Linux readahead; nothing to map them onto.
    return 0;
#else
    return posix_madvise(address, length, advice);
#endif
}

long system_page_size() {
#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return static_cast<long>(info.dwPageSize);
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

// Free physical memory right now. The layer-ahead budget is derived from this because the
// number that matters is how much room the kernel still has for read-ahead pages, not the model
// size or a constant.
size_t available_physical_bytes() noexcept {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) == 0) return 0;
    return static_cast<size_t>(status.ullAvailPhys);
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
#endif
}

// Ceiling for one prefetch request, in bytes.
//
// The budget is a *per-request* ceiling: worker_loop() spends it on the ranges of the request
// (layers current+1 .. current+window) and skips whatever does not fit. Deriving it from free
// physical memory rather than from a constant keeps the behaviour sane on both a 32 GiB desktop
// and a 4 GiB SBC:
//   - cap it at 1/8 of free RAM, so a burst of read-ahead cannot evict the pages the model is
//     actively using;
//   - floor at 64 MiB, else on a tight machine the floor rounds to zero pages and the scheduler
//     goes back to issuing nothing;
//   - ceiling at 1 GiB, because the ceiling is only a safety net: plan_request() then lowers the
//     per-request budget again to one layer's worth of registered weights, which is what actually
//     bounds the traffic.
// SLIM_ARC_BUDGET_MB overrides all of it for experiments.
size_t default_prefetch_budget_bytes() noexcept {
    // -1 is "not set, or not usable": an unset variable is silent, a malformed one is reported by
    // env_int(), and both fall through to the derived budget. 0 is a valid value (it disables the
    // read-ahead) and is not confused with the sentinel.
    const int64_t mb = env_int("SLIM_ARC_BUDGET_MB", -1, 0, 65536);
    if (mb >= 0) {
        return static_cast<size_t>(mb) * (1ULL << 20);
    }
    const size_t available = available_physical_bytes();
    if (available == 0) {
        return 128ULL << 20;
    }
    constexpr size_t budget_min = 64ULL << 20;
    constexpr size_t budget_max = 1024ULL << 20;
    const size_t share = available / 8;
    return std::min(std::max(share, budget_min), budget_max);
}

void update_atomic_peak(std::atomic<uint64_t> & peak, uint64_t candidate) noexcept {
    uint64_t current = peak.load();
    while (current < candidate && !peak.compare_exchange_weak(current, candidate)) {
    }
}

}

int tensor_layer_from_name(const char * name) {
    if (!name) return -1;
    // match "blk.%d." prefix
    if (std::strncmp(name, "blk.", 4) != 0) return -1;    const char * p = name + 4;
    if (*p < '0' || *p > '9') return -1;
    int layer = 0;
    while (*p >= '0' && *p <= '9') {
        layer = layer * 10 + (*p - '0');
        ++p;
    }
    if (*p != '.') return -1;
    return layer;
}

// Graph node names carry their layer as a trailing "-N" suffix --
// llama_context::graph_get_cb() renames every tensor produced inside a layer with
// ggml_format_name(cur, "%s-%d", name, il), e.g. "ffn_out-12". Weight names
// ("blk.7.attn_norm.weight") have no such suffix and are rejected, and the fallback names that
// ggml_build_forward_expand() assigns ("node_123", "leaf_7") contain no dash at all, so they are
// rejected too. Only the trailing "-<digits>" form is accepted: the prefix must be non-empty and
// every character after the last dash must be a digit.
static int tensor_layer_from_suffix(const char * name) {
    if (name == nullptr || name[0] == '\0') return -1;
    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || dash == name) return -1;
    const char * p = dash + 1;
    if (*p < '0' || *p > '9') return -1;
    int layer = 0;
    for (; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return -1;
        layer = layer * 10 + (*p - '0');
        if (layer > 4096) return -1; // implausible layer index: not one of our names
    }
    return layer;
}

prefetch_scheduler::prefetch_scheduler(
    int n_threads,
    int window,
    std::function<void()> request_claim_hook,
    advice_fn advice,
    page_size_query_fn page_size_query)
    : slow_storage_enabled_(env_bool("SLIM_ARC_SLOW_STORAGE", false))
    , router_prefetch_enabled_(env_bool("SLIM_ARC_ROUTER_PREFETCH", false))
    , router_mlock_enabled_(env_bool("SLIM_ARC_ROUTER_MLOCK", false))
    , shared_mlock_enabled_(env_bool("SLIM_ARC_SHARED_MLOCK", false))
    , small_mlock_enabled_(env_bool("SLIM_ARC_SMALL_MLOCK", false))
    , n_threads_(slow_storage_enabled_ ? 1 : std::max(1, n_threads))
    , window_(slow_storage_enabled_ ? 1 : std::max(1, window))
    , request_claim_hook_(std::move(request_claim_hook))
    , advice_(advice ? std::move(advice) : advice_fn{system_advice})
    , page_size_query_(page_size_query ? std::move(page_size_query) : page_size_query_fn{system_page_size}) {
    // See default_prefetch_budget_bytes(): the budget used to stay 0 because the module
    // that called set_memory_budget() (unified_io_scheduler) was never ported.
    memory_budget_.store(default_prefetch_budget_bytes());
    workers_.reserve(n_threads_);
    for (int i = 0; i < n_threads_; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

prefetch_scheduler::~prefetch_scheduler() {
    shutdown();
    dump_metrics();
    std::vector<page_range> locked;
    std::vector<page_range> shared_locked;
    std::vector<page_range> small_locked;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        locked.swap(router_locked_ranges_);
        shared_locked.swap(shared_locked_ranges_);
        small_locked.swap(small_locked_ranges_);
    }
    for (const page_range & range : locked) {
        (void) munlock(reinterpret_cast<void *>(range.address), range.length);
    }
    for (const page_range & range : shared_locked) {
        (void) munlock(reinterpret_cast<void *>(range.address), range.length);
    }
    for (const page_range & range : small_locked) {
        (void) munlock(reinterpret_cast<void *>(range.address), range.length);
    }
}

void prefetch_scheduler::shutdown() noexcept {
    std::lock_guard<std::mutex> shutdown_lock(shutdown_mtx_);
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (stop_.load()) return;
        enabled_.store(false);
        stop_.store(true);
        pending_requests_.clear();
    }
    cv_.notify_all();
    for (auto & t : workers_) {
        if (t.joinable()) t.join();
    }
}

void prefetch_scheduler::register_tensor(const char * name, void * addr, size_t size, int layer) {
    if (layer < 0 || addr == nullptr || size == 0) return;
    if (stop_.load()) return;

    // Every decision that needs the page size, and every mlock() syscall, is taken before
    // mtx_ is acquired: mlock can block on a page fault, and mtx_ is also taken by eval_cb() on
    // the compute thread, so holding it across the syscall would park inference behind a model
    // load. Only the bookkeeping below runs under the lock.
    const bool lock_shared = shared_mlock_enabled_ && name != nullptr &&
        std::strstr(name, "_shexp") != nullptr;
    const bool lock_small = small_mlock_enabled_ && name != nullptr && size <= (1ULL << 20) &&
        std::strstr(name, "_exps") == nullptr && std::strstr(name, "_shexp") == nullptr &&
        std::strstr(name, "ffn_gate_inp") == nullptr;
    // The gate is `router_mode && is_router`, not just `is_router`. With both router
    // switches off (the default configuration) the router weight is an ordinary layer tensor and
    // must stay in tensors_by_layer_: an unconditional `if (is_router) return;` would silently
    // drop it from layer-ahead prefetch exactly when the router path is switched *off*. Gating on
    // `router_mode` alone fails the other way: with only SLIM_ARC_ROUTER_MLOCK=1 set it returns for
    // every tensor, so tensors_by_layer_ stays empty and no layer prefetch is issued, with no error
    // in the log. Do not "simplify" this back -- the only case that may return early is the one
    // where the tensor was actually recorded in router_tensors_ / router_locked_ranges_ above.
    const bool router_mode = router_prefetch_enabled_ || router_mlock_enabled_;
    const bool is_router = name != nullptr && std::strstr(name, ".ffn_gate_inp") != nullptr &&
        std::strstr(name, ".weight") != nullptr;
    const bool lock_router = router_mode && is_router && router_mlock_enabled_;

    const long   raw_page_size = (lock_shared || lock_small || lock_router) ? page_size_query_() : 0;
    const size_t page_size = raw_page_size > 0 ? static_cast<size_t>(raw_page_size) : 0;

    page_range shared_range;
    page_range small_range;
    page_range router_range;
    bool shared_locked = false;
    bool small_locked  = false;
    bool router_locked = false;
    if (lock_shared) {
        shared_range  = covering_page_range(reinterpret_cast<uintptr_t>(addr), size, page_size);
        shared_locked = shared_range.valid &&
            mlock(reinterpret_cast<void *>(shared_range.address), shared_range.length) == 0;
    }
    if (lock_small) {
        small_range  = covering_page_range(reinterpret_cast<uintptr_t>(addr), size, page_size);
        small_locked = small_range.valid &&
            mlock(reinterpret_cast<void *>(small_range.address), small_range.length) == 0;
    }
    if (lock_router) {
        router_range  = covering_page_range(reinterpret_cast<uintptr_t>(addr), size, page_size);
        router_locked = router_range.valid &&
            mlock(reinterpret_cast<void *>(router_range.address), router_range.length) == 0;
    }

    std::lock_guard<std::mutex> lock(mtx_);
    if (stop_.load()) {
        // Shut down between the mlock above and this lock. These pages are not in any of the
        // tables the destructor sweeps, so they have to be released here or they stay locked for
        // the rest of the process.
        if (shared_locked) {
            (void) munlock(reinterpret_cast<void *>(shared_range.address), shared_range.length);
        }
        if (small_locked) {
            (void) munlock(reinterpret_cast<void *>(small_range.address), small_range.length);
        }
        if (router_locked) {
            (void) munlock(reinterpret_cast<void *>(router_range.address), router_range.length);
        }
        return;
    }
    if (lock_shared) {
        if (shared_locked) {
            shared_locked_ranges_.push_back(shared_range);
            atomic_saturating_add(shared_locked_bytes_, static_cast<uint64_t>(shared_range.length));
        } else {
            atomic_saturating_add(shared_lock_failures_, uint64_t{1});
        }
    }
    if (lock_small) {
        if (small_locked) {
            small_locked_ranges_.push_back(small_range);
            atomic_saturating_add(small_locked_bytes_, static_cast<uint64_t>(small_range.length));
        } else {
            atomic_saturating_add(small_lock_failures_, uint64_t{1});
        }
    }
    if (router_mode && is_router) {
        if (router_prefetch_enabled_) {
            router_tensors_.push_back({addr, size, layer, 0});
        }
        if (router_mlock_enabled_) {
            if (router_locked) {
                router_locked_ranges_.push_back(router_range);
                atomic_saturating_add(router_locked_bytes_, static_cast<uint64_t>(router_range.length));
            } else {
                atomic_saturating_add(router_lock_failures_, uint64_t{1});
            }
        }
        return;
    }
    if ((size_t)layer >= tensors_by_layer_.size()) {
        tensors_by_layer_.resize(layer + 1);
    }
    tensors_by_layer_[layer].push_back({addr, size, layer, 0});
}

void prefetch_scheduler::attach_user_callback(ggml_backend_sched_eval_callback cb, void * user_data) {
    std::lock_guard<std::mutex> lock(mtx_);
    user_cb_      = cb;
    user_cb_data_ = user_data;
}

void prefetch_scheduler::prepare_graph(const ggml_cgraph * gf) {
    if (gf == nullptr) return;

    std::lock_guard<std::mutex> lock(mtx_);

    // the graph-reuse path calls this again for the very same graph; the tables cannot have
    // Changed because the tensors were not rebuilt. The idempotency check is inside the
    // lock: eval_cb() reads prepared_gf_'s consequences (observe_/observe_layer_) on the compute
    // thread, so the whole "is it the same graph" decision belongs to the same critical section
    // that publishes the tables.
    if (prepared_gf_ == gf) return;

    prepared_gf_ = gf;
    observe_.clear();
    observe_layer_.clear();
    last_notified_.store(-1, std::memory_order_relaxed);
    graph_signalling_disabled_ = false;

    auto * graph = const_cast<ggml_cgraph *>(gf);
    const int n_nodes = ggml_graph_n_nodes(graph);
    int32_t current = -1;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * t = ggml_graph_node(graph, i);
        if (t == nullptr) continue;
        int32_t layer = tensor_layer_from_suffix(t->name);
        // a layer index that goes backwards is a name that merely ends in "-<digits>";
        // the graph is built layer by layer, so the real sequence never decreases
        if (layer >= 0 && current >= 0 && layer < current) layer = -1;
        if (layer < 0) layer = current; // carry the previous layer over unnamed/leaf nodes
        if (layer < 0) continue;        // still before the first layer (embeddings etc.)
        if (layer != current) {
            observe_.push_back(t);
            observe_layer_.push_back(layer);
            current = layer;
        }
    }

    if (observe_.size() < 2) {
        // nothing usable: stay out of the way instead of guessing
        graph_signalling_disabled_ = true;
        observe_.clear();
        observe_layer_.clear();
        LLAMA_LOG_WARN("[SLIM-ARC-PREFETCH] layer signalling disabled: no layer boundaries "
                "recovered from %d graph nodes\n", n_nodes);
        return;
    }

    // std::binary_search / std::lower_bound below compare the tensors by pointer value, so the
    // table has to be in pointer order. Layer order is *not* pointer order (the graph is built
    // layer by layer, the allocations are not), hence the explicit sort -- both tables move
    // together so that observe_layer_[i] still describes observe_[i].
    {
        std::vector<size_t> order(observe_.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [this](size_t a, size_t b) { return observe_[a] < observe_[b]; });
        std::vector<const ggml_tensor *> nodes;
        std::vector<int32_t>             layers;
        nodes.reserve(order.size());
        layers.reserve(order.size());
        for (const size_t i : order) {
            nodes.push_back(observe_[i]);
            layers.push_back(observe_layer_[i]);
        }
        observe_.swap(nodes);
        observe_layer_.swap(layers);
    }

    LLAMA_LOG_INFO("[SLIM-ARC-PREFETCH] layer signalling armed: %zu boundaries over %d "
            "graph nodes, layers %d..%d\n", observe_.size(), n_nodes,
            *std::min_element(observe_layer_.begin(), observe_layer_.end()),
            *std::max_element(observe_layer_.begin(), observe_layer_.end()));
}

void prefetch_scheduler::on_layer_boundary(int32_t layer) {
    if (layer < 0) return;
    // last_notified_ is written here (compute thread) and reset by prepare_graph()
    // (context thread), so it is atomic. Relaxed is enough: the value only suppresses a duplicate
    // advice for a layer already covered by the previous request, it publishes nothing else.
    int32_t previous = last_notified_.load(std::memory_order_relaxed);
    // the layer counter wraps at the start of every token
    if (previous >= 0 && layer < previous) {
        previous = -1;
    }
    // the request issued for `previous` already covers [previous+1, previous+window]; issuing
    // another one for a layer inside that window would advise the same pages twice
    if (previous >= 0 && layer - previous < window_) return;
    last_notified_.store(layer, std::memory_order_relaxed);
    notify_layer_compute(layer);
}

bool prefetch_scheduler::eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * sched = static_cast<prefetch_scheduler *>(user_data);
    if (sched == nullptr || t == nullptr) return true;

    // The chained user callback (attach_user_callback) and the layer-boundary tables
    // (prepare_graph) are both owned by the context thread, while this callback runs on the
    // compute thread: a graph rebuild would otherwise clear/swap observe_ under the reader.
    // Everything is therefore *read* under mtx_, and the lock is dropped before either the user
    // callback runs or on_layer_boundary() is entered -- both may re-enter this class, and
    // notify_layer_compute() takes mtx_ again, so holding it across the call would deadlock on a
    // non-recursive mutex.
    ggml_backend_sched_eval_callback user_cb      = nullptr;
    void *                          user_cb_data = nullptr;
    int32_t                         boundary_layer = -1;
    bool                            signalling_disabled = false;
    {
        std::lock_guard<std::mutex> lock(sched->mtx_);
        user_cb      = sched->user_cb_;
        user_cb_data = sched->user_cb_data_;
        signalling_disabled = sched->graph_signalling_disabled_;
        if (!signalling_disabled) {
            const auto it = std::lower_bound(sched->observe_.begin(), sched->observe_.end(),
                                             static_cast<const ggml_tensor *>(t));
            if (it != sched->observe_.end() && *it == t) {
                boundary_layer =
                    sched->observe_layer_[static_cast<size_t>(it - sched->observe_.begin())];
            }
        }
    }

    const bool has_user = user_cb != nullptr;
    const bool user_ret = has_user && user_cb(t, ask, user_cb_data);

    if (ask) {
        // returning true makes the scheduler stop at this node to hand it back with ask == false
        if (user_ret) return true;
        if (signalling_disabled) return false;
        return boundary_layer >= 0;
    }

    if (boundary_layer >= 0) {
        sched->on_layer_boundary(boundary_layer);
    }

    // keep the user's cancel semantics; without a user callback the compute always continues
    return has_user ? user_ret : true;
}

void prefetch_scheduler::notify_layer_compute(int current_layer) {
    if (!enabled_.load()) return;
    // SLIM_ARC_NOOP=1 keeps the eval callback installed and keeps the
    // graph splitting it causes, but issues no PrefetchVirtualMemory at all. This separates the
    // cost of "callback + 40 split submits" from the cost of "actually advising pages".
    if (std::getenv("SLIM_ARC_NOOP") != nullptr) return;
    // SLIM_ARC_MAX_CALLS=N bounds how many prefetch requests a process
    // issues in total. After the cap nothing is advised, which separates two explanations for
    // the 2.5x slowdown: "the call itself is expensive" (a small cap is already slow) from
    // "steady-state prefetch keeps the page cache thrashing" (a small cap is fast).
    static std::atomic<uint64_t> issued_notifications{0};
    if (const char * cap_env = std::getenv("SLIM_ARC_MAX_CALLS")) {
        const uint64_t cap = std::strtoull(cap_env, nullptr, 10);
        if (issued_notifications.fetch_add(1, std::memory_order_relaxed) >= cap) {
            return;
        }
    }
    prefetch_request request = plan_request(current_layer);
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (stop_.load()) return;
        if (next_request_generation_ == std::numeric_limits<uint64_t>::max()) {
            atomic_saturating_add(dropped_requests_, uint64_t{1});
            return;
        }
        if (slow_storage_enabled_) {
            if (!pending_requests_.empty() && pending_requests_.back().layer == current_layer) {
                return;
            }
            for (const prefetch_request & stale : pending_requests_) {
                atomic_saturating_add(budget_stale_requests_, uint64_t{1});
                atomic_saturating_add(budget_stale_bytes_, stale.covered_bytes);
            }
            pending_requests_.clear();
        } else if (pending_requests_.size() == max_pending_requests) {
            pending_requests_.pop_front();
            atomic_saturating_add(dropped_requests_, uint64_t{1});
        }
        request.generation = ++next_request_generation_;
        pending_requests_.push_back(std::move(request));
    }
    cv_.notify_one();
}

prefetch_scheduler::prefetch_request prefetch_scheduler::plan_request(int current_layer) {
    prefetch_request request;
    request.layer = current_layer;
    request.memory_budget = memory_budget_.load();
    std::vector<tensor_prefetch_info> items;
    int window_layers = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (stop_.load()) return request;
        if (router_prefetch_enabled_) {
            items = router_tensors_;
        } else {
            for (int offset = 1; offset <= window_; ++offset) {
                const int layer = current_layer + offset;
                if (layer < 0 || static_cast<size_t>(layer) >= tensors_by_layer_.size()) continue;
                const std::vector<tensor_prefetch_info> & layer_tensors = tensors_by_layer_[layer];
                if (!layer_tensors.empty()) ++window_layers;
                for (const tensor_prefetch_info & tensor : layer_tensors) {
                    if (tensor.addr != nullptr && tensor.size > 0) items.push_back(tensor);
                }
            }
        }
    }

    for (const tensor_prefetch_info & item : items) {
        request.requested_bytes = saturating_add(
            request.requested_bytes,
            static_cast<uint64_t>(item.size));
    }
    request.advice_requests = static_cast<uint64_t>(items.size());
    // Per-request ceiling, part 2: never advise more than one layer's worth of weights in a
    // single request. The window covers `window_` layers, so the request sizes are derived from
    // the actual bytes registered per layer rather than from a constant -- a dense 70B and a
    // small MoE get correctly different budgets. Workers still walk the ranges in order, so
    // "one layer" means the nearest one, which is the one the next step needs first.
    // With per-layer signalling (on_layer_boundary()) the next
    // notification is a whole window further along, so clipping every request to a single layer
    // would leave layers 2..window of each window prefetched never. The request may spend the
    // whole window; the free-memory ceiling in memory_budget_ still caps it.
    if (window_layers > 0) {
        const uint64_t per_layer = request.requested_bytes / static_cast<uint64_t>(window_layers);
        // Cap the request to ONE layer, not to the whole window.
        // The window-sized budget made every token advise ~17 GB on a 16 GB machine
        // (40 layers / window 3 notifications, x 3 layers each), which evicted the hot
        // working set and measured 2.2x slower than prefetch-off. One layer per request
        // cuts the volume by window_layers and keeps the layer ordering intact.
        if (per_layer > 0 && per_layer < request.memory_budget) {
            request.memory_budget = per_layer;
        }
    }
    const long raw_page_size = page_size_query_();
    if (raw_page_size <= 0) {
        request.invalid_ranges = request.advice_requests;
        return request;
    }
    request.page_size = static_cast<size_t>(raw_page_size);
    std::vector<page_range> planned_ranges;
    planned_ranges.reserve(items.size());
    for (const tensor_prefetch_info & item : items) {
        const page_range range = covering_page_range(
            reinterpret_cast<uintptr_t>(item.addr),
            item.size,
            request.page_size);
        if (!range.valid) {
            request.invalid_ranges = saturating_add(request.invalid_ranges, uint64_t{1});
            continue;
        }
        planned_ranges.push_back(range);
    }
    page_range_set coalesced = coalesce_page_ranges(std::move(planned_ranges));
    if (!coalesced.valid) {
        request.invalid_ranges = saturating_add(
            request.invalid_ranges,
            static_cast<uint64_t>(coalesced.input_ranges));
        return request;
    }
    request.ranges = std::move(coalesced.ranges);
    for (const page_range & range : request.ranges) {
        request.covered_bytes = saturating_add(
            request.covered_bytes,
            static_cast<uint64_t>(range.length));
    }
    return request;
}

void prefetch_scheduler::worker_loop() {
    while (true) {
        prefetch_request request;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_.load() || !pending_requests_.empty(); });
            if (stop_) return;
            request = std::move(pending_requests_.front());
            pending_requests_.pop_front();
        }
        if (request_claim_hook_) request_claim_hook_();

        std::vector<page_range> selected;
        selected.reserve(request.ranges.size());
        uint64_t skipped{0};
        uint64_t selected_bytes{0};
        uint64_t remaining = request.memory_budget;
        for (const page_range & range : request.ranges) {
            const uint64_t available = std::min<uint64_t>(remaining, range.length);
            const uint64_t admitted = request.page_size == 0 ? 0 : available - available % request.page_size;
            if (admitted > 0) {
                selected.push_back({range.address, static_cast<size_t>(admitted), 0, true, 0});
                remaining -= admitted;
                selected_bytes = saturating_add(selected_bytes, admitted);
            }
            skipped = saturating_add(skipped, static_cast<uint64_t>(range.length) - admitted);
        }

        const uint64_t inflight = budget_inflight_bytes_.fetch_add(selected_bytes) + selected_bytes;
        update_atomic_peak(budget_inflight_peak_bytes_, inflight);
        uint64_t issued{0};
        uint64_t failures = request.invalid_ranges;
        for (const page_range & range : selected) {
            if (advice_(reinterpret_cast<void *>(range.address), range.length, POSIX_MADV_WILLNEED) == 0) {
                issued = saturating_add(issued, static_cast<uint64_t>(range.length));
            } else {
                failures = saturating_add(failures, uint64_t{1});
            }
        }
        budget_inflight_bytes_.fetch_sub(selected_bytes);

        atomic_saturating_add(budget_requested_bytes_, request.requested_bytes);
        atomic_saturating_add(budget_issued_bytes_, issued);
        atomic_saturating_add(budget_skipped_bytes_, skipped);
        atomic_saturating_add(budget_madvise_failures_, failures);
        atomic_saturating_add(budget_advice_requests_, request.advice_requests);
        atomic_saturating_add(
            budget_coalesced_ranges_,
            static_cast<uint64_t>(request.ranges.size()));
        atomic_saturating_add(budget_covered_bytes_, request.covered_bytes);
        atomic_saturating_add(budget_invalid_ranges_, request.invalid_ranges);
        if (skipped > 0) {
            atomic_saturating_add(budget_rounds_throttled_, uint64_t{1});
        }
        total_bytes_.fetch_add(static_cast<size_t>(issued));
        total_calls_.fetch_add(1);
    }
}

// Per-run summary of the layer-ahead counters; printed from the destructor.
void prefetch_scheduler::dump_metrics() const {
    LLAMA_LOG_INFO(
            "[SLIM-ARC-PREFETCH] calls=%llu issued=%.1fMB window=%d threads=%d "
            "requested=%.1fMB advice_requests=%llu coalesced=%llu invalid=%llu covered=%.1fMB\n",
            static_cast<unsigned long long>(total_calls_.load()),
            static_cast<double>(total_bytes_.load()) / 1048576.0,
            effective_window(),
            n_threads_,
            static_cast<double>(budget_requested_bytes_.load()) / 1048576.0,
            static_cast<unsigned long long>(budget_advice_requests_.load()),
            static_cast<unsigned long long>(budget_coalesced_ranges_.load()),
            static_cast<unsigned long long>(budget_invalid_ranges_.load()),
            static_cast<double>(budget_covered_bytes_.load()) / 1048576.0);
    if (router_mlock_enabled_) {
        LLAMA_LOG_INFO(
            "[SLIM-ARC-ROUTER] locked_bytes=%llu lock_failures=%llu\n",
            static_cast<unsigned long long>(router_locked_bytes_.load()),
            static_cast<unsigned long long>(router_lock_failures_.load()));
    }
    if (shared_mlock_enabled_) {
        LLAMA_LOG_INFO(
            "[SLIM-ARC-SHARED] locked_bytes=%llu lock_failures=%llu\n",
            static_cast<unsigned long long>(shared_locked_bytes_.load()),
            static_cast<unsigned long long>(shared_lock_failures_.load()));
    }
    if (small_mlock_enabled_) {
        LLAMA_LOG_INFO(
            "[SLIM-ARC-SMALL] locked_bytes=%llu lock_failures=%llu\n",
            static_cast<unsigned long long>(small_locked_bytes_.load()),
            static_cast<unsigned long long>(small_lock_failures_.load()));
    }
}

// See slim-arc-prefetch.h. The loader installs the instance while init_mappings is
// running, the context reads it back on every graph_compute. A raw pointer is enough: the
// scheduler outlives the process (function-local static in the loader), and clearing it is
// never needed because the only writer runs once per model load.
static prefetch_scheduler * g_prefetch_scheduler = nullptr;

prefetch_scheduler * get_global_prefetch_scheduler() {
    return g_prefetch_scheduler;
}

void set_global_prefetch_scheduler(prefetch_scheduler * scheduler) {
    g_prefetch_scheduler = scheduler;
}

} // namespace slim_arc
