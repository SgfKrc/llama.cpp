// SLIM-ARC: Tensor-level asynchronous prefetch scheduler
//
// Implements layer-ahead prefetch on top of upstream llama.cpp's mmap.
// When computing layer N, async madvise(WILLNEED) for layers N+1..N+window.

#include "slim-arc-prefetch.h"

#include "slim-arc-expert-reclaim.h"
#include "slim-arc-page-range.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string_view>
#include <system_error>
// [QLH port] The upstream patch targets Linux. Windows has no madvise/mincore/sysconf, so the
// three call sites below get a win32 implementation:
//   MADV_WILLNEED  -> PrefetchVirtualMemory (Win8+), which is the same hint to the memory
//                     manager: bring these pages resident now.
//   MADV_DONTNEED  -> DiscardVirtualMemory (Win8.1+), the closest equivalent (contents become
//                     undefined and the pages are dropped).
//   MADV_* others  -> no-op; they only steer the kernel readahead heuristic, which Windows
//                     does not expose.
//   mincore        -> unsupported; the only consumer is the expert hot-lock LRU, which treats
//                     an unknown residency as "skip this range".
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
#if defined(__linux__)
#include <fcntl.h>
#endif

namespace slim_arc {

namespace {
// [QLH port] mlock/munlock wrap VirtualLock/VirtualUnlock. The scheduler locks hot expert
// ranges so the memory manager keeps them resident; the win32 call takes the same
// (address, length) pair and fails the same way, so the call sites need no changes.
#if defined(_WIN32)
int mlock(const void * addr, size_t length) {
    return VirtualLock(const_cast<void *>(addr), length) ? 0 : -1;
}

int munlock(const void * addr, size_t length) {
    return VirtualUnlock(const_cast<void *>(addr), length) ? 0 : -1;
}

// 64x64 -> 128 multiply for the LFRU score. MSVC has no __int128, so the product is kept as
// a (high, low) pair; only comparison and equality are used.
struct wide_uint {
    uint64_t hi{0};
    uint64_t lo{0};

    bool operator==(const wide_uint & other) const { return hi == other.hi && lo == other.lo; }
    bool operator!=(const wide_uint & other) const { return !(*this == other); }
    bool operator<(const wide_uint & other) const {
        return hi < other.hi || (hi == other.hi && lo < other.lo);
    }
};

wide_uint wide_mul(uint64_t a, uint64_t b) {
    uint64_t hi = 0;
    const uint64_t lo = _umul128(a, b, &hi);
    return wide_uint{hi, lo};
}
#else
using wide_uint = unsigned __int128;

wide_uint wide_mul(uint64_t a, uint64_t b) {
    return static_cast<wide_uint>(a) * static_cast<wide_uint>(b);
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

// [QLH] expert_hot_locked_bytes_ is unsigned, and both places that maintain it subtract from it.
// A bookkeeping slip (an entry counted twice, or an eviction after a partially failed mlock) must
// not wrap the counter around: the wrapped value is huge, so the headroom below turns into "no
// room at all, ever", the eviction loop stops running, and the hot cache silently stops admitting
// experts for the rest of the process. Both operations go through these two helpers instead.
uint64_t hot_cache_release(uint64_t locked, uint64_t bytes) noexcept {
    return locked > bytes ? locked - bytes : 0;
}

uint64_t hot_cache_headroom(uint64_t budget, uint64_t locked) noexcept {
    return locked < budget ? budget - locked : 0;
}

void append_unique_nonnegative(std::vector<int> & ids, int id) {
    if (id >= 0 && std::find(ids.begin(), ids.end(), id) == ids.end()) {
        ids.push_back(id);
    }
}

// [QLH] The SLIM_ARC_* switches, parsed in one place.
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
// That is a visible difference for exactly one switch: SLIM_ARC_EXPERT_POP used to return 0
// silently and now logs before returning 0. The value is unchanged; the extra line is this
// consolidation showing through, not a new fault.
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
        std::fprintf(stderr, "SLIM-ARC: invalid %s; expected an integer in [%lld,%lld]\n", name,
                     static_cast<long long>(lo), static_cast<long long>(hi));
        return def;
    }
    return parsed;
}

struct resident_range_result {
    std::vector<page_range> ranges;
    uint64_t nonresident_bytes{0};
    bool valid{false};
};

resident_range_result resident_page_ranges(const page_range & range, size_t page_size) {
    if (!range.valid || range.length == 0 || page_size == 0 || range.length % page_size != 0) return {};
    const size_t page_count = range.length / page_size;
#if defined(__APPLE__)
    std::vector<char> residency(page_count);
#else
    std::vector<unsigned char> residency(page_count);
#endif
#if defined(_WIN32)
    // No mincore equivalent that answers "is this page resident" for a file mapping without
    // touching it. Returning an invalid result makes the sole caller (expert hot-lock LRU)
    // skip the range and count a lock failure instead of prefetching blindly.
    (void) residency;
    return {};
#else
    if (mincore(reinterpret_cast<void *>(range.address), range.length, residency.data()) != 0) return {};
#endif

    resident_range_result result;
    result.valid = true;
    size_t run_start = page_count;
    for (size_t page = 0; page <= page_count; ++page) {
        const bool resident = page < page_count && (residency[page] & 1) != 0;
        if (resident && run_start == page_count) run_start = page;
        if ((!resident || page == page_count) && run_start != page_count) {
            const size_t run_pages = page - run_start;
            result.ranges.push_back({
                range.address + run_start * page_size,
                run_pages * page_size,
                0,
                true,
                0});
            run_start = page_count;
        }
        if (page < page_count && !resident) {
            result.nonresident_bytes = saturating_add(result.nonresident_bytes, static_cast<uint64_t>(page_size));
        }
    }
    return result;
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

int system_file_advice(int file_id, uint64_t offset, size_t length) {
#if defined(__linux__)
    if (file_id < 0 || length == 0 ||
        offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
        length > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        return EINVAL;
    }
    return posix_fadvise(
        file_id,
        static_cast<off_t>(offset),
        static_cast<off_t>(length),
        POSIX_FADV_WILLNEED);
#else
    (void) file_id;
    (void) offset;
    (void) length;
    return ENOTSUP;
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

// [QLH] Free physical memory right now. The layer-ahead budget is derived from this because the
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

// [QLH] Ceiling for one prefetch request, in bytes.
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

int parse_cross_layer_transition_topk() noexcept {
    if (!env_bool("SLIM_ARC_CROSS_LAYER_TRANSITION", false)) return 0;
    // Two variables, so the "required" check has to stay separate: env_int() answers the default
    // for an unset variable, which would hide the one case this switch is supposed to report.
    if (std::getenv("SLIM_ARC_CROSS_LAYER_TRANSITION_TOPK") == nullptr) {
        std::fprintf(
            stderr,
            "SLIM-ARC: SLIM_ARC_CROSS_LAYER_TRANSITION_TOPK is required when transition is enabled\n");
        return 0;
    }
    return static_cast<int>(env_int("SLIM_ARC_CROSS_LAYER_TRANSITION_TOPK", 0, 1, 64));
}

void update_atomic_peak(std::atomic<uint64_t> & peak, uint64_t candidate) noexcept {
    uint64_t current = peak.load();
    while (current < candidate && !peak.compare_exchange_weak(current, candidate)) {
    }
}

uint32_t ratio_permille(uint64_t numerator, uint64_t denominator) noexcept {
    if (denominator == 0 || numerator == 0) return 0;
    if (numerator >= denominator) return 1000;
    uint32_t low = 0;
    uint32_t high = 1000;
    while (low < high) {
        const uint32_t middle = low + (high - low + 1) / 2;
        const uint64_t whole = (denominator / 1000) * middle;
        const uint64_t remainder_product = (denominator % 1000) * middle;
        const uint64_t threshold = whole + (remainder_product + 999) / 1000;
        if (numerator >= threshold) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

struct expert_page_candidate {
    int expert_id;
    page_range range;
};

struct expert_page_owner {
    uintptr_t address;
    size_t length;
    int expert_id;
};

std::vector<expert_page_owner> assign_page_owners(
    const std::vector<expert_page_candidate> & candidates) {
    struct event {
        uintptr_t position;
        int expert_id;
        int delta;
    };
    std::vector<event> events;
    events.reserve(candidates.size() * 2);
    for (const expert_page_candidate & candidate : candidates) {
        events.push_back({candidate.range.address, candidate.expert_id, 1});
        events.push_back({
            candidate.range.address + candidate.range.length,
            candidate.expert_id,
            -1});
    }
    std::sort(events.begin(), events.end(), [](const event & left, const event & right) {
        if (left.position != right.position) return left.position < right.position;
        if (left.expert_id != right.expert_id) return left.expert_id < right.expert_id;
        return left.delta < right.delta;
    });

    std::vector<expert_page_owner> owners;
    if (events.empty()) return owners;
    std::map<int, size_t> active;
    uintptr_t previous = events.front().position;
    size_t index = 0;
    while (index < events.size()) {
        const uintptr_t position = events[index].position;
        if (position > previous && !active.empty()) {
            const int owner = active.begin()->first;
            const size_t length = static_cast<size_t>(position - previous);
            if (!owners.empty() &&
                owners.back().expert_id == owner &&
                owners.back().address + owners.back().length == previous) {
                owners.back().length += length;
            } else {
                owners.push_back({previous, length, owner});
            }
        }
        while (index < events.size() && events[index].position == position) {
            const event & current = events[index];
            if (current.delta > 0) {
                ++active[current.expert_id];
            } else {
                const auto found = active.find(current.expert_id);
                if (found != active.end() && --found->second == 0) active.erase(found);
            }
            ++index;
        }
        previous = position;
    }
    return owners;
}
}

std::vector<size_t> select_prefetch_items(
    const std::vector<size_t> & item_sizes,
    uint64_t budget_bytes,
    uint64_t * requested_bytes,
    uint64_t * skipped_bytes) {
    uint64_t requested{0};
    uint64_t skipped{0};
    uint64_t remaining = budget_bytes;
    std::vector<size_t> selected;
    selected.reserve(item_sizes.size());
    for (size_t index = 0; index < item_sizes.size(); ++index) {
        const uint64_t size = item_sizes[index];
        requested = saturating_add(requested, size);
        if (size > 0 && size <= remaining) {
            selected.push_back(index);
            remaining -= size;
        } else {
            skipped = saturating_add(skipped, size);
        }
    }
    if (requested_bytes != nullptr) {
        *requested_bytes = requested;
    }
    if (skipped_bytes != nullptr) {
        *skipped_bytes = skipped;
    }
    return selected;
}

void prefetch_scheduler::set_phase(compute_phase phase) {
    if (stop_.load()) return;
    phase_.store(phase);
    const char * dyn = getenv("SLIM_ARC_DYNAMIC_MADV");
    if (dyn != nullptr && std::strcmp(dyn, "0") == 0) return;
    int advice = POSIX_MADV_SEQUENTIAL;
    if (phase == compute_phase::DECODE) {
        // SLIM-ARC FIX 2026-08-07: the decode-phase advice is configurable (ablation
        // result: at the extreme RK3588 ratio of 45GB/8GB, RANDOM actually made decode
        // 5.4x slower, so the default is SEQUENTIAL -- keeping the kernel's sequential
        // readahead to use the SSD's sequential bandwidth; SLIM_ARC_DECODE_MADV=
        // RANDOM/NORMAL overrides it. RANDOM only suits a machine with memory to spare,
        // where the model can stay partly resident).
        const char * dec = getenv("SLIM_ARC_DECODE_MADV");
        if (dec != nullptr && std::strcmp(dec, "RANDOM") == 0) {
            advice = POSIX_MADV_RANDOM;
        } else if (dec != nullptr && std::strcmp(dec, "NORMAL") == 0) {
            advice = POSIX_MADV_NORMAL;
        }
    }
    std::vector<std::pair<void *, size_t>> regions;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        regions.reserve(mmap_regions_.size());
        for (const mmap_region_info & region : mmap_regions_) {
            regions.emplace_back(region.addr, region.size);
        }
    }
    for (const auto & region : regions) {
        (void) advice_(region.first, region.second, advice);
    }
    if (phase == compute_phase::DECODE && (expert_random_madv_enabled_ || expert_normal_madv_enabled_)) {
        std::vector<page_range> expert_ranges;
        {
            std::lock_guard<std::mutex> lock(expert_state_mtx_);
            expert_ranges = expert_madv_ranges_;
        }
        const int expert_advice = expert_random_madv_enabled_ ? POSIX_MADV_RANDOM : POSIX_MADV_NORMAL;
        for (const page_range & range : expert_ranges) {
            atomic_saturating_add(expert_madv_advice_calls_, uint64_t{1});
            if (advice_(reinterpret_cast<void *>(range.address), range.length, expert_advice) == 0) {
                atomic_saturating_add(
                    expert_madv_advice_bytes_,
                    static_cast<uint64_t>(range.length));
            } else {
                atomic_saturating_add(expert_madv_advice_failures_, uint64_t{1});
            }
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

// [QLH] SLIM-ARC 2026-11-02: graph node names carry their layer as a trailing "-N" suffix --
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
    page_size_query_fn page_size_query,
    file_advice_fn file_advice)
    : slow_storage_enabled_(env_bool("SLIM_ARC_SLOW_STORAGE", false))
    , router_prefetch_enabled_(env_bool("SLIM_ARC_ROUTER_PREFETCH", false))
    , router_mlock_enabled_(env_bool("SLIM_ARC_ROUTER_MLOCK", false))
    , shared_mlock_enabled_(env_bool("SLIM_ARC_SHARED_MLOCK", false))
    , small_mlock_enabled_(env_bool("SLIM_ARC_SMALL_MLOCK", false))
    , expert_hot_budget_bytes_(
          static_cast<size_t>(env_int("SLIM_ARC_EXPERT_HOT_MB", 0, 1, 1024)) << 20)
    , expert_hot_lru_enabled_(env_bool("SLIM_ARC_EXPERT_HOT_LRU", false))
    , expert_hot_lfru_enabled_(expert_hot_lru_enabled_ && env_bool("SLIM_ARC_EXPERT_HOT_LFRU", false))
    , expert_hot_admission_threshold_(
          static_cast<uint32_t>(env_int("SLIM_ARC_EXPERT_HOT_ADMIT_HITS", 1, 1, 64)))
    , expert_prefetch_disabled_(env_bool("SLIM_ARC_NO_EXPERT_PREFETCH", false))
    , expert_file_advice_enabled_(env_bool("SLIM_ARC_EXPERT_FADVISE", false))
    , expert_random_madv_enabled_(env_bool("SLIM_ARC_EXPERT_MADV_RANDOM", false))
    , expert_normal_madv_enabled_(env_bool("SLIM_ARC_EXPERT_MADV_NORMAL", false))
    , cross_layer_transition_topk_(parse_cross_layer_transition_topk())
    , cross_layer_transition_enabled_(cross_layer_transition_topk_ > 0)
    , n_threads_(slow_storage_enabled_ ? 1 : std::max(1, n_threads))
    , window_(slow_storage_enabled_ ? 1 : std::max(1, window))
    , request_claim_hook_(std::move(request_claim_hook))
    , advice_(advice ? std::move(advice) : advice_fn{system_advice})
    , page_size_query_(page_size_query ? std::move(page_size_query) : page_size_query_fn{system_page_size})
    , file_advice_(file_advice ? std::move(file_advice) : file_advice_fn{system_file_advice})
    , reclaim_waste_enabled_(env_bool("SLIM_ARC_EXPERT_RECLAIM_WASTE", false))
    , expert_residency_enabled_(env_bool("SLIM_ARC_EXPERT_RESIDENCY", false)) {
    conf_gating_ = env_bool("SLIM_ARC_EXPERT_CONF", false);
    pop_k_ = static_cast<int>(env_int("SLIM_ARC_EXPERT_POP", 0, 0, 64));
    // [QLH] See default_prefetch_budget_bytes(): the budget used to stay 0 because the module
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
    std::vector<page_range> expert_hot_locked;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        locked.swap(router_locked_ranges_);
        shared_locked.swap(shared_locked_ranges_);
        small_locked.swap(small_locked_ranges_);
    }
    {
        std::lock_guard<std::mutex> lock(expert_hot_mtx_);
        for (const expert_hot_entry & entry : expert_hot_entries_) {
            expert_hot_locked.insert(expert_hot_locked.end(), entry.ranges.begin(), entry.ranges.end());
        }
        expert_hot_entries_.clear();
        expert_hot_locked_bytes_ = 0;
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
    for (const page_range & range : expert_hot_locked) {
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

    // [QLH] Every decision that needs the page size, and every mlock() syscall, is taken before
    // mtx_ is acquired: mlock can block on a page fault, and mtx_ is also taken by eval_cb() on
    // the compute thread, so holding it across the syscall would park inference behind a model
    // load. Only the bookkeeping below runs under the lock.
    const bool lock_shared = shared_mlock_enabled_ && name != nullptr &&
        std::strstr(name, "_shexp") != nullptr;
    const bool lock_small = small_mlock_enabled_ && name != nullptr && size <= (1ULL << 20) &&
        std::strstr(name, "_exps") == nullptr && std::strstr(name, "_shexp") == nullptr &&
        std::strstr(name, "ffn_gate_inp") == nullptr;
    // router 张量走独立的两张表，不进 tensors_by_layer_；但 return 只能在 is_router 时执行——
    // 早先写成「开了 router_prefetch 或 router_mlock 就整块 return」，于是单独开
    // SLIM_ARC_ROUTER_MLOCK=1 会把所有非 router 张量一起吞掉，tensors_by_layer_ 永远为空、
    // 层预取一条都不发，而日志里没有任何错。
    //
    // [QLH] The gate is `router_mode && is_router`, not just `is_router`. With both router
    // switches off (the default configuration) the router weight is an ordinary layer tensor and
    // must stay in tensors_by_layer_: an unconditional `if (is_router) return;` would silently
    // drop it from layer-ahead prefetch exactly when the router path is switched *off*. Do not
    // "simplify" this back -- the only case that may return early is the one where the tensor was
    // actually recorded in router_tensors_ / router_locked_ranges_ above.
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

bool prefetch_scheduler::register_mapping(void * addr, size_t size, int file_id) {
    if (addr == nullptr || size == 0) return false;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (stop_.load()) return false;
        mmap_regions_.push_back({addr, size, file_id});
    }
    const char * dyn = std::getenv("SLIM_ARC_DYNAMIC_MADV");
    if (dyn == nullptr || std::strcmp(dyn, "0") != 0) {
        (void) advice_(addr, size, POSIX_MADV_SEQUENTIAL);
    }
    return true;
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
    // changed because the tensors were not rebuilt. [QLH] The idempotency check is inside the
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
        std::fprintf(stderr, "[SLIM-ARC-PREFETCH] layer signalling disabled: no layer boundaries "
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

    std::fprintf(stderr, "[SLIM-ARC-PREFETCH] layer signalling armed: %zu boundaries over %d "
            "graph nodes, layers %d..%d\n", observe_.size(), n_nodes,
            *std::min_element(observe_layer_.begin(), observe_layer_.end()),
            *std::max_element(observe_layer_.begin(), observe_layer_.end()));
}

void prefetch_scheduler::on_layer_boundary(int32_t layer) {
    if (layer < 0) return;
    // [QLH] last_notified_ is written here (compute thread) and reset by prepare_graph()
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

    // [QLH] The chained user callback (attach_user_callback) and the layer-boundary tables
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
    // [QLH] 2026-11-02 A/B: SLIM_ARC_NOOP=1 keeps the eval callback installed and keeps the
    // graph splitting it causes, but issues no PrefetchVirtualMemory at all. This separates the
    // cost of "callback + 40 split submits" from the cost of "actually advising pages".
    if (std::getenv("SLIM_ARC_NOOP") != nullptr) return;
    // [QLH] 2026-11-02 A/B: SLIM_ARC_MAX_CALLS=N bounds how many prefetch requests a process
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
    // [QLH] Per-request ceiling, part 2: never advise more than one layer's worth of weights in a
    // single request. The window covers `window_` layers, so the request sizes are derived from
    // the actual bytes registered per layer rather than from a constant -- a dense 70B and a
    // small MoE get correctly different budgets. Workers still walk the ranges in order, so
    // "one layer" means the nearest one, which is the one the next step needs first.
    // [QLH] SLIM-ARC 2026-11-02: with per-layer signalling (on_layer_boundary()) the next
    // notification is a whole window further along, so clipping every request to a single layer
    // would leave layers 2..window of each window prefetched never. The request may spend the
    // whole window; the free-memory ceiling in memory_budget_ still caps it.
    if (window_layers > 0) {
        const uint64_t per_layer = request.requested_bytes / static_cast<uint64_t>(window_layers);
        // [QLH] 2026-11-02 A/B: cap the request to ONE layer, not to the whole window.
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

size_t prefetch_scheduler::pending_request_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pending_requests_.size();
}

prefetch_budget_stats prefetch_scheduler::budget_stats() const {
    return {
        budget_requested_bytes_.load(),
        budget_issued_bytes_.load(),
        budget_skipped_bytes_.load(),
        budget_rounds_throttled_.load(),
        budget_madvise_failures_.load(),
        budget_advice_requests_.load(),
        budget_coalesced_ranges_.load(),
        budget_covered_bytes_.load(),
        budget_invalid_ranges_.load(),
        budget_stale_requests_.load(),
        budget_stale_bytes_.load(),
        budget_inflight_peak_bytes_.load(),
    };
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

// ============================================================================================
// [QLH] NOT WIRED UP -- kept for future work, not for current behaviour.
//
// WHAT THIS IS: the complete MoE expert path -- expert tensor registration, router observation,
// cross-layer transition prediction, hot-expert residency (mlock), page reclaim, popularity /
// confidence gating and the residency budget. It was ported along with SLIM-ARC so that a later
// change can switch it on without re-deriving any of it.
//
// CURRENT STATE: nothing in this repository calls it. Every entry point below has exactly zero
// external references in src/, tools/, common/ and examples/:
//     register_expert_tensor   cache_router_experts   prefetch_experts   cancel_expert_prefetch
//     set_expert_residency_pressure   set_expert_budget   reset_expert_budget_usage
//     budget_stats   expert_*_statistics
// All three consequences are silent:
//   - the expert counters printed by dump_metrics() ([SLIM-ARC-METRICS], [SLIM-ARC-HOT],
//     [SLIM-ARC-TRANSITION], [SLIM-ARC-EXPERT-MADV]) are always zero, so the summary cannot
//     distinguish "not wired" from "wired and idle";
//   - expert_pressure_ only ever changes through set_expert_residency_pressure(), which is also
//     never called, so it stays expert_pressure_state::missing; select_resident_experts() then
//     takes its "no pressure signal" path and admits every candidate that fits the budget, i.e.
//     the residency policy degenerates to pass-through even with SLIM_ARC_EXPERT_RESIDENCY=1;
//   - SLIM_ARC_EXPERT_HOT_MB has no effect, because update_expert_hot_cache() is only reachable
//     from cache_router_experts().
//
// HOW TO WIRE IT (in this order):
//   1. llama-model-loader: next to the existing register_tensor() call, call
//      register_expert_tensor(name, addr, ggml_nbytes(t), layer, n_experts = ne[2]) for every
//      merged *_exps tensor, and register_mapping() for the mapping it lives in (the file-offset
//      advice path needs the file id).
//   2. Feed a real pressure source: sample expert-page residency (mincore() on POSIX,
//      QueryWorkingSetEx() on Windows) into an expert_pressure_sample and drive
//      expert_pressure_controller::update(). That controller is NOT a member of
//      prefetch_scheduler yet -- set_expert_residency_pressure() currently takes the state as a
//      parameter, so the controller has to be added here and its return value passed instead.
//   3. On the layer boundary that already exists (on_layer_boundary / notify_layer_compute),
//      call prefetch_experts(layer, ids, n) with the router's selection, then settle the same
//      generation with cache_router_experts(layer, ids, n, generation).
// Until all three exist, treat everything below as unreachable and do not read its counters as
// evidence about a run.
// ============================================================================================
// ---- SLIM-ARC FIX 2026-08-05: Phase 2a MoE expert prefetch ----
void prefetch_scheduler::register_expert_tensor(const char *, void * addr, size_t size, int layer, int n_experts) {
    if (addr == nullptr || size == 0 || n_experts < 1 || layer < 0) return;
    if (size % static_cast<size_t>(n_experts) != 0) return;
    page_range random_range;
    if (expert_random_madv_enabled_ || expert_normal_madv_enabled_) {
        const long raw_page_size = page_size_query_();
        random_range = covering_page_range(
            reinterpret_cast<uintptr_t>(addr),
            size,
            raw_page_size > 0 ? static_cast<size_t>(raw_page_size) : 0);
    }
    if (cross_layer_transition_enabled_) {
        std::lock_guard<std::mutex> transition_lock(expert_transition_mtx_);
        (void) expert_transition_table_.register_layer(layer, n_experts);
    }
    int file_id = -1;
    uint64_t file_offset = 0;
    if (expert_file_advice_enabled_) {
        const uintptr_t tensor_start = reinterpret_cast<uintptr_t>(addr);
        std::lock_guard<std::mutex> mapping_lock(mtx_);
        for (const mmap_region_info & mapping : mmap_regions_) {
            const uintptr_t mapping_start = reinterpret_cast<uintptr_t>(mapping.addr);
            if (mapping_start == 0 || tensor_start < mapping_start ||
                tensor_start - mapping_start > mapping.size ||
                size > mapping.size - (tensor_start - mapping_start)) {
                continue;
            }
            file_id = mapping.file_id;
            file_offset = static_cast<uint64_t>(tensor_start - mapping_start);
            break;
        }
    }
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    if ((size_t)layer >= experts_by_layer_.size()) {
        experts_by_layer_.resize(layer + 1);
    }
    experts_by_layer_[layer].push_back({addr, size, n_experts, file_id, file_offset});
    if (random_range.valid) expert_madv_ranges_.push_back(random_range);
}

void prefetch_scheduler::observe_expert_transition(
    int layer,
    const std::vector<int> & source,
    const std::vector<int> & target) {
    if (!cross_layer_transition_enabled_ || source.empty() || target.empty()) return;
    std::lock_guard<std::mutex> lock(expert_transition_mtx_);
    expert_transition_table_.observe(
        layer,
        source.data(),
        static_cast<int>(source.size()),
        target.data(),
        static_cast<int>(target.size()));
}

std::vector<int> prefetch_scheduler::predict_expert_transition(
    int layer, const std::vector<int> & source) {
    if (!cross_layer_transition_enabled_ || source.empty()) return {};
    std::lock_guard<std::mutex> lock(expert_transition_mtx_);
    return expert_transition_table_.predict(
        layer,
        source.data(),
        static_cast<int>(source.size()),
        cross_layer_transition_topk_);
}

void prefetch_scheduler::record_expert_transition_result(
    const std::vector<int> & predicted,
    const std::vector<int> & actual) {
    if (!cross_layer_transition_enabled_ || predicted.empty() || actual.empty()) return;
    std::lock_guard<std::mutex> lock(expert_transition_mtx_);
    expert_transition_table_.record_result(
        predicted.data(),
        static_cast<int>(predicted.size()),
        actual.data(),
        static_cast<int>(actual.size()));
}

expert_transition_stats prefetch_scheduler::expert_transition_statistics() const noexcept {
    std::lock_guard<std::mutex> lock(expert_transition_mtx_);
    return expert_transition_table_.statistics();
}

void prefetch_scheduler::cache_router_experts(int layer, const int * expert_ids, int n) {
    cache_router_experts(layer, expert_ids, n, 0);
}

void prefetch_scheduler::cache_router_experts(
    int layer, const int * expert_ids, int n, uint64_t generation) {
    if (layer < 0 || expert_ids == nullptr || n <= 0) return;
    if (stop_.load()) return;
    std::vector<int> current;
    current.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        append_unique_nonnegative(current, expert_ids[i]);
    }

    std::vector<int> prefetched;
    std::vector<expert_tensor_info> reclaim_tensors;
    std::vector<int> stable_experts;
    std::vector<expert_tensor_info> hot_tensors;
    std::vector<expert_tensor_info> file_tensors;
    {
        std::lock_guard<std::mutex> lock(expert_state_mtx_);
        if ((size_t)layer >= cached_router_experts_.size()) {
            cached_router_experts_.resize(layer + 1);
        }
        if (expert_hot_budget_bytes_ > 0) {
            for (const int expert_id : current) {
                if (std::find(
                        cached_router_experts_[layer].begin(),
                        cached_router_experts_[layer].end(),
                        expert_id) != cached_router_experts_[layer].end()) {
                    stable_experts.push_back(expert_id);
                }
            }
            if (!stable_experts.empty() && static_cast<size_t>(layer) < experts_by_layer_.size()) {
                hot_tensors = experts_by_layer_[layer];
            }
        }
        if (expert_file_advice_enabled_ && !current.empty() &&
            static_cast<size_t>(layer) < experts_by_layer_.size()) {
            file_tensors = experts_by_layer_[layer];
        }
        // Each router observation can only settle the exact prefetch generation the
        // caller carries; token 0 merely updates the predictor.
        if (generation != 0 && static_cast<size_t>(layer) < pending_expert_prefetches_.size()) {
            std::vector<expert_prefetch_record> & pending = pending_expert_prefetches_[layer];
            const auto it = std::find_if(pending.begin(), pending.end(), [generation](const auto & record) {
                return record.generation == generation;
            });
            if (it != pending.end()) {
                size_t hit_bytes = 0;
                size_t waste_bytes = 0;
                for (size_t index = 0; index < it->expert_ids.size(); ++index) {
                    if (std::find(current.begin(), current.end(), it->expert_ids[index]) != current.end()) {
                        hit_bytes = saturating_add(hit_bytes, it->issued_bytes[index]);
                    } else {
                        waste_bytes = saturating_add(waste_bytes, it->issued_bytes[index]);
                    }
                }
                if (reclaim_waste_enabled_ && enabled_.load()) {
                    prefetched = it->expert_ids;
                    if (static_cast<size_t>(layer) < experts_by_layer_.size()) {
                        reclaim_tensors = experts_by_layer_[layer];
                    }
                }
                atomic_saturating_add(expert_hit_bytes_, hit_bytes);
                atomic_saturating_add(expert_waste_bytes_, waste_bytes);
                const uint64_t accounted = saturating_add(
                    static_cast<uint64_t>(hit_bytes), static_cast<uint64_t>(waste_bytes));
                if (accounted > 0) {
                    const uint32_t sample_milli = ratio_permille(static_cast<uint64_t>(waste_bytes), accounted);
                    expert_waste_ewma_milli_ = update_waste_ewma_milli(
                        expert_waste_ewma_milli_, sample_milli, expert_waste_ewma_initialized_);
                    expert_waste_ewma_initialized_ = true;
                    expert_waste_samples_ = saturating_add(expert_waste_samples_, uint64_t{1});
                    expert_waste_restricted_ = expert_waste_controller_.update(sample_milli);
                }
                pending.erase(it);
            } else {
                atomic_saturating_add(expert_unmatched_generations_, uint64_t{1});
            }
        } else if (generation != 0) {
            atomic_saturating_add(expert_unmatched_generations_, uint64_t{1});
        }
        atomic_saturating_add(router_samples_, uint64_t{1});
        // SLIM-ARC FIX 2026-08-09: 2-token history sliding (used by the improvement-2
        // gate). prev = old cached (t-2), cached = new routing (t-1)
        if ((size_t)layer >= prev_router_experts_.size()) {
            prev_router_experts_.resize(cached_router_experts_.size());
        }
        prev_router_experts_[layer] = cached_router_experts_[layer];
        // SLIM-ARC FIX 2026-08-09: accumulate hot-expert frequencies (improvement 4)
        if ((size_t)layer >= expert_pop_counts_.size()) {
            expert_pop_counts_.resize(cached_router_experts_.size());
        }
        int n_experts = 0;
        if ((size_t)layer < experts_by_layer_.size()) {
            for (const auto & e : experts_by_layer_[layer]) {
                n_experts = std::max(n_experts, e.n_experts);
            }
        }
        if ((int)expert_pop_counts_[layer].size() < n_experts) {
            expert_pop_counts_[layer].resize(n_experts, 0);
        }
        for (int eid : current) {
            if (eid >= 0 && eid < (int)expert_pop_counts_[layer].size()) {
                expert_pop_counts_[layer][eid] = saturating_increment_popularity(expert_pop_counts_[layer][eid]);
            }
        }
        if (expert_residency_enabled_ && !current.empty()) ++popularity_samples_since_decay_;
        if (expert_residency_enabled_ && popularity_samples_since_decay_ == 64) {
            popularity_samples_since_decay_ = 0;
            for (auto & layer_counts : expert_pop_counts_) {
                for (uint32_t & count : layer_counts) count /= 2;
            }
            atomic_saturating_add(popularity_decay_count_, uint64_t{1});
        }
        cached_router_experts_[layer] = current;
    }
    if (!prefetched.empty()) {
        reclaim_wrong_expert_pages(prefetched, current, reclaim_tensors);
    }
    if (!file_tensors.empty()) {
        advise_selected_expert_files(current, file_tensors);
    }
    if (!stable_experts.empty() && !hot_tensors.empty()) {
        update_expert_hot_cache(layer, stable_experts, hot_tensors);
    }
}

void prefetch_scheduler::cancel_expert_prefetch(int layer, uint64_t generation) {
    if (generation == 0) return;

    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    if (layer < 0 || static_cast<size_t>(layer) >= pending_expert_prefetches_.size()) {
        atomic_saturating_add(expert_unmatched_generations_, uint64_t{1});
        return;
    }

    std::vector<expert_prefetch_record> & pending = pending_expert_prefetches_[layer];
    const auto it = std::find_if(pending.begin(), pending.end(), [generation](const auto & record) {
        return record.generation == generation;
    });
    if (it == pending.end()) {
        atomic_saturating_add(expert_unmatched_generations_, uint64_t{1});
        return;
    }

    size_t waste_bytes = 0;
    for (size_t issued_bytes : it->issued_bytes) {
        waste_bytes = saturating_add(waste_bytes, issued_bytes);
    }
    atomic_saturating_add(expert_waste_bytes_, waste_bytes);
    pending.erase(it);
}

std::vector<int> prefetch_scheduler::cached_experts_snapshot(int layer) const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    if (layer < 0 || static_cast<size_t>(layer) >= cached_router_experts_.size()) {
        return {};
    }
    return cached_router_experts_[layer];
}

size_t prefetch_scheduler::pending_expert_records(int layer) const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    if (layer < 0 || static_cast<size_t>(layer) >= pending_expert_prefetches_.size()) {
        return 0;
    }
    return pending_expert_prefetches_[layer].size();
}

expert_reclaim_stats prefetch_scheduler::expert_reclaim_statistics() const {
    return {
        reclaim_candidate_experts_.load(),
        reclaim_calls_.load(),
        reclaim_reclaimed_bytes_.load(),
        reclaim_skipped_bytes_.load(),
        reclaim_madvise_failures_.load(),
        reclaim_invalid_layouts_.load(),
        reclaim_invalid_ids_.load(),
    };
}

void prefetch_scheduler::set_expert_residency_pressure(
    expert_pressure_state pressure, size_t budget_bytes) {
    if (!expert_residency_enabled_) return;
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    expert_residency_budget_ = budget_bytes;
    expert_pressure_ = pressure;
    expert_residency_snapshot_set_ = true;
}

expert_pressure_state prefetch_scheduler::current_expert_pressure() const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    return expert_pressure_;
}

expert_residency_runtime_stats prefetch_scheduler::expert_residency_statistics() const noexcept {
    return {
        residency_samples_.load(),
        residency_admitted_experts_.load(),
        residency_admitted_bytes_.load(),
        residency_skipped_bytes_.load(),
        residency_fallbacks_.load(),
        residency_pressure_missing_.load(),
        residency_pressure_normal_.load(),
        residency_pressure_high_.load(),
        residency_pressure_critical_.load(),
    };
}

expert_runtime_metrics prefetch_scheduler::expert_runtime_statistics() const noexcept {
    return {
        router_samples_.load(),
        static_cast<uint64_t>(expert_prefetch_bytes_.load()),
        static_cast<uint64_t>(expert_hit_bytes_.load()),
        static_cast<uint64_t>(expert_waste_bytes_.load()),
        expert_advice_requests_.load(),
        expert_coalesced_ranges_.load(),
        expert_covered_bytes_.load(),
        expert_advice_failures_.load(),
        expert_invalid_ranges_.load(),
    };
}

std::vector<uint32_t> prefetch_scheduler::expert_popularity_snapshot(int layer) const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    if (layer < 0 || static_cast<size_t>(layer) >= expert_pop_counts_.size()) return {};
    return expert_pop_counts_[layer];
}

uint32_t prefetch_scheduler::expert_waste_ewma_milli() const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    return expert_waste_ewma_milli_;
}

uint64_t prefetch_scheduler::expert_waste_sample_count() const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    return expert_waste_samples_;
}

bool prefetch_scheduler::expert_waste_restricted() const {
    std::lock_guard<std::mutex> lock(expert_state_mtx_);
    return expert_waste_restricted_;
}

expert_hot_cache_stats prefetch_scheduler::expert_hot_cache_statistics() const noexcept {
    std::lock_guard<std::mutex> lock(expert_hot_mtx_);
    return {
        static_cast<uint64_t>(expert_hot_budget_bytes_),
        expert_hot_locked_bytes_,
        expert_hot_admissions_,
        expert_hot_hits_,
        expert_hot_evictions_,
        expert_hot_budget_rejections_,
        expert_hot_nonresident_bytes_,
        expert_hot_lock_failures_,
        static_cast<uint64_t>(expert_hot_entries_.size()),
        expert_hot_admission_skips_,
        expert_hot_admission_threshold_};
}

expert_file_advice_stats prefetch_scheduler::expert_file_advice_statistics() const noexcept {
    return {
        expert_file_advice_calls_.load(),
        expert_file_advice_bytes_.load(),
        expert_file_advice_failures_.load(),
        expert_file_advice_invalid_ranges_.load()};
}

void prefetch_scheduler::advise_selected_expert_files(
    const std::vector<int> & selected,
    const std::vector<expert_tensor_info> & tensors) {
    if (!expert_file_advice_enabled_ || selected.empty() || stop_.load()) return;
    for (const expert_tensor_info & tensor : tensors) {
        if (tensor.file_id < 0 || tensor.size == 0 || tensor.n_experts < 1 ||
            tensor.size % static_cast<size_t>(tensor.n_experts) != 0) {
            atomic_saturating_add(expert_file_advice_invalid_ranges_, uint64_t{1});
            continue;
        }
        const size_t expert_bytes = tensor.size / static_cast<size_t>(tensor.n_experts);
        for (const int expert_id : selected) {
            if (expert_id < 0 || expert_id >= tensor.n_experts ||
                static_cast<uint64_t>(expert_id) >
                    (std::numeric_limits<uint64_t>::max() - tensor.file_offset) / expert_bytes) {
                atomic_saturating_add(expert_file_advice_invalid_ranges_, uint64_t{1});
                continue;
            }
            const uint64_t offset = tensor.file_offset + static_cast<uint64_t>(expert_id) * expert_bytes;
            atomic_saturating_add(expert_file_advice_calls_, uint64_t{1});
            if (file_advice_(tensor.file_id, offset, expert_bytes) == 0) {
                atomic_saturating_add(expert_file_advice_bytes_, static_cast<uint64_t>(expert_bytes));
            } else {
                atomic_saturating_add(expert_file_advice_failures_, uint64_t{1});
            }
        }
    }
}

void prefetch_scheduler::update_expert_hot_cache(
    int layer,
    const std::vector<int> & stable_experts,
    const std::vector<expert_tensor_info> & tensors) {
    if (expert_hot_budget_bytes_ == 0 || stable_experts.empty() || tensors.empty() || stop_.load()) return;
    const long raw_page_size = page_size_query_();
    if (raw_page_size <= 0) return;
    const size_t page_size = static_cast<size_t>(raw_page_size);

    std::vector<expert_tensor_view> tensor_views;
    tensor_views.reserve(tensors.size());
    for (const expert_tensor_info & tensor : tensors) {
        tensor_views.push_back({reinterpret_cast<uintptr_t>(tensor.addr), tensor.size, tensor.n_experts});
    }

    std::lock_guard<std::mutex> lock(expert_hot_mtx_);
    const uint64_t admission_headroom = std::max<uint64_t>(
        static_cast<uint64_t>(expert_hot_budget_bytes_) / 100, uint64_t{1});
    const bool filter_saturated_admission = expert_hot_admission_threshold_ > 1 &&
        expert_hot_locked_bytes_ >= static_cast<uint64_t>(expert_hot_budget_bytes_) - admission_headroom;
    if (filter_saturated_admission) {
        int n_experts = 0;
        for (const expert_tensor_info & tensor : tensors) {
            n_experts = std::max(n_experts, tensor.n_experts);
        }
        if (static_cast<size_t>(layer) >= expert_hot_admission_counts_.size()) {
            expert_hot_admission_counts_.resize(static_cast<size_t>(layer) + 1);
        }
        if (n_experts > 0 && expert_hot_admission_counts_[layer].size() < static_cast<size_t>(n_experts)) {
            expert_hot_admission_counts_[layer].resize(static_cast<size_t>(n_experts), 0);
        }
    }
    if (!expert_hot_lru_enabled_) {
        auto entry = expert_hot_entries_.begin();
        while (entry != expert_hot_entries_.end()) {
            if (entry->layer != layer ||
                std::find(stable_experts.begin(), stable_experts.end(), entry->expert_id) != stable_experts.end()) {
                ++entry;
                continue;
            }
            for (const page_range & range : entry->ranges) {
                (void) munlock(reinterpret_cast<void *>(range.address), range.length);
            }
            expert_hot_locked_bytes_ = hot_cache_release(expert_hot_locked_bytes_, entry->locked_bytes);
            expert_hot_evictions_ = saturating_add(expert_hot_evictions_, uint64_t{1});
            entry = expert_hot_entries_.erase(entry);
        }
    }

    for (const int expert_id : stable_experts) {
        uint32_t admission_count = expert_hot_admission_threshold_;
        if (filter_saturated_admission) {
            if (expert_id < 0 || static_cast<size_t>(expert_id) >= expert_hot_admission_counts_[layer].size()) {
                continue;
            }
            uint32_t & observed = expert_hot_admission_counts_[layer][expert_id];
            observed = saturating_add(observed, uint32_t{1});
            admission_count = observed;
        }
        const auto existing = std::find_if(
            expert_hot_entries_.begin(), expert_hot_entries_.end(), [layer, expert_id](const expert_hot_entry & item) {
                return item.layer == layer && item.expert_id == expert_id;
            });
        if (existing != expert_hot_entries_.end()) {
            expert_hot_hits_ = saturating_add(expert_hot_hits_, uint64_t{1});
            existing->frequency = saturating_add(existing->frequency, uint64_t{1});
            if (expert_hot_touch_clock_ != std::numeric_limits<uint64_t>::max()) ++expert_hot_touch_clock_;
            existing->last_touch = expert_hot_touch_clock_;
            continue;
        }
        if (filter_saturated_admission && admission_count < expert_hot_admission_threshold_) {
            expert_hot_admission_skips_ = saturating_add(expert_hot_admission_skips_, uint64_t{1});
            continue;
        }

        const expert_reclaim_plan plan = build_expert_reclaim_plan(tensor_views, {expert_id}, page_size);
        std::vector<page_range> resident_ranges;
        uint64_t candidate_bytes{0};
        bool valid = !plan.items.empty();
        for (const expert_reclaim_item & item : plan.items) {
            const resident_range_result resident = resident_page_ranges(
                {item.address, item.length, 0, true, 0}, page_size);
            if (!resident.valid) {
                valid = false;
                break;
            }
            expert_hot_nonresident_bytes_ = saturating_add(
                expert_hot_nonresident_bytes_, resident.nonresident_bytes);
            for (const page_range & range : resident.ranges) {
                candidate_bytes = saturating_add(candidate_bytes, static_cast<uint64_t>(range.length));
                resident_ranges.push_back(range);
            }
        }
        if (!valid || resident_ranges.empty()) {
            expert_hot_lock_failures_ = saturating_add(expert_hot_lock_failures_, uint64_t{1});
            continue;
        }
        if (candidate_bytes > static_cast<uint64_t>(expert_hot_budget_bytes_)) {
            expert_hot_budget_rejections_ = saturating_add(expert_hot_budget_rejections_, uint64_t{1});
            continue;
        }
        if (expert_hot_lru_enabled_) {
            while (candidate_bytes > hot_cache_headroom(
                       static_cast<uint64_t>(expert_hot_budget_bytes_), expert_hot_locked_bytes_)) {
                auto victim = expert_hot_entries_.end();
                for (auto entry = expert_hot_entries_.begin(); entry != expert_hot_entries_.end(); ++entry) {
                    const bool selected_now = entry->layer == layer && std::find(
                        stable_experts.begin(), stable_experts.end(), entry->expert_id) != stable_experts.end();
                    if (selected_now) continue;
                    const auto lower_lfru_score = [this](
                                                        const expert_hot_entry & candidate,
                                                        const expert_hot_entry & incumbent) {
                        const uint64_t candidate_age = saturating_add(
                            expert_hot_touch_clock_ - candidate.last_touch, uint64_t{1});
                        const uint64_t incumbent_age = saturating_add(
                            expert_hot_touch_clock_ - incumbent.last_touch, uint64_t{1});
                        const wide_uint candidate_score =
                            wide_mul(candidate.frequency, incumbent_age);
                        const wide_uint incumbent_score =
                            wide_mul(incumbent.frequency, candidate_age);
                        if (candidate_score != incumbent_score) return candidate_score < incumbent_score;
                        if (candidate.last_touch != incumbent.last_touch) {
                            return candidate.last_touch < incumbent.last_touch;
                        }
                        if (candidate.layer != incumbent.layer) return candidate.layer < incumbent.layer;
                        return candidate.expert_id < incumbent.expert_id;
                    };
                    const bool replace_victim = victim == expert_hot_entries_.end() ||
                        (expert_hot_lfru_enabled_
                             ? lower_lfru_score(*entry, *victim)
                             : entry->last_touch < victim->last_touch);
                    if (replace_victim) {
                        victim = entry;
                    }
                }
                if (victim == expert_hot_entries_.end()) break;
                for (const page_range & range : victim->ranges) {
                    (void) munlock(reinterpret_cast<void *>(range.address), range.length);
                }
                expert_hot_locked_bytes_ = hot_cache_release(expert_hot_locked_bytes_, victim->locked_bytes);
                expert_hot_evictions_ = saturating_add(expert_hot_evictions_, uint64_t{1});
                expert_hot_entries_.erase(victim);
            }
        }
        if (candidate_bytes > hot_cache_headroom(
                static_cast<uint64_t>(expert_hot_budget_bytes_), expert_hot_locked_bytes_)) {
            expert_hot_budget_rejections_ = saturating_add(expert_hot_budget_rejections_, uint64_t{1});
            continue;
        }

        std::vector<page_range> locked_ranges;
        bool lock_failed = false;
        for (const page_range & range : resident_ranges) {
            if (mlock(reinterpret_cast<void *>(range.address), range.length) != 0) {
                expert_hot_lock_failures_ = saturating_add(expert_hot_lock_failures_, uint64_t{1});
                lock_failed = true;
                break;
            }
            locked_ranges.push_back(range);
        }
        if (lock_failed) {
            for (const page_range & range : locked_ranges) {
                (void) munlock(reinterpret_cast<void *>(range.address), range.length);
            }
            continue;
        }
        if (expert_hot_touch_clock_ != std::numeric_limits<uint64_t>::max()) ++expert_hot_touch_clock_;
        expert_hot_entries_.push_back(
            {layer, expert_id, std::move(locked_ranges), candidate_bytes, expert_hot_touch_clock_, 1});
        expert_hot_locked_bytes_ = saturating_add(expert_hot_locked_bytes_, candidate_bytes);
        expert_hot_admissions_ = saturating_add(expert_hot_admissions_, uint64_t{1});
    }
}

void prefetch_scheduler::reclaim_wrong_expert_pages(
    const std::vector<int> & prefetched,
    const std::vector<int> & selected,
    const std::vector<expert_tensor_info> & tensors) {
    if (!reclaim_waste_enabled_ || !enabled_.load()) return;

    const std::vector<int> wasted = wasted_expert_ids(prefetched, selected);
    atomic_saturating_add(reclaim_candidate_experts_, static_cast<uint64_t>(wasted.size()));
    if (wasted.empty()) return;

    const long raw_page_size = page_size_query_();
    if (raw_page_size <= 0) {
        atomic_saturating_add(reclaim_invalid_layouts_, uint64_t{1});
        return;
    }

    std::vector<expert_tensor_view> tensor_views;
    tensor_views.reserve(tensors.size());
    for (const expert_tensor_info & tensor : tensors) {
        tensor_views.push_back({reinterpret_cast<uintptr_t>(tensor.addr), tensor.size, tensor.n_experts});
    }
    const expert_reclaim_plan plan = build_expert_reclaim_plan(
        tensor_views, wasted, static_cast<size_t>(raw_page_size));
    atomic_saturating_add(reclaim_skipped_bytes_, plan.skipped_bytes);
    atomic_saturating_add(reclaim_invalid_layouts_, plan.invalid_layouts);
    atomic_saturating_add(reclaim_invalid_ids_, plan.invalid_ids);

    for (const expert_reclaim_item & item : plan.items) {
        if (!enabled_.load()) return;
        if (item.length == 0) continue;
        atomic_saturating_add(reclaim_calls_, uint64_t{1});
        if (advice_(reinterpret_cast<void *>(item.address), item.length, POSIX_MADV_DONTNEED) == 0) {
            atomic_saturating_add(reclaim_reclaimed_bytes_, static_cast<uint64_t>(item.length));
        } else {
            atomic_saturating_add(reclaim_madvise_failures_, uint64_t{1});
        }
    }
}

uint64_t prefetch_scheduler::prefetch_experts(int layer, const int * expert_ids, int n) {
    if (expert_prefetch_disabled_) return 0;
    return issue_expert_willneed(layer, expert_ids, n);
}

uint64_t prefetch_scheduler::issue_expert_willneed(int layer, const int * expert_ids, int n) {
    if (!enabled_.load()) return 0;
    if (layer < 0 || expert_ids == nullptr || n <= 0) return 0;

    std::vector<int> requested;
    requested.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        append_unique_nonnegative(requested, expert_ids[i]);
    }
    if (requested.empty()) return 0;

    std::vector<expert_tensor_info> exps;
    std::vector<int> prev;
    std::vector<uint32_t> population;
    expert_pressure_state pressure = expert_pressure_state::missing;
    uint64_t policy_budget = std::numeric_limits<uint64_t>::max();
    bool residency_snapshot_set = false;
    uint32_t waste_ewma = 0;
    bool waste_restricted = false;
    {
        std::lock_guard<std::mutex> lock(expert_state_mtx_);
        if (static_cast<size_t>(layer) >= experts_by_layer_.size()) return 0;
        exps = experts_by_layer_[layer];
        if (static_cast<size_t>(layer) < prev_router_experts_.size()) {
            prev = prev_router_experts_[layer];
        }
        if (static_cast<size_t>(layer) < expert_pop_counts_.size()) {
            population = expert_pop_counts_[layer];
        }
        pressure = expert_pressure_;
        residency_snapshot_set = expert_residency_snapshot_set_;
        if (residency_snapshot_set) policy_budget = expert_residency_budget_;
        waste_ewma = expert_waste_ewma_milli_;
        waste_restricted = expert_waste_restricted_;
    }
    if (exps.empty()) return 0;

    // SLIM-ARC FIX 2026-08-09: build the "actual prefetch target set" first.
    // Improvement 2 (confidence gating, from CommitMoE/DALI): with
    // SLIM_ARC_EXPERT_CONF=1, prefetch only the stable experts active in both of the
    // last two tokens (A_t ∩ A_{t-1}); low-confidence experts do not waste WILLNEED and
    // rely on demand paging instead, which cuts useless I/O.
    std::vector<int> target;
    target.reserve(requested.size());
    if (conf_gating_ && !prev.empty()) {
        for (int eid : requested) {
            if (std::find(prev.begin(), prev.end(), eid) != prev.end()) {
                append_unique_nonnegative(target, eid);
            }
        }
    } else {
        for (int eid : requested) {
            append_unique_nonnegative(target, eid);
        }
    }
    // SLIM-ARC FIX 2026-08-09: improvement 4 -- union in the top-K hot experts (from
    // ReMoE/DALI locality). Wider coverage and fewer misses; enabled by
    // SLIM_ARC_EXPERT_POP=K.
    if (pop_k_ > 0 && !population.empty()) {
        std::vector<std::pair<int, uint32_t>> rank;
        for (int eid = 0; eid < static_cast<int>(population.size()); ++eid) {
            if (population[eid] > 0) rank.emplace_back(eid, population[eid]);
        }
        std::sort(rank.begin(), rank.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });
        int added = 0;
        for (const auto & pr : rank) {
            if (added >= pop_k_) break;
            int eid = pr.first;
            if (std::find(target.begin(), target.end(), eid) == target.end()) {
                target.push_back(eid);
                ++added;
            }
        }
    }
    target.erase(std::remove_if(target.begin(), target.end(), [&exps](int eid) {
        return std::none_of(exps.begin(), exps.end(), [eid](const expert_tensor_info & expert) {
            return expert.addr != nullptr && expert.n_experts > 0 &&
                   expert.size % static_cast<size_t>(expert.n_experts) == 0 &&
                   eid < expert.n_experts && expert.size / static_cast<size_t>(expert.n_experts) > 0;
        });
    }), target.end());

    if (expert_residency_enabled_) {
        if (!residency_snapshot_set && expert_budget_enabled_.load()) {
            policy_budget = expert_budget_.load();
        }

        std::vector<int> candidate_ids = target;
        if (pressure != expert_pressure_state::missing) {
            candidate_ids = requested;
            std::vector<std::pair<int, uint32_t>> rank;
            for (int eid = 0; eid < static_cast<int>(population.size()); ++eid) {
                if (population[eid] > 0) rank.emplace_back(eid, population[eid]);
            }
            std::stable_sort(rank.begin(), rank.end(), [](const auto & left, const auto & right) {
                return left.second > right.second;
            });
            for (const auto & [eid, unused] : rank) {
                (void) unused;
                append_unique_nonnegative(candidate_ids, eid);
            }
        }

        size_t available_experts = 0;
        std::vector<expert_candidate> candidates;
        candidates.reserve(candidate_ids.size());
        uint64_t minimum_bytes = std::numeric_limits<uint64_t>::max();
        for (const expert_tensor_info & expert : exps) {
            if (expert.addr != nullptr && expert.n_experts > 0 &&
                expert.size % static_cast<size_t>(expert.n_experts) == 0) {
                available_experts = std::max(available_experts, static_cast<size_t>(expert.n_experts));
            }
        }
        for (const int eid : candidate_ids) {
            uint64_t bytes = 0;
            for (const expert_tensor_info & expert : exps) {
                if (expert.addr == nullptr || expert.n_experts <= 0 || eid >= expert.n_experts ||
                    expert.size % static_cast<size_t>(expert.n_experts) != 0) continue;
                bytes = saturating_add(bytes, static_cast<uint64_t>(
                    expert.size / static_cast<size_t>(expert.n_experts)));
            }
            if (bytes > 0) minimum_bytes = std::min(minimum_bytes, bytes);
            const uint32_t popularity = eid >= 0 && static_cast<size_t>(eid) < population.size()
                ? population[static_cast<size_t>(eid)] : 0;
            candidates.push_back({
                eid,
                bytes,
                popularity,
                std::find(requested.begin(), requested.end(), eid) != requested.end() &&
                    std::find(prev.begin(), prev.end(), eid) != prev.end(),
                std::find(requested.begin(), requested.end(), eid) != requested.end(),
            });
        }

        size_t max_experts = pressure == expert_pressure_state::missing ? target.size() : candidates.size();
        max_experts = std::min(max_experts, available_experts);
        max_experts = std::min(max_experts, candidates.size());
        if (minimum_bytes == std::numeric_limits<uint64_t>::max() || minimum_bytes == 0) {
            max_experts = 0;
        } else {
            const uint64_t budget_count = policy_budget / minimum_bytes;
            max_experts = std::min<uint64_t>(max_experts, budget_count);
        }

        const expert_residency_decision decision = select_resident_experts({
            pressure,
            policy_budget,
            max_experts,
            waste_ewma,
            std::move(candidates),
            waste_restricted,
        });
        atomic_saturating_add(residency_samples_, uint64_t{1});
        atomic_saturating_add(residency_admitted_experts_, static_cast<uint64_t>(decision.expert_ids.size()));
        atomic_saturating_add(residency_admitted_bytes_, decision.admitted_bytes);
        atomic_saturating_add(residency_skipped_bytes_, decision.skipped_bytes);
        if (decision.fallback) atomic_saturating_add(residency_fallbacks_, uint64_t{1});
        switch (pressure) {
            case expert_pressure_state::missing:
                atomic_saturating_add(residency_pressure_missing_, uint64_t{1});
                break;
            case expert_pressure_state::normal:
                atomic_saturating_add(residency_pressure_normal_, uint64_t{1});
                break;
            case expert_pressure_state::high:
                atomic_saturating_add(residency_pressure_high_, uint64_t{1});
                break;
            case expert_pressure_state::critical:
                atomic_saturating_add(residency_pressure_critical_, uint64_t{1});
                break;
        }
        target = std::move(decision.expert_ids);
    }
    if (target.empty()) return 0;

    const long raw_page_size = page_size_query_();
    const size_t page_size = raw_page_size > 0 ? static_cast<size_t>(raw_page_size) : 0;
    uint64_t advice_requests = 0;
    uint64_t invalid_ranges = 0;
    std::vector<expert_page_candidate> page_candidates;
    page_candidates.reserve(target.size() * exps.size());
    for (const int eid : target) {
        for (const expert_tensor_info & expert : exps) {
            if (expert.addr == nullptr || expert.n_experts <= 0 || eid >= expert.n_experts ||
                expert.size % static_cast<size_t>(expert.n_experts) != 0) continue;
            const size_t per_expert = expert.size / static_cast<size_t>(expert.n_experts);
            if (per_expert == 0) continue;
            advice_requests = saturating_add(advice_requests, uint64_t{1});
            if (static_cast<size_t>(eid) > std::numeric_limits<size_t>::max() / per_expert) {
                invalid_ranges = saturating_add(invalid_ranges, uint64_t{1});
                continue;
            }
            const size_t offset = static_cast<size_t>(eid) * per_expert;
            const uintptr_t base = reinterpret_cast<uintptr_t>(expert.addr);
            if (offset > std::numeric_limits<uintptr_t>::max() - base) {
                invalid_ranges = saturating_add(invalid_ranges, uint64_t{1});
                continue;
            }
            const page_range range = covering_page_range(base + offset, per_expert, page_size);
            if (!range.valid) {
                invalid_ranges = saturating_add(invalid_ranges, uint64_t{1});
                continue;
            }
            page_candidates.push_back({eid, range});
        }
    }

    std::vector<page_range> candidate_ranges;
    candidate_ranges.reserve(page_candidates.size());
    for (const expert_page_candidate & candidate : page_candidates) {
        candidate_ranges.push_back(candidate.range);
    }
    page_range_set coalesced = coalesce_page_ranges(std::move(candidate_ranges));
    if (!coalesced.valid) {
        invalid_ranges = saturating_add(
            invalid_ranges,
            static_cast<uint64_t>(coalesced.input_ranges));
        coalesced.ranges.clear();
        page_candidates.clear();
    }
    uint64_t covered_bytes = 0;
    for (const page_range & range : coalesced.ranges) {
        covered_bytes = saturating_add(covered_bytes, static_cast<uint64_t>(range.length));
    }
    atomic_saturating_add(expert_advice_requests_, advice_requests);
    atomic_saturating_add(
        expert_coalesced_ranges_,
        static_cast<uint64_t>(coalesced.ranges.size()));
    atomic_saturating_add(expert_covered_bytes_, covered_bytes);
    atomic_saturating_add(expert_invalid_ranges_, invalid_ranges);
    atomic_saturating_add(expert_advice_failures_, invalid_ranges);
    if (coalesced.ranges.empty()) return 0;

    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(expert_state_mtx_);
        if (next_expert_generation_ == 0 ||
            next_expert_generation_ == std::numeric_limits<uint64_t>::max()) {
            return 0;
        }
        if (static_cast<size_t>(layer) >= pending_expert_prefetches_.size()) {
            pending_expert_prefetches_.resize(static_cast<size_t>(layer) + 1);
        }
        std::vector<expert_prefetch_record> & pending = pending_expert_prefetches_[layer];
        if (pending.size() >= max_pending_expert_records_) {
            atomic_saturating_add(expert_pending_rejected_generations_, uint64_t{1});
            return 0;
        }
        if (expert_budget_enabled_.load()) {
            const size_t budget = expert_budget_.load();
            if (budget == 0 || covered_bytes > std::numeric_limits<size_t>::max()) return 0;
            const size_t reservation = static_cast<size_t>(covered_bytes);
            size_t used = expert_budget_used_.load();
            while (true) {
                if (used > budget || reservation > budget - used) return 0;
                if (expert_budget_used_.compare_exchange_weak(used, used + reservation)) break;
            }
        }
        generation = next_expert_generation_++;
        pending.push_back({generation, {}, {}});
    }

    const std::vector<expert_page_owner> owners = assign_page_owners(page_candidates);
    std::vector<bool> successful_ranges(coalesced.ranges.size(), false);
    size_t bytes = 0;
    uint64_t callback_failures = 0;
    for (size_t index = 0; index < coalesced.ranges.size(); ++index) {
        const page_range & range = coalesced.ranges[index];
        if (advice_(reinterpret_cast<void *>(range.address), range.length, POSIX_MADV_WILLNEED) == 0) {
            successful_ranges[index] = true;
            bytes = saturating_add(bytes, range.length);
        } else {
            callback_failures = saturating_add(callback_failures, uint64_t{1});
        }
    }
    atomic_saturating_add(expert_advice_failures_, callback_failures);

    std::map<int, size_t> bytes_by_expert;
    size_t range_index = 0;
    for (const expert_page_owner & owner : owners) {
        while (range_index < coalesced.ranges.size() &&
               coalesced.ranges[range_index].address + coalesced.ranges[range_index].length <= owner.address) {
            ++range_index;
        }
        if (range_index >= coalesced.ranges.size() || !successful_ranges[range_index]) continue;
        const page_range & range = coalesced.ranges[range_index];
        if (owner.address < range.address || owner.length >
            range.address + range.length - owner.address) continue;
        bytes_by_expert[owner.expert_id] = saturating_add(
            bytes_by_expert[owner.expert_id],
            owner.length);
    }

    std::vector<int> issued_experts;
    std::vector<size_t> issued_expert_bytes;
    issued_experts.reserve(target.size());
    issued_expert_bytes.reserve(target.size());
    for (const int eid : target) {
        const auto found = bytes_by_expert.find(eid);
        if (found == bytes_by_expert.end() || found->second == 0) continue;
        issued_experts.push_back(eid);
        issued_expert_bytes.push_back(found->second);
    }
    {
        std::lock_guard<std::mutex> lock(expert_state_mtx_);
        std::vector<expert_prefetch_record> & pending = pending_expert_prefetches_[layer];
        const auto it = std::find_if(pending.begin(), pending.end(), [generation](const auto & record) {
            return record.generation == generation;
        });
        if (it == pending.end()) {
            atomic_saturating_add(expert_unmatched_generations_, uint64_t{1});
            return 0;
        }
        if (issued_experts.empty()) {
            pending.erase(it);
            return 0;
        }
        it->expert_ids = std::move(issued_experts);
        it->issued_bytes = std::move(issued_expert_bytes);
    }

    if (bytes > 0) {
        atomic_saturating_add(total_bytes_, bytes);
        atomic_saturating_add(expert_prefetch_bytes_, bytes);
    }
    return generation;
}

// SLIM-ARC FIX 2026-08-09: summary of the expert-prefetch observability metrics
// (improvement 1). issued = bytes actually handed to WILLNEED (issue side); hit/waste =
// hit/wasted bytes (cache side, checked against the prefetch set with the actual
// routing). Hit rate = hit/(hit+waste). The two accountings may differ slightly because
// the prefetch and cache passes do not cover exactly the same layers; the hit rate is
// not affected.
void prefetch_scheduler::dump_metrics() const {
    // [QLH] The upstream dump only reports the expert-side counters, which stay at zero unless
    // the expert prefetcher is switched on. Without this line a plain layer-ahead run is
    // indistinguishable from a scheduler that never fired at all.
    std::fprintf(stderr,
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
    size_t issued = expert_prefetch_bytes_.load();
    size_t hit    = expert_hit_bytes_.load();
    size_t waste  = expert_waste_bytes_.load();
    size_t total  = saturating_add(hit, waste);
    double hr = total > 0 ? 100.0 * (double) hit / (double) total : 0.0;
    fprintf(stderr,
            "[SLIM-ARC-METRICS] expert prefetch: samples=%llu issued=%.1fMB "
            "hit=%.1fMB waste=%.1fMB hit_rate=%.2f%% (accounted %.1fMB)\n",
            static_cast<unsigned long long>(router_samples_.load()),
            issued / 1048576.0, hit / 1048576.0, waste / 1048576.0, hr,
            total / 1048576.0);
    if (cross_layer_transition_enabled_) {
        const expert_transition_stats transition = expert_transition_statistics();
        std::fprintf(
            stderr,
            "[SLIM-ARC-TRANSITION] schema=1 updates=%llu prediction_rounds=%llu "
            "empty_rounds=%llu predicted_experts=%llu matched_experts=%llu decays=%llu\n",
            static_cast<unsigned long long>(transition.updates),
            static_cast<unsigned long long>(transition.prediction_rounds),
            static_cast<unsigned long long>(transition.empty_rounds),
            static_cast<unsigned long long>(transition.predicted_experts),
            static_cast<unsigned long long>(transition.matched_experts),
            static_cast<unsigned long long>(transition.decays));
    }
    if (router_mlock_enabled_) {
        std::fprintf(
            stderr,
            "[SLIM-ARC-ROUTER] locked_bytes=%llu lock_failures=%llu\n",
            static_cast<unsigned long long>(router_locked_bytes_.load()),
            static_cast<unsigned long long>(router_lock_failures_.load()));
    }
    if (shared_mlock_enabled_) {
        std::fprintf(
            stderr,
            "[SLIM-ARC-SHARED] locked_bytes=%llu lock_failures=%llu\n",
            static_cast<unsigned long long>(shared_locked_bytes_.load()),
            static_cast<unsigned long long>(shared_lock_failures_.load()));
    }
    if (small_mlock_enabled_) {
        std::fprintf(
            stderr,
            "[SLIM-ARC-SMALL] locked_bytes=%llu lock_failures=%llu\n",
            static_cast<unsigned long long>(small_locked_bytes_.load()),
            static_cast<unsigned long long>(small_lock_failures_.load()));
    }
    if (expert_hot_budget_bytes_ > 0) {
        const expert_hot_cache_stats stats = expert_hot_cache_statistics();
        std::fprintf(
            stderr,
            "[SLIM-ARC-HOT] budget_bytes=%llu locked_bytes=%llu admissions=%llu hits=%llu "
            "evictions=%llu budget_rejections=%llu nonresident_bytes=%llu lock_failures=%llu "
            "entries=%llu admission_skips=%llu admission_threshold=%llu lru=%d lfru=%d\n",
            static_cast<unsigned long long>(stats.budget_bytes),
            static_cast<unsigned long long>(stats.locked_bytes),
            static_cast<unsigned long long>(stats.admissions),
            static_cast<unsigned long long>(stats.hits),
            static_cast<unsigned long long>(stats.evictions),
            static_cast<unsigned long long>(stats.budget_rejections),
            static_cast<unsigned long long>(stats.nonresident_bytes),
            static_cast<unsigned long long>(stats.lock_failures),
            static_cast<unsigned long long>(stats.entries),
            static_cast<unsigned long long>(stats.admission_skips),
            static_cast<unsigned long long>(stats.admission_threshold),
            expert_hot_lru_enabled_ ? 1 : 0,
            expert_hot_lfru_enabled_ ? 1 : 0);
    }
    if (expert_random_madv_enabled_ || expert_normal_madv_enabled_) {
        std::fprintf(
            stderr,
            "[SLIM-ARC-EXPERT-MADV] mode=%s calls=%llu bytes=%llu failures=%llu\n",
            expert_random_madv_enabled_ ? "RANDOM" : "NORMAL",
            static_cast<unsigned long long>(expert_madv_advice_calls_.load()),
            static_cast<unsigned long long>(expert_madv_advice_bytes_.load()),
            static_cast<unsigned long long>(expert_madv_advice_failures_.load()));
    }
}

// [QLH port] See slim-arc-prefetch.h. The loader installs the instance while init_mappings is
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
