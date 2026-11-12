#include "llama.h"

#if defined(LLAMA_KVMEM)
#include "llama-kvmem-hooks.h"
#endif

// [QLH-KVMEM] Process-global adapter controls, reachable from common/ without pulling the
// adapter's own headers (and their include directory) into the common library.
//
// The adapter snapshots its parameters when a llama_context builds its memory, so this has to
// run before llama_init_from_model(). Everything not listed here keeps the adapter default.
bool llama_kvmem_apply_params(int32_t budget, int32_t gen_reserve) {
#if defined(LLAMA_KVMEM)
    struct llama_kvmem_params kvmem = {};

    kvmem.enabled     = true;
    kvmem.budget      = budget      > 0 ? (uint32_t) budget      : 0;
    kvmem.gen_reserve = gen_reserve > 0 ? (uint32_t) gen_reserve : 0;
    // 1 = retrieval: block scores come from the query rows instead of pure recency.
    kvmem.method      = 1;

    llama_kvmem_set_params(&kvmem);
    return true;
#else
    (void) budget;
    (void) gen_reserve;
    return false;
#endif
}
