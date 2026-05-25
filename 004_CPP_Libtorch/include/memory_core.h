// ============================================================================
// memory_core.h  —  Cutie-Nuke, the C++ memory core (replaces MemoryManager).
//
// Two pieces, both proven in Python (memory_read_reference.py, bit-exact):
//   KVStore     : single bucket, three torch::Tensor buffers k/s/v laid out
//                 [permanent | temporary] along the token axis, with perm_end_pt.
//                 add(permanent?) appends; removeOldMemory() FIFO-trims temporary
//                 tokens past max_work_tokens. use_long_term=False -> no prototypes,
//                 no usage scoring, no compression. ~the whole store.
//   memoryRead  : fp32 transcription of get_similarity (selection/qe branch) ->
//                 do_softmax(top_k=30, scatter) -> readout (bmm). torch:: ops,
//                 no custom CUDA. Run on the shared stream with autocast OFF.
//
// Config (locked): CK=64, CV=256, top_k=30, max_mem_frames=5, HW=68*120=8160,
//   max_work_tokens = 5*8160 = 40800.
// ============================================================================
#pragma once

#include <torch/torch.h>

namespace cutie {

struct KVStore {
    // buffers: k (1,CK,N), s (1,1,N), v (1,CV,N). N = perm + temp tokens.
    torch::Tensor k, s, v;
    int64_t perm_end_pt = 0;
    int64_t max_work_tokens;     // = max_mem_frames * HW (40800)

    explicit KVStore(int64_t max_work_tokens_) : max_work_tokens(max_work_tokens_) {}

    bool empty() const { return !k.defined(); }

    // Append a memory frame. key (1,CK,N_e) shr (1,1,N_e) val (1,CV,N_e), already
    // flattened to token axis. permanent=true -> seed roto (force_permanent),
    // bump perm_end_pt; false -> every-mem_every temporary frame.
    void add(const torch::Tensor& key, const torch::Tensor& shr,
             const torch::Tensor& val, bool permanent);

    // FIFO: if temp token count exceeds max_work_tokens, keep the LAST
    // max_work_tokens temporary tokens; permanent region untouched.
    void removeOldMemory();
};

// fp32 memory read. Inputs: store buffers + query key/selection (1,CK,h,w).
// Returns visual_readout (1,CV,h,w). Mirrors MemoryManager.read (single obj, no LT).
// Run inside a no-autocast scope on the shared stream.
torch::Tensor memoryRead(const KVStore& store,
                         const torch::Tensor& query_key,
                         const torch::Tensor& selection,
                         int top_k = 30);

// the three primitives, exposed for unit testing against the Python reference
torch::Tensor getSimilarity(const torch::Tensor& mk, const torch::Tensor& ms,
                            const torch::Tensor& qk, const torch::Tensor& qe);
torch::Tensor doSoftmax(const torch::Tensor& sim, int top_k);
torch::Tensor readout(const torch::Tensor& affinity, const torch::Tensor& mv);

} // namespace cutie
