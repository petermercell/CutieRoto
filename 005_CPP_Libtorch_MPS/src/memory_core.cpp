// ============================================================================
// memory_core.cpp  —  KVStore + fp32 memory read implementation
// Transcribed from cutie/model/utils/memory_utils.py (validated bit-exact in
// memory_read_reference.py).
// ============================================================================
#include "memory_core.h"
#include <cmath>
#include <algorithm>

namespace cutie {

using torch::indexing::Slice;
using torch::indexing::None;

// ---------------------------------------------------------------------------
// KVStore
// ---------------------------------------------------------------------------
void KVStore::add(const torch::Tensor& key, const torch::Tensor& shr,
                  const torch::Tensor& val, bool permanent)
{
    auto k_ = key.flatten(2);   // (1,CK,Ne)
    auto s_ = shr.flatten(2);   // (1,1,Ne)
    auto v_ = val.flatten(2);   // (1,CV,Ne)
    const int64_t ne = k_.size(-1);

    if (!k.defined()) {
        k = k_.clone(); s = s_.clone(); v = v_.clone();
    } else {
        k = torch::cat({k, k_}, -1);
        s = torch::cat({s, s_}, -1);
        v = torch::cat({v, v_}, -1);
    }
    if (permanent) perm_end_pt += ne;
}

void KVStore::removeOldMemory()
{
    const int64_t N = k.size(-1);
    const int64_t temp = N - perm_end_pt;
    if (temp <= max_work_tokens) return;
    const int64_t keep_from = N - max_work_tokens;   // newest max_work_tokens temp
    const int64_t p = perm_end_pt;
    k = torch::cat({k.index({Slice(), Slice(), Slice(None, p)}),
                    k.index({Slice(), Slice(), Slice(keep_from, None)})}, -1);
    s = torch::cat({s.index({Slice(), Slice(), Slice(None, p)}),
                    s.index({Slice(), Slice(), Slice(keep_from, None)})}, -1);
    v = torch::cat({v.index({Slice(), Slice(), Slice(None, p)}),
                    v.index({Slice(), Slice(), Slice(keep_from, None)})}, -1);
    // perm_end_pt unchanged
}

// ---------------------------------------------------------------------------
// read primitives (fp32) — exact transcription of memory_utils.py
// ---------------------------------------------------------------------------
torch::Tensor getSimilarity(const torch::Tensor& mk_in, const torch::Tensor& ms_in,
                            const torch::Tensor& qk_in, const torch::Tensor& qe_in)
{
    // mk (1,CK,N)  ms (1,1,N)  qk (1,CK,HW)  qe (1,CK,HW)  -> sim (1,N,HW)
    auto mk = mk_in.to(torch::kFloat32).flatten(2);
    auto qk = qk_in.to(torch::kFloat32).flatten(2);
    auto qe = qe_in.to(torch::kFloat32).flatten(2);
    auto ms = ms_in.to(torch::kFloat32).flatten(1).unsqueeze(2);   // (1,N,1)
    const int64_t CK = mk.size(1);

    auto mk_t   = mk.transpose(1, 2);                    // (1,N,CK)
    auto a_sq   = torch::matmul(mk_t.pow(2), qe);         // (1,N,HW)
    auto two_ab = 2.0 * torch::matmul(mk_t, qk * qe);     // (1,N,HW)
    auto b_sq   = (qe * qk.pow(2)).sum(1, /*keepdim=*/true);   // (1,1,HW)
    auto sim = (-a_sq + two_ab - b_sq) * ms / std::sqrt((double)CK);
    return sim;
}

torch::Tensor doSoftmax(const torch::Tensor& sim, int top_k)
{
    // top-k over memory tokens (dim=1), exp-normalize, scatter into zeros.
    auto topk = torch::topk(sim, top_k, /*dim=*/1);
    auto values  = std::get<0>(topk);
    auto indices = std::get<1>(topk);
    // Stable softmax: subtract per-column max before exp so large sim values can't
    // overflow exp()->Inf (Inf/Inf=NaN). Mathematically identical to plain
    // exp/sum. Cutie's non-topk branch does this; the topk branch relies on fp16
    // bounding, but our fp32 path can overflow with some token distributions
    // (e.g. 2 keyframes) -> we subtract the max here to be safe.
    auto maxes = std::get<0>(torch::max(values, /*dim=*/1, /*keepdim=*/true));
    auto x_exp = (values - maxes).exp();
    x_exp = x_exp / x_exp.sum(1, /*keepdim=*/true);
    auto affinity = torch::zeros_like(sim).scatter_(1, indices, x_exp);
    return affinity;
}

torch::Tensor readout(const torch::Tensor& affinity, const torch::Tensor& mv_in)
{
    // affinity (1,N,HW)  mv (1,CV,N) -> mem (1,CV,HW)
    auto mv = mv_in.to(torch::kFloat32);
    return torch::bmm(mv, affinity);
}

// ---------------------------------------------------------------------------
// full read (single object, no long-term)
// ---------------------------------------------------------------------------
torch::Tensor memoryRead(const KVStore& store,
                         const torch::Tensor& query_key,
                         const torch::Tensor& selection,
                         int top_k)
{
    const int64_t h  = query_key.size(-2);
    const int64_t w  = query_key.size(-1);
    const int64_t HW = h * w;
    const int64_t CV = store.v.size(1);
    const int64_t N  = store.k.defined() ? store.k.size(-1) : 0;

    // Peak-VRAM bound. getSimilarity/doSoftmax build (1,N,HW) intermediates (sim
    // plus the dense scattered affinity), so peak grows with the store size N and
    // eventually exceeds the card -> the propagation stops mid-clip. The read is
    // independent per query column (each HW position's top-k softmax over memory
    // tokens and its readout bmm don't touch other columns), so we can slice the
    // HW axis and cap peak at ~(1,N,chunk) no matter how large N gets. This is
    // EXACT — identical math, just computed in slices. chunk >= HW falls through
    // to the original single-pass path (so small reads, incl. the unit test when
    // N is small, are byte-for-byte the validated code).
    const int64_t kBudget = 32LL * 1024 * 1024;            // ~max N*chunk elements/pass
    const int64_t chunk   = (N > 0) ? std::max<int64_t>(1, kBudget / N) : HW;

    if (chunk >= HW) {
        auto sim = getSimilarity(store.k, store.s, query_key, selection);
        auto aff = doSoftmax(sim, top_k);
        sim.reset();   // free the (1,N,HW) sim before readout: doSoftmax already
                       // used it, so it and the dense affinity needn't both live.
        auto mem = readout(aff, store.v);          // (1,CV,HW)
        return mem.view({1, CV, h, w});
    }

    // chunked over HW (peak ~ 2*(1,N,chunk))
    auto qk = query_key.to(torch::kFloat32).flatten(2);    // (1,CK,HW)
    auto qe = selection.to(torch::kFloat32).flatten(2);    // (1,CK,HW)
    auto out = torch::empty({1, CV, HW},
                 torch::TensorOptions().dtype(torch::kFloat32).device(store.v.device()));
    for (int64_t c = 0; c < HW; c += chunk) {
        const int64_t e = std::min(HW, c + chunk);
        auto qk_c  = qk.index({Slice(), Slice(), Slice(c, e)});      // (1,CK,chunk)
        auto qe_c  = qe.index({Slice(), Slice(), Slice(c, e)});
        auto sim_c = getSimilarity(store.k, store.s, qk_c, qe_c);    // (1,N,chunk)
        auto aff_c = doSoftmax(sim_c, top_k);
        sim_c.reset();
        auto mem_c = readout(aff_c, store.v);                        // (1,CV,chunk)
        out.index_put_({Slice(), Slice(), Slice(c, e)}, mem_c);
    }
    return out.view({1, CV, h, w});
}

} // namespace cutie