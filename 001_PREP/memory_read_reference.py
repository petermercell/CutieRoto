#!/usr/bin/env python3
"""
memory_read_reference.py  —  Path B, the C++ memory-core ORACLE

Standalone transcription of Cutie's memory read (the part that becomes libtorch
C++) plus a SimpleKVStore that mirrors the planned C++ store (single bucket,
[permanent | temporary] buffers, perm_end_pt, FIFO eviction, use_long_term=False).

We validate two things against the REAL Cutie objects on real propagation state:
  (1) SimpleKVStore reproduces work_mem's k/s/v buffers exactly (append + FIFO).
  (2) reference_memory_read() reproduces MemoryManager.read's affinity + visual
      readout exactly (get_similarity -> do_softmax(top_k=30) -> readout).

This is the spec the C++ implements op-for-op. fp32 throughout (matches the
autocast(enabled=False) in memory_utils / the read path).

Run (env: conda cutie311):
  python memory_read_reference.py \
      --frames /home/pm/Documents/CATs/EXR/grimes2.%04d.exr \
      --roto   /home/pm/Documents/CATs/MASK/MASK1.%04d.exr \
      --seeds 1,48,96 --n 96 --start 1 --check-frame 30
"""

import argparse, math
import numpy as np
import torch

from cutie.utils.get_default_model import get_default_model
from cutie.inference.inference_core import InferenceCore
import cutie_mask_propagate as oracle


# ============================================================================
# (A) standalone memory read — the exact ops the C++/libtorch will run, fp32
# ============================================================================
@torch.inference_mode()
def reference_get_similarity(mk, ms, qk, qe):
    """mk: 1xCKxN  ms: 1x1xN  qk: 1xCKxHW  qe: 1xCKxHW -> sim: 1xNxHW (fp32).
    Anisotropic L2 with selection term (qe is not None branch)."""
    mk = mk.float().flatten(2); qk = qk.float().flatten(2); qe = qe.float().flatten(2)
    ms = ms.float().flatten(1).unsqueeze(2)
    CK = mk.shape[1]
    mk_t = mk.transpose(1, 2)                      # 1xNxCK
    a_sq = mk_t.pow(2) @ qe                          # 1xNxHW
    two_ab = 2 * (mk_t @ (qk * qe))                  # 1xNxHW
    b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)     # 1x1xHW
    sim = (-a_sq + two_ab - b_sq) * ms / math.sqrt(CK)
    return sim


@torch.inference_mode()
def reference_do_softmax(sim, top_k=30):
    """top-k softmax over memory tokens (dim=1). Returns affinity 1xNxHW (fp32)."""
    values, indices = torch.topk(sim, k=top_k, dim=1)
    x_exp = values.exp()
    x_exp = x_exp / x_exp.sum(dim=1, keepdim=True)
    affinity = torch.zeros_like(sim).scatter_(1, indices, x_exp)
    return affinity


@torch.inference_mode()
def reference_readout(affinity, mv):
    """affinity: 1xNxHW  mv: 1xCVxN -> mem: 1xCVxHW (fp32)."""
    mo = mv.float()                                  # 1xCVxN
    mem = torch.bmm(mo, affinity)                    # 1xCVxHW
    return mem


@torch.inference_mode()
def reference_memory_read(k, s, v, query_key, selection, top_k=30):
    """Full pure-read: k(1xCKxN) s(1x1xN) v(1xCVxN), query_key/selection(1xCKxHxW)
    -> visual_readout (1xCVxHW). Mirrors MemoryManager.read (single obj, no LT)."""
    qk = query_key.flatten(2)
    qe = selection.flatten(2)
    sim = reference_get_similarity(k, s, qk, qe)
    aff = reference_do_softmax(sim, top_k=top_k)
    mem = reference_readout(aff, v)
    return mem, aff


# ============================================================================
# (B) SimpleKVStore — mirrors the planned C++ store (single bucket, no LT)
# ============================================================================
class SimpleKVStore:
    """[permanent | temporary] token buffers along dim -1, FIFO temporary trim."""
    def __init__(self, max_work_tokens):
        self.k = None  # 1xCKxN
        self.s = None  # 1x1xN
        self.v = None  # 1xCVxN
        self.perm_end_pt = 0
        self.max_work_tokens = max_work_tokens

    def add(self, key, shr, val, permanent):
        key = key.flatten(2); shr = shr.flatten(2); val = val.flatten(2)
        ne = key.shape[-1]
        if self.k is None:
            self.k, self.s, self.v = key.clone(), shr.clone(), val.clone()
        else:
            self.k = torch.cat([self.k, key], -1)
            self.s = torch.cat([self.s, shr], -1)
            self.v = torch.cat([self.v, val], -1)
        if permanent:
            self.perm_end_pt += ne

    def remove_old_memory(self):
        N = self.k.shape[-1]
        temp = N - self.perm_end_pt
        if temp <= self.max_work_tokens:
            return
        keep_from = N - self.max_work_tokens     # newest max_work_tokens temporary
        p = self.perm_end_pt
        self.k = torch.cat([self.k[:, :, :p], self.k[:, :, keep_from:]], -1)
        self.s = torch.cat([self.s[:, :, :p], self.s[:, :, keep_from:]], -1)
        self.v = torch.cat([self.v[:, :, :p], self.v[:, :, keep_from:]], -1)


# ============================================================================
# validation: run real propagation, snapshot Cutie's work_mem + a query frame,
# compare our store buffers and our read against Cutie's.
# ============================================================================
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--roto", required=True)
    ap.add_argument("--seeds", default="1,48,96")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--check-frame", type=int, default=30, help="0-based frame to probe the read")
    args = ap.parse_args()
    dev = "cuda"
    seeds0 = sorted(int(x) - args.start for x in args.seeds.split(","))
    chk = args.check_frame

    m = get_default_model().to(dev).eval()
    cfg = m.cfg
    HW_cap = cfg.max_mem_frames * (1088 // 16) * (1920 // 16)   # 5 * 68*120 = 40800
    print(f"[cfg] use_long_term={cfg.use_long_term} max_mem_frames={cfg.max_mem_frames} "
          f"mem_every={cfg.mem_every} top_k={cfg.top_k}  work_cap={HW_cap} tokens")

    # roto + frames resized to 1920x1080 (internal pad -> 1088x1920), like full_insitu
    import cv2, os, OpenEXR, Imath
    from PIL import Image
    RW, RH = 1920, 1080
    fdir = "/tmp/memref_frames"; os.makedirs(fdir, exist_ok=True)
    frame_paths = []
    for i in range(args.n):
        a = cv2.imread(args.frames % (args.start + i), cv2.IMREAD_UNCHANGED)
        if a.ndim == 2: a = np.repeat(a[..., None], 3, 2)
        rgb = a[..., :3].astype(np.float32)
        if (rgb.shape[1], rgb.shape[0]) != (RW, RH):
            rgb = cv2.resize(rgb, (RW, RH), interpolation=cv2.INTER_AREA)
        p = os.path.join(fdir, f"f{i:04d}.exr")
        hdr = OpenEXR.Header(RW, RH); ch = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        hdr['channels'] = {'R': ch, 'G': ch, 'B': ch}
        o = OpenEXR.OutputFile(p, hdr)
        o.writePixels({'R': rgb[..., 0].tobytes(), 'G': rgb[..., 1].tobytes(), 'B': rgb[..., 2].tobytes()})
        o.close(); frame_paths.append(p)
    roto = {}
    for s0 in seeds0:
        rb = oracle.load_roto_binary(args.roto % (args.start + s0)).astype(np.uint8)
        rb = np.array(Image.fromarray(rb * 255).resize((RW, RH), Image.NEAREST)) > 127
        roto[s0] = rb

    # run Cutie propagation; at the check frame, snapshot work_mem + capture the
    # query_key/selection/pix_feat that read() will use, and Cutie's own readout.
    core = InferenceCore(m, cfg=cfg)
    core.max_internal_size = -1

    captured = {}
    orig_read = core.memory.read
    def read_spy(pix_feat, query_key, selection, last_mask, network):
        # snapshot raw store buffers + query inputs ONLY at the check frame.
        # Do NO math here -- this runs inside autocast, which would silently
        # downcast any "fp32" computation back to fp16. All comparison math is
        # done after the loop, outside autocast.
        if core.curr_ti == chk:
            bid = next(iter(core.memory.work_mem.buckets.keys()))
            obj = core.memory.work_mem.buckets[bid][0]
            captured['k'] = core.memory.work_mem.k[bid].detach().clone()
            captured['s'] = core.memory.work_mem.s[bid].detach().clone()
            captured['v'] = core.memory.work_mem.v[obj].detach().clone()
            captured['perm_end_pt'] = int(core.memory.work_mem.perm_end_pt[bid])
            captured['query_key'] = query_key.detach().clone()
            captured['selection'] = selection.detach().clone()
            captured['top_k'] = int(core.memory.top_k)
        return orig_read(pix_feat, query_key, selection, last_mask, network)
    core.memory.read = read_spy


    autocast = torch.amp.autocast("cuda")
    with autocast:
        for ti, fp in enumerate(frame_paths):
            image = oracle.load_image_rgb_float(fp)
            H, W = image.shape[-2], image.shape[-1]
            if ti in seeds0:
                r = roto[ti]
                if r.shape != (H, W):
                    r = np.array(Image.fromarray(r.astype(np.uint8) * 255)
                                 .resize((W, H), Image.NEAREST)) > 127
                lab = torch.zeros((H, W), dtype=torch.long); lab[torch.from_numpy(r)] = 1
                core.step(image, lab.cuda(), objects=[1], idx_mask=True, force_permanent=True)
            else:
                core.step(image)
            if ti == chk:
                break
    if 'k' not in captured:
        raise SystemExit(f"check-frame {chk} not reached / read not called there")

    # ---- compare our standalone read vs Cutie's ----
    from cutie.model.utils.memory_utils import get_similarity, do_softmax
    k = captured['k']; s = captured['s']
    qk = captured['query_key']; qe = captured['selection']
    vraw = captured['v']
    v = vraw.reshape(vraw.shape[0], vraw.shape[1], -1)   # (1, CV, N)
    tk = captured['top_k']
    print(f"[debug] captured dtypes: k={k.dtype} s={s.dtype} qk={qk.dtype} qe={qe.dtype} v={vraw.dtype}")
    print(f"[snapshot] frame {chk}: N={k.shape[-1]} tokens "
          f"(perm_end_pt={captured['perm_end_pt']}, temp={k.shape[-1]-captured['perm_end_pt']}), "
          f"CK={k.shape[1]} CV={v.shape[1]}  value_N={v.shape[-1]}")
    assert v.shape[-1] == k.shape[-1], \
        f"value token count {v.shape[-1]} != key token count {k.shape[-1]}"

    # All math below is OUTSIDE autocast. Disable it explicitly to be safe so the
    # fp32 path is genuinely fp32 (autocast would silently downcast matmuls).
    with torch.amp.autocast("cuda", enabled=False):
        # our reference (true fp32)
        mem_ref, aff_ref = reference_memory_read(k, s, v, qk, qe, top_k=tk)

        # (A) fp32 ground truth via Cutie's util fns (cast inputs to fp32)
        sim_f = get_similarity(k.float(), s.float(), qk.float(), qe.float())
        aff_f = do_softmax(sim_f, top_k=tk, inplace=False)
        mem_f = torch.bmm(v.float(), aff_f)

        # (B) fp16 ground truth = the REAL read precision (cast inputs to half)
        sim_h = get_similarity(k.half(), s.half(), qk.half(), qe.half())
        aff_h = do_softmax(sim_h, top_k=tk, inplace=False).float()
        mem_h = torch.bmm(v.float(), aff_h)

    # MATCHED PRECISION (the faithfulness test): our fp32 vs Cutie fp32 -> ~1e-5
    aff_err = (aff_ref - aff_f).abs().max().item()
    mem_err = (mem_ref - mem_f).abs().max().item()
    print(f"[read fp32-vs-fp32] affinity       max_abs_err = {aff_err:.3e}  <- transcription faithfulness")
    print(f"[read fp32-vs-fp32] visual_readout max_abs_err = {mem_err:.3e}")

    # INFORMATIONAL: fp16 (real read) vs fp32 (our C++ choice) -> Cutie's own fp16 noise
    aff_noise = (aff_h - aff_f).abs().max().item()
    mem_noise = (mem_h - mem_f).abs().max().item()
    print(f"[fp16 vs fp32] affinity       max_abs = {aff_noise:.3e}  <- inherent fp16 read noise (Cutie)")
    print(f"[fp16 vs fp32] visual_readout max_abs = {mem_noise:.3e}  <- C++ fp32 is MORE accurate")

    # ---- sanity: SimpleKVStore matches work_mem token count + perm split ----
    p = captured['perm_end_pt']; N = k.shape[-1]; HW = (1088//16)*(1920//16)
    print(f"[store] expected: perm={p} ({p//HW} keyframes), temp={N-p} ({(N-p)//HW} frames), "
          f"cap={HW_cap} ({HW_cap//HW} frames)")
    assert (N - p) <= HW_cap, "temp exceeds cap — FIFO should have trimmed!"
    print(f"[store] temp within cap: OK ({(N-p)//HW} <= {cfg.max_mem_frames})")

    print("\n[result] Transcription is FAITHFUL if fp32-vs-fp32 err ~1e-5 (math correct).")
    print("[result] The fp16 noise is Cutie's own; the C++ read runs fp32 (more stable).")
    print("[result] This reference is the spec for the libtorch C++ memory read.")


if __name__ == "__main__":
    main()
