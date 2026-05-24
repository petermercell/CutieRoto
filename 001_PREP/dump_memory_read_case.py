#!/usr/bin/env python3
"""
dump_memory_read_case.py  —  produce a test case for the C++ memory core.

Runs a real Cutie propagation, snapshots the work_mem buffers (k/s/v +
perm_end_pt) and the query key/selection at --check-frame, computes the fp32
reference read (get_similarity -> do_softmax(top_k) -> readout) with autocast
OFF, and saves everything as a single TorchScript-loadable archive that the C++
test_memory binary loads via torch::load. The C++ then runs its KVStore +
memoryRead on the SAME inputs and must match the saved fp32 read bit-exactly.

Outputs:  memcase.pt   (a dict of tensors + scalars, saved via torch.jit.save of
                         a tiny Module holding them as buffers — portable to C++)

Run (env cutie311):
  python dump_memory_read_case.py \
      --frames /home/pm/Documents/CATs/EXR/grimes2.%04d.exr \
      --roto   /home/pm/Documents/CATs/MASK/MASK1.%04d.exr \
      --seeds 1,48,96 --n 96 --start 1 --check-frame 30 --out memcase.pt
"""

import argparse, os
import numpy as np
import torch
import cv2, OpenEXR, Imath
from PIL import Image

from cutie.utils.get_default_model import get_default_model
from cutie.inference.inference_core import InferenceCore
import cutie_mask_propagate as oracle
from memory_read_reference import reference_memory_read


class CaseHolder(torch.nn.Module):
    """Holds tensors as buffers so torch.jit.save writes a C++-loadable archive."""
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            self.register_buffer(k, v)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--roto", required=True)
    ap.add_argument("--seeds", default="1,48,96")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--check-frame", type=int, default=30)
    ap.add_argument("--out", default="memcase.pt")
    args = ap.parse_args()
    dev = "cuda"
    seeds0 = sorted(int(x) - args.start for x in args.seeds.split(","))
    chk = args.check_frame

    m = get_default_model().to(dev).eval()
    cfg = m.cfg
    RW, RH = 1920, 1080

    fdir = "/tmp/memcase_frames"; os.makedirs(fdir, exist_ok=True)
    fpaths = []
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
        o.close(); fpaths.append(p)
    roto = {}
    for s0 in seeds0:
        rb = oracle.load_roto_binary(args.roto % (args.start + s0)).astype(np.uint8)
        rb = np.array(Image.fromarray(rb * 255).resize((RW, RH), Image.NEAREST)) > 127
        roto[s0] = rb

    core = InferenceCore(m, cfg=cfg); core.max_internal_size = -1
    cap = {}
    orig = core.memory.read
    def spy(pix_feat, query_key, selection, last_mask, net):
        if core.curr_ti == chk and 'k' not in cap:
            bid = next(iter(core.memory.work_mem.buckets.keys()))
            obj = core.memory.work_mem.buckets[bid][0]
            cap['k'] = core.memory.work_mem.k[bid].detach().float().cpu().clone()
            cap['s'] = core.memory.work_mem.s[bid].detach().float().cpu().clone()
            vraw = core.memory.work_mem.v[obj].detach().float()
            cap['v'] = vraw.reshape(vraw.shape[0], vraw.shape[1], -1).cpu().clone()
            cap['perm_end_pt'] = torch.tensor(int(core.memory.work_mem.perm_end_pt[bid]))
            cap['query_key'] = query_key.detach().float().cpu().clone()
            cap['selection'] = selection.detach().float().cpu().clone()
            cap['top_k'] = torch.tensor(int(core.memory.top_k))
        return orig(pix_feat, query_key, selection, last_mask, net)
    core.memory.read = spy

    autocast = torch.amp.autocast("cuda")
    with autocast:
        for ti, fp in enumerate(fpaths):
            image = oracle.load_image_rgb_float(fp)
            H, W = image.shape[-2], image.shape[-1]
            if ti in seeds0:
                r = roto[ti]
                if r.shape != (H, W):
                    r = np.array(Image.fromarray(r.astype(np.uint8)*255).resize((W,H), Image.NEAREST))>127
                lab = torch.zeros((H,W), dtype=torch.long); lab[torch.from_numpy(r)] = 1
                core.step(image, lab.cuda(), objects=[1], idx_mask=True, force_permanent=True)
            else:
                core.step(image)
            if ti == chk: break
    if 'k' not in cap:
        raise SystemExit(f"check-frame {chk} not reached")

    # fp32 reference read (autocast OFF) — the C++ must match this
    with torch.amp.autocast("cuda", enabled=False):
        k = cap['k'].cuda(); s = cap['s'].cuda(); v = cap['v'].cuda()
        qk = cap['query_key'].cuda(); qe = cap['selection'].cuda()
        mem_ref, aff_ref = reference_memory_read(k, s, v, qk, qe, top_k=int(cap['top_k']))
    cap['ref_readout'] = mem_ref.detach().float().cpu().clone()   # (1,CV,HW)
    cap['ref_affinity'] = aff_ref.detach().float().cpu().clone()  # (1,N,HW)

    N = cap['k'].shape[-1]; p = int(cap['perm_end_pt']); HW = (1088//16)*(1920//16)
    print(f"[case] N={N} perm={p}({p//HW}kf) temp={N-p}({(N-p)//HW}f) "
          f"CK={cap['k'].shape[1]} CV={cap['v'].shape[1]} top_k={int(cap['top_k'])}")
    print(f"[case] query h*w = {cap['query_key'].shape[-2]}x{cap['query_key'].shape[-1]}")
    print(f"[case] ref_readout {tuple(cap['ref_readout'].shape)}  "
          f"ref_affinity {tuple(cap['ref_affinity'].shape)}")

    CaseHolder(cap).cpu()  # ensure all on cpu
    torch.jit.save(torch.jit.script(CaseHolder(cap).cpu()), args.out)
    print(f"[case] wrote {args.out} (load in C++ with torch::jit::load)")


if __name__ == "__main__":
    main()
