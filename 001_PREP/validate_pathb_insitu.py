#!/usr/bin/env python3
"""
validate_pathb_insitu.py  —  FULL Path-B end-to-end reference (the C++ spec)

This is what the C++ plugin will do, in Python, end to end:
  - TRT engines for encode_image (E1), transform_key (E2), mask_encoder (E3),
    decoder (E5)  [via the existing hooks]
  - the KV memory store (Cutie's work_mem; proven identical to our SimpleKVStore
    layout: permanent|temporary + FIFO, use_long_term=False)
  - the memory READ done in fp32 (our reference_memory_read, the Option-B choice)
    instead of Cutie's native fp16 read
  - pixel_fusion + object transformer kept in libtorch (Path B)

Compares two passes on the same clip:
  (A) ORACLE: pure PyTorch, native (fp16) read  -- the ground truth
  (B) PATH B: all TRT engines + fp32 memory read -- what the C++ implements
and writes alpha mattes at native plate res for visual inspection in Nuke.

The ONLY intended differences B-vs-A: the four engines (fp16/fp32 vs autocast
torch) and the fp32 read (vs fp16). Everything else identical. If IoU stays high
with no NaN AND the mattes look clean, the Path-B architecture is validated and
the C++ has a proven spec + reference output.

Run (env: conda cutie311; CUDA12.8 + TRT10.9 on LD_LIBRARY_PATH):
  python validate_pathb_insitu.py \
      --frames /home/pm/Documents/CATs/EXR/grimes2.%04d.exr \
      --roto   /home/pm/Documents/CATs/MASK/MASK1.%04d.exr \
      --seeds  1,48,96 --n 96 --start 1 \
      --out-dir /home/pm/Documents/SAM3/out/pathb_val
"""

import argparse, os, types, math
import numpy as np
import torch
import cv2
import OpenEXR, Imath
from PIL import Image

from cutie.utils.get_default_model import get_default_model
from cutie.inference.inference_core import InferenceCore
import cutie_mask_propagate as oracle
from trt_feature_store import TRTImageFeatureStore
from trt_decoder_hook import install_trt_decoder
from trt_mask_encoder_hook import install_trt_mask_encoder
from memory_read_reference import reference_memory_read


def write_exr_alpha(path, a):
    H, W = a.shape
    hdr = OpenEXR.Header(W, H)
    hdr['channels'] = {'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    o = OpenEXR.OutputFile(path, hdr); o.writePixels({'A': a.astype(np.float32).tobytes()}); o.close()


def resize_exr_rgb(src, dst, W, H):
    a = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if a.ndim == 2: a = np.repeat(a[..., None], 3, 2)
    rgb = a[..., :3].astype(np.float32)
    if (rgb.shape[1], rgb.shape[0]) != (W, H):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    hdr = OpenEXR.Header(W, H); ch = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    hdr['channels'] = {'R': ch, 'G': ch, 'B': ch}
    o = OpenEXR.OutputFile(dst, hdr)
    o.writePixels({'R': rgb[..., 0].tobytes(), 'G': rgb[..., 1].tobytes(), 'B': rgb[..., 2].tobytes()})
    o.close()


def install_fp32_read(network, memory_manager):
    """Patch MemoryManager.read so the affinity/readout (get_similarity/do_softmax/
    readout) run in fp32 via reference_memory_read, while keeping storage +
    pixel_fusion + transformer exactly as Cutie's. This is the Path-B read seam:
    the C++ owns this fp32 read; libtorch does fusion + transformer."""
    mm = memory_manager
def install_fp32_read(network, memory_manager, ts_module=None):
    """Patch MemoryManager.read so the affinity/readout (get_similarity/do_softmax/
    readout) run in fp32 via reference_memory_read, while keeping storage +
    pixel_fusion + transformer exactly as Cutie's. This is the Path-B read seam:
    the C++ owns this fp32 read; libtorch does fusion + transformer.

    If ts_module is given (a torch.jit.load of fusion_transformer.pt), the fusion +
    transformer run THROUGH THE TRACED .pt instead of the eager modules -- i.e. the
    exact artifact + call contract the C++ uses. This validates the .pt in the full
    recurrent loop, not just on one random input."""
    mm = memory_manager
    orig_read = mm.read

    def read_fp32(pix_feat, query_key, selection, last_mask, net):
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        all_readout_mem = {}
        for bucket_id, bucket in mm.work_mem.buckets.items():
            mk = mm.work_mem.k[bucket_id]
            ms = mm.work_mem.s[bucket_id]
            # ---- fp32 memory read (Option B) ----
            with torch.amp.autocast("cuda", enabled=False):
                for objects in [bucket]:
                    mv = torch.stack([mm.work_mem.v[o] for o in objects], dim=1)  # 1xMxCVxN
                    M = mv.shape[1]
                    visual_readouts = []
                    for oi in range(M):
                        v_oi = mv[:, oi]                      # 1xCVxN
                        mem_oi, _ = reference_memory_read(
                            mk, ms, v_oi, query_key, selection, top_k=mm.top_k)
                        visual_readouts.append(mem_oi.view(bs, 1, mm.CV, h, w))
                    visual_readout = torch.cat(visual_readouts, dim=1)
                this_sensory = mm._get_sensory_by_ids(objects)
                this_last_mask = mm._get_mask_by_ids(last_mask, objects)
                this_obj_mem = mm._get_object_mem_by_ids(objects)  # (1,1,Q,emb+1)

            if ts_module is not None:
                # ---- TRACED fusion_transformer.pt (the C++ artifact) ----
                # wrapper does obj_memory.unsqueeze(2) internally -> pass pre-unsqueeze.
                with torch.amp.autocast("cuda", enabled=False):
                    readout_memory = ts_module(
                        pix_feat.float(), visual_readout.float(), this_sensory.float(),
                        this_last_mask.float(), this_obj_mem.float())
            else:
                # ---- eager libtorch pixel_fusion + transformer ----
                pixel_readout = net.pixel_fusion(pix_feat, visual_readout,
                                                 this_sensory, this_last_mask)
                obj_mem = this_obj_mem.unsqueeze(2) if this_obj_mem is not None else None
                readout_memory, _ = net.readout_query(pixel_readout, obj_mem)

            for i, obj in enumerate(objects):
                all_readout_mem[obj] = readout_memory[:, i]
        return all_readout_mem

    mm.read = read_fp32
    return orig_read


@torch.inference_mode()
def propagate(network, cfg, store, frame_paths, roto_resized, seeds0, device, tag,
              fp32_read=False, ts_module=None):
    core = InferenceCore(network, cfg=cfg, image_feature_store=store)
    core.max_internal_size = -1
    if fp32_read:
        install_fp32_read(network, core.memory, ts_module=ts_module)
    out = []
    autocast = torch.amp.autocast("cuda")
    with autocast:
        for ti, fp in enumerate(frame_paths):
            image = oracle.load_image_rgb_float(fp)
            H, W = image.shape[-2], image.shape[-1]
            if ti in seeds0:
                r = roto_resized[ti]
                if r.shape != (H, W):
                    r = np.array(Image.fromarray(r.astype(np.uint8) * 255)
                                 .resize((W, H), Image.NEAREST)) > 127
                lab = torch.zeros((H, W), dtype=torch.long); lab[torch.from_numpy(r)] = 1
                prob = core.step(image, lab.to(device), objects=[1], idx_mask=True,
                                 force_permanent=True)
            else:
                prob = core.step(image)
            alpha = prob[1].float().cpu().numpy() if prob.shape[0] > 1 else prob[0].float().cpu().numpy()
            out.append(alpha)
            if ti % 16 == 0 or ti in seeds0:
                tagseed = " <seed>" if ti in seeds0 else ""
                print(f"  [{tag}] frame {ti:3d} fg_max={alpha.max():.3f}{tagseed}")
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--roto", required=True)
    ap.add_argument("--seeds", default="1,48,96")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--resize-to", default="1920x1080")
    ap.add_argument("--out-size", default=None)
    ap.add_argument("--encode-engine", default="encode_image.fp16.engine")
    ap.add_argument("--key-engine", default="transform_key.fp16.engine")
    ap.add_argument("--mask-encoder-engine", default="mask_encoder.fp32.engine")
    ap.add_argument("--decoder-engine", default="mask_decoder.fp32.engine")
    ap.add_argument("--ts-module", default=None,
                    help="path to fusion_transformer.pt; if set, pass B runs fusion+"
                         "transformer through the TRACED .pt (the C++ artifact) "
                         "instead of eager modules.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = "cuda"
    seeds0 = sorted(int(s) - args.start for s in args.seeds.split(","))
    RW, RH = (int(x) for x in args.resize_to.lower().split("x"))
    _probe = cv2.imread(args.frames % args.start, cv2.IMREAD_UNCHANGED)
    NW, NH = _probe.shape[1], _probe.shape[0]
    OW, OH = (int(x) for x in args.out_size.lower().split("x")) if args.out_size else (NW, NH)
    print(f"[io] native={NW}x{NH} internal={RW}x{RH} output={OW}x{OH}")

    fdir = "/tmp/pathb_frames"; os.makedirs(fdir, exist_ok=True)
    frame_paths = []
    for i in range(args.n):
        dst = os.path.join(fdir, f"f{i:04d}.exr")
        resize_exr_rgb(args.frames % (args.start + i), dst, RW, RH)
        frame_paths.append(dst)
    roto_resized = {}
    for s0 in seeds0:
        a = oracle.load_roto_binary(args.roto % (args.start + s0)).astype(np.uint8)
        a = np.array(Image.fromarray(a * 255).resize((RW, RH), Image.NEAREST)) > 127
        roto_resized[s0] = a
    print(f"[prep] {len(frame_paths)} frames + {len(seeds0)} roto resized to {RW}x{RH}")

    def fresh_model():
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        return get_default_model().to(device).eval()

    netA = fresh_model()
    cfg = netA.cfg
    print("[A] ORACLE (pure PyTorch, native fp16 read)")
    alphaA = propagate(netA, cfg, None, frame_paths, roto_resized, seeds0, device, "A",
                       fp32_read=False)

    print("[B] PATH B (TRT engines + fp32 read + libtorch fusion/transformer)")
    netB = fresh_model()
    install_trt_decoder(netB, args.decoder_engine, device=device)
    install_trt_mask_encoder(netB, args.mask_encoder_engine, device=device)
    storeB = TRTImageFeatureStore(netB, encode_engine=args.encode_engine, key_engine=args.key_engine)
    ts_mod = None
    if args.ts_module:
        ts_mod = torch.jit.load(args.ts_module).to(device).eval()
        print(f"[B]   fusion+transformer via TRACED {args.ts_module} (the C++ artifact)")
    alphaB = propagate(netB, cfg, storeB, frame_paths, roto_resized, seeds0, device, "B",
                       fp32_read=True, ts_module=ts_mod)

    print("\n[compare] per-frame IoU@0.5 / soft_max_abs")
    worst_iou, worst_abs, nan_frames = 1.0, 0.0, []
    for i, (a, b) in enumerate(zip(alphaA, alphaB)):
        if np.isnan(b).any(): nan_frames.append(i)
        am, bm = a > 0.5, b > 0.5
        union = np.logical_or(am, bm).sum()
        iou = 1.0 if union == 0 else np.logical_and(am, bm).sum() / union
        mab = float(np.abs(a - b).max())
        worst_iou = min(worst_iou, iou); worst_abs = max(worst_abs, mab)
        if i % 16 == 0 or iou < 0.99:
            print(f"  frame {i:3d} IoU={iou:.4f} max_abs={mab:.3e}")
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            def to_out(x):
                if (x.shape[1], x.shape[0]) == (OW, OH): return x
                return cv2.resize(x, (OW, OH), interpolation=cv2.INTER_LINEAR)
            write_exr_alpha(os.path.join(args.out_dir, f"oracle_{i:04d}.exr"), to_out(a))
            write_exr_alpha(os.path.join(args.out_dir, f"pathb_{i:04d}.exr"), to_out(b))
            write_exr_alpha(os.path.join(args.out_dir, f"diff_{i:04d}.exr"), to_out(np.abs(a - b)))
    print(f"\n[result] worst IoU={worst_iou:.4f}  worst soft_max_abs={worst_abs:.3e}")
    print(f"[result] NaN frames: {nan_frames if nan_frames else 'none'}")
    if args.out_dir: print(f"[result] wrote mattes at {OW}x{OH} to {args.out_dir}")
    print("[result] Path B = the C++ architecture. Clean here => C++ has a proven spec.")


if __name__ == "__main__":
    main()
