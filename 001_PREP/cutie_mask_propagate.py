#!/usr/bin/env python3
"""
cutie_mask_propagate.py  (Cutie base-mega native mask seeding)
--------------------------------------------------------------
Seed Cutie's video object segmenter from hand-roto'd alpha masks on a few
keyframes and propagate the masklet across the whole clip.

Same CLI / IO as sam2_mask_propagate.py so outputs are directly A/B-comparable
in Nuke (drop both out dirs on top of each other).

Why Cutie may beat SAM 2 on fine detail (e.g. thin wing spokes):
  * max_internal_size = -1 runs at the plate's native resolution (SAM 2 was
    locked to 1024 internally, which broke up sub-pixel structure).
  * Object-level + pixel-level memory with retained high-res features, trained
    on VOS data to propagate THIS mask faithfully (vs SAM's "find the object").

Key Cutie specifics handled here:
  * step(image, mask, objects, *, force_permanent=...) walks frames in order.
    We pin each roto keyframe with force_permanent=True so EVERY frame can read
    all keyframes (XMem++-style permanent memory), not just later frames.
  * Masks are integer-label tensors (idx_mask=True). White roto (255) is mapped
    to the object's label id before seeding.
  * output_prob_to_mask() converts soft probs -> a single label image; we split
    it back into per-object binary PNGs to match the SAM 2 runner's layout.

Verified API on this install:
  get_default_model() -> CUTIE (base-mega)
  InferenceCore(cutie, cfg=cutie.cfg); .max_internal_size = -1
  step(image: CHW float cuda, mask: HxW int cuda or None, objects: list[int] or None,
       *, idx_mask=True, force_permanent=False) -> output_prob
  output_prob_to_mask(output_prob) -> HxW int label tensor

Usage:
  python cutie_mask_propagate.py \\
    --start-number 1 \\
    --frames /home/pm/Documents/CATs/JPG \\
    --prompt 1:1:/home/pm/Documents/CATs/MASK/MASK.0001.png \\
    --prompt 48:1:/home/pm/Documents/CATs/MASK/MASK.0048.png \\
    --prompt 96:1:/home/pm/Documents/CATs/MASK/MASK.0096.png \\
    --output /home/pm/Documents/SAM3/out/cutie_001
"""

import argparse
import os
import re
from pathlib import Path

# Enable OpenCV's OpenEXR codec BEFORE importing cv2 (must be set first).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

# OpenEXR (3.x) for reading alpha-only EXR rotos and writing named-'A' mattes;
# cv2 can't read a lone-A EXR. Optional: cv2 fallback if the binding is absent.
try:
    import OpenEXR as _OpenEXR
    _HAVE_OPENEXR = True
except Exception:
    _HAVE_OPENEXR = False

_EXR_EXTS = (".exr", ".EXR")


def _natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def parse_prompt(s):
    """FRAME:OBJ_ID:MASK_PATH -> (frame:int, obj_id:int, path:str)"""
    parts = s.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--prompt must be FRAME:OBJ_ID:MASK_PATH, got {s!r}")
    frame, obj_id, path = parts
    return int(frame), int(obj_id), path


def _read_exr_matte(path):
    """Read an EXR matte -> float32 (H,W). Prefers OpenEXR (handles alpha-only
    EXRs and arbitrary channel names that cv2 chokes on); cv2 fallback."""
    if _HAVE_OPENEXR:
        with _OpenEXR.File(str(path)) as f:
            ch = f.parts[0].channels
            names = list(ch.keys())
            # prefer a real alpha; else luma of RGB; else whatever single channel exists
            if "A" in ch:
                pix = ch["A"].pixels
            elif all(c in ch for c in ("R", "G", "B")):
                pix = (ch["R"].pixels.astype(np.float32)
                       + ch["G"].pixels.astype(np.float32)
                       + ch["B"].pixels.astype(np.float32)) / 3.0
            else:
                pix = ch[names[0]].pixels
                print(f"[roto] {Path(path).name}: using channel {names[0]!r} "
                      f"(available: {names})")
            return np.asarray(pix, dtype=np.float32)
    # cv2 fallback (may fail on alpha-only EXRs)
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise SystemExit(
            f"failed to read EXR roto {path} (cv2 can't read alpha-only EXR and "
            f"OpenEXR is unavailable; pip install OpenEXR)")
    if a.ndim == 3:
        a = a[..., 3] if a.shape[2] == 4 else a[..., :3].mean(axis=2)
    return a.astype(np.float32)


def load_roto_binary(path):
    """Load a roto matte -> bool (H,W) foreground mask.
    PNG/JPEG via PIL (alpha if present, else luma); EXR via OpenEXR/cv2 (float)."""
    p = Path(path)
    if p.suffix in _EXR_EXTS:
        return _read_exr_matte(p) > 0.5  # float matte: foreground where > 0.5
    img = Image.open(p)
    if img.mode in ("RGBA", "LA"):
        a = np.array(img)[..., -1]
    else:
        a = np.array(img.convert("L"))
    return a > 127  # 8-bit: foreground where > 127


def _linear_to_srgb(x):
    """Linear -> sRGB OETF (display-referred), x in [0,1]."""
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92,
                    (1 + a) * np.power(np.clip(x, 0, None), 1 / 2.4) - a)


def load_image_rgb_float(path, exr_exposure=0.0, exr_gamma=None):
    """Load any frame as a CHW float32 torch tensor (RGB, 0..1) on CUDA for Cutie.

    JPEG/PNG: standard sRGB 8-bit -> to_tensor (already 0..1, display-referred).
    EXR: linear float (cv2 BGRA) -> RGB, apply exposure (stops) + linear->sRGB
         (or a custom gamma), clamp [0,1]. Cutie only needs a sane display image
         to TRACK; plate precision is irrelevant to silhouette finding.
    """
    p = Path(path)
    if p.suffix in _EXR_EXTS:
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # HxWx{3,4} float32 BGR(A)
        if a is None:
            raise SystemExit(f"cv2 failed to read EXR (codec?): {p}")
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        rgb = a[..., :3][..., ::-1].astype(np.float32)  # BGR->RGB, drop alpha
        if exr_exposure:
            rgb = rgb * (2.0 ** exr_exposure)
        rgb = np.clip(rgb, 0.0, 1.0)
        if exr_gamma:                       # explicit gamma override
            rgb = np.power(rgb, 1.0 / exr_gamma)
        else:                               # default: linear -> sRGB
            rgb = _linear_to_srgb(rgb)
        rgb = np.clip(rgb, 0.0, 1.0)
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
        return t.cuda().float()
    # 8-bit sRGB path
    return to_tensor(Image.open(p).convert("RGB")).cuda().float()


def _write_binary_png(out_root, obj_id, real_frame, mask_bool_hw):
    obj_dir = out_root / f"obj_{obj_id}"
    obj_dir.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask_bool_hw).astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(obj_dir / f"mask.{real_frame:04d}.png")


def _write_alpha_exr(out_root, obj_id, real_frame, mask_bool_hw):
    """Write a single-channel float32 EXR matte in a channel literally named 'A'
    (drops straight into Nuke's alpha, no shuffle). Falls back to cv2 (channel
    named 'Y'/'R') only if the OpenEXR binding isn't available."""
    obj_dir = out_root / f"obj_{obj_id}"
    obj_dir.mkdir(parents=True, exist_ok=True)
    alpha = np.ascontiguousarray(np.asarray(mask_bool_hw).astype(np.float32))  # HxW 0/1
    out = obj_dir / f"mask.{real_frame:04d}.exr"
    if _HAVE_OPENEXR:
        header = {"compression": _OpenEXR.ZIP_COMPRESSION}
        channels = {"A": _OpenEXR.Channel(alpha)}
        with _OpenEXR.File(header, channels) as f:
            f.write(str(out))
    else:
        if not cv2.imwrite(str(out), alpha):  # single-channel, but named Y/R
            raise SystemExit(f"cv2 failed to write EXR: {out}")


def main():
    ap = argparse.ArgumentParser(description="Cutie mask-seeded video propagation")
    ap.add_argument("--frames", required=True, help="directory of frames")
    ap.add_argument("--prompt", action="append", required=True, type=parse_prompt,
                    metavar="FRAME:OBJ_ID:MASK_PATH",
                    help="seed obj_id with roto mask on FRAME; repeatable")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--start-number", type=int, default=0,
                    help="real frame number of the first frame (maps timeline "
                         "numbers to 0-based indices; applies to prompts AND "
                         "output filenames)")
    ap.add_argument("--max-internal-size", type=int, default=-1,
                    help="resize shorter edge to this for internal processing; "
                         "-1 = native resolution (best detail, most VRAM). "
                         "Step down to 1080/720 if you OOM.")
    ap.add_argument("--no-permanent", action="store_true",
                    help="do NOT pin keyframes to permanent memory (use plain "
                         "sequential memory like the scripting demo)")
    ap.add_argument("--no-passthrough-seeds", action="store_true",
                    help="write Cutie's re-emitted mask on seed frames instead "
                         "of the verbatim hand roto (default: pass roto through)")
    ap.add_argument("--mask-format", choices=["png", "exr"], default="png",
                    help="output matte format. png = 8-bit L; exr = 1-channel "
                         "float32 alpha (hard 0/1 edges until a matte head is added)")
    ap.add_argument("--exr-exposure", type=float, default=0.0,
                    help="exposure in stops applied to EXR input before sRGB "
                         "conversion (only affects what Cutie sees, not output)")
    ap.add_argument("--exr-gamma", type=float, default=None,
                    help="override the default linear->sRGB with a plain gamma "
                         "(e.g. 2.2) for EXR input; default uses true sRGB OETF")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required.")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- frame list (natural sort so grimes2.0001..0096 order correctly) ---
    frame_dir = Path(args.frames)
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".exr", ".EXR")
    frames = sorted([f for f in frame_dir.iterdir() if f.suffix in exts],
                    key=lambda f: _natural_key(f.name))
    if not frames:
        raise SystemExit(f"no frames found in {frame_dir}")
    num_frames = len(frames)
    print(f"[frames] {num_frames} frames, index 0 = {frames[0].name} ... "
          f"index {num_frames-1} = {frames[-1].name}")

    # ---- map real frame numbers -> 0-based indices, group seeds by frame ----
    start = args.start_number

    def to_idx(real_frame):
        idx = real_frame - start
        if not (0 <= idx < num_frames):
            raise SystemExit(
                f"--prompt frame {real_frame} -> index {idx} out of range "
                f"[0,{num_frames-1}] (start-number={start})")
        return idx

    # seeds_by_idx[frame_idx] = list of (obj_id, roto_bool_HW, roto_path)
    seeds_by_idx = {}
    all_obj_ids = set()
    for real_frame, obj_id, path in args.prompt:
        fidx = to_idx(real_frame)
        roto = load_roto_binary(path)
        seeds_by_idx.setdefault(fidx, []).append((obj_id, roto, path))
        all_obj_ids.add(obj_id)
        print(f"[seed] frame {real_frame} (idx {fidx})  obj {obj_id}  <- {path}")
    seed_idxs = set(seeds_by_idx.keys())

    # ---- build model -------------------------------------------------------
    print("[build] loading Cutie base-mega ...")
    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = args.max_internal_size
    print(f"[build] max_internal_size = {processor.max_internal_size} "
          f"({'native res' if args.max_internal_size < 0 else str(args.max_internal_size)+'px short edge'})")

    permanent = not args.no_permanent

    # ---- single forward pass: seed at keyframes, propagate elsewhere -------
    write_fn = _write_alpha_exr if args.mask_format == "exr" else _write_binary_png
    print(f"[output] format = {args.mask_format}")
    autocast = torch.amp.autocast("cuda")
    written = 0
    with torch.inference_mode(), autocast:
        for ti, fpath in enumerate(frames):
            image = load_image_rgb_float(
                fpath, exr_exposure=args.exr_exposure, exr_gamma=args.exr_gamma)
            H, W = image.shape[-2], image.shape[-1]

            if ti in seed_idxs:
                # build a single integer-label mask for all objects seeded here
                label = torch.zeros((H, W), dtype=torch.long)
                obj_ids_here = []
                for obj_id, roto, _path in seeds_by_idx[ti]:
                    r = roto
                    if r.shape != (H, W):
                        # resize roto to frame res (nearest -> keep crisp labels)
                        r = np.array(Image.fromarray(r.astype(np.uint8) * 255)
                                     .resize((W, H), Image.NEAREST)) > 127
                    label[torch.from_numpy(r)] = obj_id
                    obj_ids_here.append(obj_id)
                label = label.cuda()
                output_prob = processor.step(
                    image, label, objects=sorted(set(obj_ids_here)),
                    idx_mask=True, force_permanent=permanent,
                )
            else:
                output_prob = processor.step(image)

            label_mask = processor.output_prob_to_mask(output_prob)  # (H,W) int
            lm = label_mask.cpu().numpy().astype(np.int32)

            real = ti + start
            for obj_id in sorted(all_obj_ids):
                if (not args.no_passthrough_seeds) and ti in seed_idxs and \
                        any(o == obj_id for o, _, _ in seeds_by_idx[ti]):
                    # verbatim roto passthrough on seeded keyframes
                    roto = next(r for o, r, _ in seeds_by_idx[ti] if o == obj_id)
                    if roto.shape != (H, W):
                        roto = np.array(Image.fromarray(roto.astype(np.uint8) * 255)
                                        .resize((W, H), Image.NEAREST)) > 127
                    write_fn(out_root, obj_id, real, roto)
                else:
                    write_fn(out_root, obj_id, real, lm == obj_id)
                written += 1

            if ti % 10 == 0 or ti == num_frames - 1:
                print(f"[propagate] frame idx {ti}/{num_frames-1} done")

    print(f"[done] wrote {written} masks under {out_root}")


if __name__ == "__main__":
    main()
