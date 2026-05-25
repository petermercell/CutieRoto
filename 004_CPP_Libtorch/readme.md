# Debugging & Tracing — CutieRoto

CutieRoto emits diagnostic tracing to **stderr**, so the most useful thing you can
do when something misbehaves is **launch Nuke from a terminal** and read the
`[CutieRoto] …` lines. This document explains the two trace levels, what each line
means, how to read a healthy run, and how to diagnose the common failure modes.

---

## Trace levels

There are two independent levels, both writing to stderr:

| Level | Env var | Default | Volume | What it covers |
|-------|---------|---------|--------|----------------|
| **Lifecycle** | `CUTIE_DEBUG` | **on** | low | Process run: pipeline build, keyframe encoding, memory-store growth, per-frame propagation, errors |
| **Per-scanline engine** | `CUTIE_DEBUG_ENGINE` | off | firehose | The viewer serve path inside `engine()`: one burst per scanline × every worker thread |

> **Note:** the lifecycle level is currently **forced on** in source
> (`cutieDebugOn()` returns `true`). To make it opt-in via `CUTIE_DEBUG`, restore the
> `getenv("CUTIE_DEBUG")` form left commented directly above that function in
> `src/CutieRoto.cpp`, and rebuild. The per-scanline level is already opt-in.

### Enabling

```bash
# lifecycle trace only (current default — just launch from a terminal)
/opt/Nuke17.0v1/Nuke17.0

# add the per-scanline engine firehose (rarely needed; see warning below)
CUTIE_DEBUG_ENGINE=1 /opt/Nuke17.0v1/Nuke17.0
```

> **Warning:** `CUTIE_DEBUG_ENGINE=1` logs every scanline from every viewer worker
> thread and flushes stderr per line, which serializes the worker pool and noticeably
> slows scrubbing. Use it only when debugging the cache serve geometry; leave it off
> for normal work and performance measurement.

---

## Lifecycle trace (`CUTIE_DEBUG`)

These print during knob interaction and during a Process run.

| Line | Meaning |
|------|---------|
| `knob_changed: '<name>'` | A knob fired (e.g. `showPanel`, `process`). |
| `-> process button` | The Process button was pressed. |
| `processAllFrames: ENTER` | The two-phase analyze + propagate run started. |
| `buildPipeline: CUDA ok, N device(s), using device D` | GPU validated; engines/`.pt` loaded onto device `D`. |
| `plate dims for cache: fW=… fH=…` | Plate (input 0) resolution; the cached matte size. |
| `range A..B, K keyframes` | Propagation range and number of reference keyframes parsed. |
| `analyzed keyframe frame=F (perm now P)` | A keyframe was encoded into **permanent** memory. `P` grows by **8160** per keyframe. |
| `displayStep: visual readout [1 256 68 120] store N=… perm=…` | One propagated frame. `N` = total memory tokens, `perm` = permanent (keyframe) tokens. |
| `propagated frame N (cache=N)` | Progress marker, printed every 10 frames. |
| `clear() with N engine serve(s) in flight (safe: shared_ptr keeps buffers alive)` | A Process overlapped a live viewer serve. **Benign** — the shared_ptr cache keeps each in-flight buffer alive. (This was the once-fatal race.) |
| `processAllFrames: DONE, cached N frames` | Run finished; `N` matmtches the range length on success. |
| `CUDA error [expr] at file:line -> <msg>` | A CUDA call/sync failed at a located stage. Converted to a clean `Op::error` instead of a crash. |
| NaN-guard messages (`nanchk`) | A stage produced NaNs; names the stage so you can localize. |
| `CutieRoto … pull frame N: <what>` (Nuke error console) | An input pull threw (e.g. CUDA OOM); the node stops cleanly with the frame and reason. |

### Memory-store invariants

The store is fixed-config; these relationships are your sanity checks:

- **`perm` should equal `keyframes × 8160`.** 3 keyframes → `perm = 24480`. If it doesn't, a keyframe didn't encode.
- **`N` caps at `perm + 40800`.** Temporary (propagated) frames are FIFO-trimmed to `max_work_tokens = 5 × 8160 = 40800`; the permanent region is never trimmed. Once the clip has propagated a few frames, `N` stops growing — that bound is what keeps VRAM flat.
- **`cached N frames` on `DONE` should equal the range length** (`B − A + 1`).

---

## Per-scanline engine trace (`CUTIE_DEBUG_ENGINE`)

These come from `engine()` — the viewer pull path that serves the cached matte.
High volume; interleaved across threads (lines may fragment mid-write, which is normal).

| Line | Meaning |
|------|---------|
| `engine ENTER: frame=F y=Y x=X r=R` | A scanline request entered `engine()`. |
| `engine: frame=F cached=C cw=… ch=… fW=… fH=…` | Per-frame summary (first scanline only): whether the frame is cached and its dimensions. |
| `engine[y=Y x=X r=R]: input0->get` | Pulled the plate row (pass-through path). |
| `engine[y=Y]: get writable Chan_Alpha (x=X r=R)` | Acquired the output alpha row. |
| `engine[y=Y]: outA=0x… (r-x=…)` | Output buffer pointer and row width. |
| `engine[y=Y]: SERVE cached size=… cw=… ch=… fX=… fY=… fW=… fH=…` | Serving this scanline from the cached matte; dimensions travel **with** the buffer. |
| `engine[y=Y]: py=… dyTop=… sy=… srcRow=cached+…` | The source-row math mapping plate-space `y` into the cached matte. |
| `engine[y=Y]: serve loop done` | Finished serving the scanline. |

A wall of `SERVE cached size=… cw=… ch=…` lines from many threads with **no crash**
is the expected, healthy picture under concurrent viewer load.

---

## Reading a healthy Process run

Abridged, with annotations:

```
[CutieRoto] -> process button
[CutieRoto] processAllFrames: ENTER
[CutieRoto] buildPipeline: CUDA ok, 1 device(s), using device 0
[CutieRoto] plate dims for cache: fW=2048 fH=1080
[CutieRoto] range 1..96, 3 keyframes
[CutieRoto] analyzed keyframe frame=1  (perm now 8160)     # 1 × 8160
[CutieRoto] analyzed keyframe frame=50 (perm now 16320)    # 2 × 8160
[CutieRoto] analyzed keyframe frame=96 (perm now 24480)    # 3 × 8160  -> perm done
[CutieRoto] displayStep: … store N=24480 perm=24480        # temp starts empty
[CutieRoto] displayStep: … store N=32640 perm=24480        # +1 temp frame (every 5th)
…
[CutieRoto] displayStep: … store N=65280 perm=24480        # N capped: 24480 + 40800
[CutieRoto] propagated frame 30 (cache=30)
…                                                          # N stays 65280 the rest of the clip
[CutieRoto] processAllFrames: DONE, cached 96 frames        # == range length
```

The tell-tale of a correct run: `perm` settles at `keyframes × 8160`, `N` rises and
then **pins** at `perm + 40800`, and `cached N frames` equals the range.

---

## Diagnosing common symptoms

**Process stops part-way (node "killed", Nuke survives).** Almost always VRAM. Look at
the **last `store N=…` line before the stop** and any `CUDA error […]` message. Because
the per-frame peak grows with `N`, a smaller GPU crosses its limit at a smaller `N`
(and a bigger plate raises the baseline, so it stops sooner). Mitigations, in order:
1. Confirm it's memory: watch `nvidia-smi` during Process; if it dies near the card's limit, it's VRAM.
2. Lower the chunk budget: `kBudget` in `src/memory_core.cpp` (default `32 * 1024 * 1024`) caps the memory-read peak at ≈ `2 × (1, N, chunk)`. Halving it roughly halves that peak (at the cost of more, smaller passes).
3. The proactive `emptyCache` in `displayStep` already reclaims fragmented pool below 3 GB free; on a very small card you can raise that threshold.

**`CUDA error […]`.** A located CUDA failure (device/stream/kernel). The first one is
the real cause; subsequent frames may fail too because a kernel fault corrupts the
context. Note the `file:line` and the error string.

**NaNs (`nanchk`).** A stage produced NaNs. With very few keyframes the fp32 memory
read can be sensitive; the softmax is max-subtracted to guard overflow, so persistent
NaNs usually point at a bad input (empty roto, mis-mapped keyframe).

**`clear() with N serve(s) in flight`.** Not an error — informational. It means a
Process began while the viewer was still serving cached frames; the shared_ptr cache
makes that safe. Seeing it is confirmation the concurrency fix is doing its job.

---

## Diagnostic builds (tests)

These live in `src/tests/` and are wired into CMake. They don't need Nuke.

**Cache-race regression test** — proves the serve/clear concurrency fix and fails if it
regresses. Configure with the sanitizer, build both targets, run:

```bash
cmake -B build -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 -DCUTIE_LIBTORCH_BACKEND=ON \
      -DCUTIE_SANITIZE_RACE=ON -DCUTIE_RACE_SANITIZER=address   # or =thread
cmake --build build -j --target test_cache_race test_cache_race_buggy

./build/test_cache_race                                    # FIXED: must be CLEAN
ASAN_OPTIONS=halt_on_error=1 ./build/test_cache_race_buggy # BUGGY: must REPORT
```

Expected: the fixed build runs both phases and exits clean; the buggy build aborts in
**phase 1** at `deterministicProbe` — `heap-use-after-free` under AddressSanitizer, or a
`data race` / `SEGV` at the same line under ThreadSanitizer. ASan is the reliable gate
(the bug is fundamentally a use-after-free).

**Memory-core check** — validates the chunked memory read against the Python fp32
reference (errors should be ≈ 0, well under `1e-3`):

```bash
cmake --build build -j --target test_memory
./build/test_memory test_data/memcase.pt
```

**Embedded-asset smoke test** — loads the 5 embedded `.pt` blobs and runs the full
engine chain, verifying handoff shapes:

```bash
cmake --build build -j --target test_torch_embedded
./build/test_torch_embedded
```

---

## Making it quiet for release

The lifecycle trace prints on every launch while it is forced on. For a release build,
edit `cutieDebugOn()` in `src/CutieRoto.cpp` to use the commented `getenv("CUTIE_DEBUG")`
form, and rebuild. Then a normal launch is silent, and both levels are opt-in:

```bash
CUTIE_DEBUG=1 /opt/Nuke17.0v1/Nuke17.0          # lifecycle trace
CUTIE_DEBUG_ENGINE=1 /opt/Nuke17.0v1/Nuke17.0   # + per-scanline engine firehose
```
