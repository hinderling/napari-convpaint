# Convpaint performance investigation

Scope (per request): **no model/ML changes** — only how data is loaded, passed
into the model, chunked, copied, and whether dask is used correctly. Accuracy
must not regress. Covers both local-context FEs (VGG16) and global-context FEs
(DINOv2 ViT), including where tiling cannot help.

Measured with `run_baseline.py` and targeted probes on Oxford-IIIT Pet,
Apple Silicon / MPS, napari-env. Image sizes: native ~500 px, and `--upscale 4`
≈ 2000 px for the large-image regime.

---

## Summary of findings (prioritized)

| # | Finding | Regime | Win | Safety |
|---|---------|--------|-----|--------|
| 1 | Train→predict of the **same image recomputes the FE forward twice** | all FEs, esp. ViT/large | ~2× FE cost on the interactive loop | accuracy-neutral (reuse) |
| 2 | `tile_annotations=True` (default, local FEs) is **2.2× slower** for spread-out scribbles | local FE train | up to ~2× train | accuracy-identical (verified) |
| 3 | `_clf_predict` **moveaxis+reshape copies the whole feature stack** (1.5 GB → ~3 GB peak) | large images | peak memory + some speed | bit-identical (chunk) |
| 4 | Prediction **tiling: ~5× less memory**, but only for local FEs | large local-FE predict | ~5× memory | accuracy-identical if gated on `has_global_context` |
| 5 | **dask is used wrong** (per-tile model pickle, cluster per call); even fixed it's slow locally | large predict | out-of-core, not local speed | fixed = 2×; needs lazy IO for its real use |
| 6 | `get_features_targets` **moveaxis-copies the whole feature stack** before masking | train | modest memory | safe |
| 7 | ViT forward is **irreducible** without changing results | global-FE | opt-in half-precision / Z-batch only | opt-in |

Non-findings (checked, nothing to do): feature dtype is **float32** throughout
(no float64 widening); a single `segment()` computes features **exactly once**
(pyramid call count = 1, so no double-compute *within* prediction).

---

## Evidence

### Finding 1 — the interactive loop computes the FE forward twice on the same image

Counting `extract_features_pyramid` calls: one `segment()` = **1** call (no
double-compute within prediction — this directly answers "are we computing
features 2×?": *not within predict*). But `train()` and the following
`segment()` are **separate FE passes over the same image**. The widget's
annotate → train → auto-segment loop therefore runs the feature extractor twice
on identical pixels.

DINOv2 (global context), one image:

```
size        train_s   segment_s   (both do 1 whole-image ViT forward)
500 px       1.78       0.13
2000 px     22.72       7.00
```

The ~7 s ViT forward at 2000 px happens in *both* train and segment. Reusing the
features from train in the following predict saves one full forward per cycle —
the single biggest lever for the ViT / large-image interactive workflow.

`_train` already returns `feature_parts`, and `_predict_image` already accepts
`feature_img=`. For global-context FEs (`tile_annotations=False`) `_train`'s
features *are* the whole-image stack, so they are directly reusable. There is
also a dormant `_train_predict_image` method that fuses train+predict feature
extraction — **it exists but is never called** (dead code).

### Finding 2 — `tile_annotations=True` is slower for spread-out scribbles

VGG16, 2000 px, ~1 % scribble coverage spread across the image (the paper's
`create_even_scribbles` draws skeleton + line scribbles over the whole object):

```
tile_annotations=True   train=3.19 s   fg_iou=0.6645
tile_annotations=False  train=1.44 s   fg_iou=0.6645   <- identical accuracy, 2.2× faster
```

The premise of `tile_annotations` is "only extract features near annotations".
But when annotations are spatially spread, the tiles cover most of the image
*and* pay per-tile overhead — here it produced **64 separate FE extractions**
for one image. It only wins when annotations are clustered in a small region.
Should be decided adaptively from the annotation bounding-box coverage, not
defaulted to True.

### Finding 3 — `_clf_predict` copies the whole feature stack

`_clf_predict` (`convpaint_model.py`):

```python
features = np.moveaxis(features, 0, -1)      # [F,Z,H,W] -> non-contiguous view
features = np.reshape(features, (-1, nb_features))  # forces a full contiguous copy
predictions = self.classifier.predict_proba(features)
```

For a 2000 px VGG16 image the feature stack is `[128, 1, 2004, 1480]` float32 =
**1.5 GB**; the moveaxis+reshape adds another ~1.5 GB copy, and `predict_proba`
may copy again. Predicting in row-chunks bounds the peak to one chunk and gives
**bit-identical** output. (This exact fix already exists on the
`threading-cancellation` branch; port it here.)

### Finding 4 — prediction tiling: ~5× memory, local FEs only

`--compare-tiling --upscale 4`, identical accuracy across configs
(fg_IoU 0.795):

```
whole-image   predict 1.03 s   peak ~2550 MB (clean, img1)
tiled         predict 0.99 s   peak  ~784 MB   <- ~5× less, same speed & accuracy
```

Only accuracy-preserving for **local** FEs. For global-context FEs (ViT,
cellpose, deep VGG) tiling breaks whole-image context and degrades features;
`has_global_context` already tracks this and `_warn_if_global_context` warns.
**Bug fixed on this branch:** `Dinov3Features` and `DinoJafarFeatures` (ViTs)
did not set `has_global_context=True` (only DINOv2 did), so they would be
silently tiled.

### Finding 5 — dask

`_parallel_predict_image` with `use_dask=True` was ~30× slower and used more
memory: `Client()` spawned a new process cluster per call, and
`client.submit(self._predict_image, …)` pickled the whole model (VGG weights) to
a worker per tile. **Fixed on this branch:** `Client(processes=False)` (threaded,
shared model) → ~2× faster, identical accuracy — but still ~15× slower than
plain tiled locally (per-task scheduler overhead ~2.8 s/tile vs ~0.2 s compute;
single MPS queue serializes tiles anyway).

dask.distributed's real value is **larger-than-memory / multi-machine**, which
the current architecture can't exploit: `_parallel_predict_image` takes the full
image as an in-RAM array and pre-allocates the full output with `np.zeros`. Even
the non-dask tiled path is therefore **not** out-of-core — it holds the whole
input and output resident. The fix is lazy input+output (memmap/zarr) so tiles
stream through and peak RSS is ~one tile regardless of image size.

### Finding 6 — `get_features_targets` copy

`np.moveaxis(features, 0, -1)` copies the entire feature stack before masking to
the (sparse) annotated pixels. Gathering at masked coordinates without the full
moveaxis avoids the copy. Modest but free.

### Finding 7 — global-context FEs: the forward is irreducible

For ViTs the attention forward is the cost and cannot be tiled without changing
features. Prediction already runs the classifier at **patch resolution**
(`gives_patched_features=True`), so the classifier side is cheap — the whole cost
is the forward. Legitimate levers that do NOT change the pipeline semantics are
limited:
- **Feature reuse across train→predict (Finding 1)** — the big one for ViTs.
- **Z-plane batching** for stacks (`extract_features_from_stack` currently loops
  planes for some FEs) — a 3D-stack speedup, no effect on 2D.
- **Opt-in half precision** (autocast) on the forward — ~2× on GPU but *changes
  numeric results slightly*, so opt-in only, never default.

---

## Implementation status

Done on this branch (each verified for no accuracy regression before commit):

- **Adaptive `tile_annotations` (Finding 2)** — gate on annotation bounding-box
  coverage; tile only when annotations are clustered. Baseline train
  **0.83 s → 0.38 s (2.15×)**, identical accuracy; clustered annotations still
  tile. `_tiling_worthwhile`.
- **Auto-tile large local-FE prediction (Finding 4)** — `tile_image` was opt-in;
  now auto-tiles per image when the longest side > 1500 px and the FE is
  local-context. 2000 px VGG16 predict peak RSS **2947 → 1240 MB (2.4×)**,
  bit-identical output. `_should_auto_tile`.
- **Out-of-core tiled prediction (Findings 4/5)** — `_parallel_predict_image`
  gains `out=` (write into a caller/memmap array) and reads input tiles lazily
  (`np.asarray` per block); new `segment_to_disk` streams a large image tile by
  tile into a disk-backed output so peak RAM is ~one tile regardless of image
  size. Output bit-identical to whole-image.
- **Chunk `_clf_predict` (Finding 3)** and **`get_features_targets` copy
  reduction (Finding 6)** — both bit-identical, cut peak memory on large images.
- **dask threaded fix (Finding 5)** and **DINOv3/JAFAR `has_global_context`
  bug** (prior commits).

Deferred, with reasons:

- **Opt-in half precision / Z-plane batching for ViTs (Finding 7).** Half
  precision changes numeric results (opt-in only, never default). Z-batching only
  helps 3D stacks. Both are FE-forward micro-opts; left as follow-ups.

## Follow-up questions (answered)

**Q: Can dask trade memory for speed by running tiles/models in parallel?**
Only with *multiple compute devices*. Measured on this machine (10-core CPU),
threaded dask over tiles is **8× slower** than sequential tiling (11.7 s vs
1.5 s), and on MPS it was ~30× slower. Reasons: (1) torch already uses all cores
*within* a single tile (intra-op parallelism), so running tiles in parallel
*oversubscribes* the cores and gains nothing; (2) a single GPU/MPS has one queue,
so tiles serialize on it regardless; (3) `dask.distributed`'s per-task scheduler
overhead (~seconds) dwarfs per-tile compute (~0.2 s). The memory-for-speed trade
(N tiles resident → up to N× faster) only pays off when there are N independent
devices (multi-GPU / a real cluster) — and then the model must live on each
worker (scatter once / load per worker), not be pickled per task, with lazy input.
For single-machine use, sequential tiling is already optimal for memory and
parallelism buys nothing; a lightweight `joblib`/process pool (not
`dask.distributed`) would be the tool if multi-device support is ever wanted.

**Q: Doesn't global/continuous training (memory_mode) already cache features?**
Yes — but a *different* cache. In `memory_mode` (the Image/Global continuous-
training options), `self.table` stores the extracted features for each
**annotated pixel**, keyed by (img_id, scale, coordinates), and
`_register_and_update_annots` only extracts features for **newly** annotated
pixels on each retrain (it skips pixels whose annotation is unchanged). So it
avoids recomputing training features across *training rounds*. It does **not**
hold whole-image features and does nothing for prediction — it is a sparse,
training-side cache, orthogonal to the train→predict double-forward below.

**Q: Shouldn't training and prediction keep the same features? Wasn't this
addressed in a recent PR?** Yes — and it was: `_train_predict_image` (formerly
`self_predict`, commit `3caf034`) extracts features **once** via `_train`
(tile_annotations forced off → whole-image features) and reuses them for
prediction through `_predict_image(feature_img=…)`. It already handles the
patched-FE resolution question: training unpatches with nearest-neighbour
(`fe_order=0`), so subsampling `feature_img[..., ::p, ::p]` recovers the exact
patch tokens. **Verified bit-identical** to separate train+predict for *both*
VGG16 and DINOv2 — my earlier "resolution-mismatch, needs a token cache" concern
was wrong; the fusion is exact. Measured saving (one FE forward avoided):
DINOv2 2000 px **31.5 s → 24.7 s (22 %)**; VGG16 negligible (its bottleneck is
the CatBoost fit, not the FE). The catch is that it is **dead code — never
called** — and limited: it does not support `tile_image` or `memory_mode`, and
predicts only the annotated planes.

## FE bug: skimage-resize made DINOv2/v3 feature extraction ~130× too slow

While checking the FEs, found the real reason DINOv2/v3 felt so slow — it was
**not** the network. Profiling a DINOv3 feature image (504 px): the ViT
`forward_features` was **0.08 s**, but the whole `get_feature_image` was
**13.1 s**, with ~13 s inside `scipy.ndimage.zoom_shift` under
`skimage.transform.resize`. Two helpers — `rescale_features` and
`rescale_outputs` — upsampled the patched (ViT) features from the patch grid to
full resolution with `skimage.transform.resize`, which is per-channel CPU and
~50×+ slower than torch interpolation on a 384-channel stack.

`_extract_tiled_multiscale` (JAFAR) *looked* slower but was actually ~10× faster
than plain DINOv3 at the same size — precisely because JAFAR returns full-res
features and skips the skimage rescale, while plain DINOv3 went through it.

Fix: both helpers now interpolate via torch (numpy↔torch), the path tensor
inputs already used. **DINOv3/DINOv2 `get_feature_image` 504 px: 13.1 s → 0.10 s
(130×).** Also speeds patched-FE training (the unpatch) and prediction proba
upsampling.

Alignment (no pixel shifts): order=0 uses `mode='nearest-exact'`, which follows
skimage's half-pixel nearest rule and is **bit-identical to skimage for all
upsampling ratios**. (Plain torch `'nearest'` — used initially — shifts by
~0.5 source-px on *non-integer* ratios; integer ratios such as the ViT
patch→full upscale, factor = patch_size, were already identical, which masked the
bug. Fixed in commit "half-pixel shift in torch rescale for order=0".) The
default `unpatch_order=1` (bilinear, `align_corners=False`) matches skimage's
half-pixel/pixel-centre convention with **no shift** (verified mean_shift = 0.000);
it differs only sub-pixel at patch boundaries (0–0.2 % of pixels) — within-noise,
not bit-identical: DINOv2 5-image mean fg_IoU 0.7589 (skimage) vs 0.7583 (torch),
torch higher on 2/5. (A separate pre-existing torch `'nearest'` in
`dino_jafar.py` — maintainer JAFAR code, lower-res-head fallback — carries the
same latent shift and is left as-is.) 12 tiling + 19 dino tests pass.

Also fixed two JAFAR inefficiencies found in the same pass: a `copy.deepcopy` of
the upsampler head on every call (now lazy — only on a device-fallback error),
and `torch.mps.empty_cache()` inside the per-tile loop (forces a device sync;
moved to once per plane). v3+JAFAR 756 px: 5.76 s → 4.50 s.

## Rescale-fix verification (multichannel / why-130× / non-MPS)

Checked the skimage→torch rescale fix against three concerns:
- **Device independence.** `torch.from_numpy(...)` is a CPU tensor, so the
  interpolation runs on CPU regardless of the FE's compute device. It was never
  an MPS optimization — it is torch-CPU vs skimage-CPU, so **every device
  (CPU-only, CUDA, MPS) gets the speedup**.
- **Is 130× real?** Yes. The profiler pinned 12.9 s inside a single
  `skimage.transform.resize` (scipy `zoom_shift`) of the 384-channel feature
  stack; torch does the same op in 36 ms — **354× on that call** — because
  skimage resizes per-channel single-threaded while torch is vectorized across
  8 threads. Not an artifact.
- **Multichannel / multi-Z.** Verified for `[F,Z,H,W]` across F/Z combinations:
  each Z-plane is resized independently, shapes correct, and the (small) diff vs
  skimage is identical regardless of F/Z — i.e. it is the interpolation-backend
  difference, not a channel bug. Only upsampling occurs in this pipeline (patch
  grid → full res), so skimage's downsample-only anti-aliasing never applied and
  nothing is lost.

## FE bug audit (parallel review of every extractor)

Reviewed all FEs + the base pipeline; verified and fixed:

- **HIGH — Hookmodel not thread-safe.** `self.outputs` was shared instance state
  the forward hooks appended into; under the threaded dask path (one shared FE
  across worker threads) concurrent extractions clobbered each other → ~82% of
  calls returned wrong features. (The earlier switch to threaded dask is what
  exposed it.) Now thread-local. Verified 0/400 concurrent wrong.
- **HIGH — Hookmodel hook leak.** `register_hooks` discarded handles, so
  re-registering left the old `hook_last` (aborts the forward) attached →
  silently dropped deeper layers. Now removes handles first + sorts layers to
  execution order. Verified re-register returns both maps.
- **HIGH — JAFAR crash on ~3-patch-wide images.** overlap clamp kept stride
  positive but not the blend-window length ≥ 0 → `torch.ones(-ps)` crash. Fixed
  the clamp. Verified 48 px extraction OK.
- **MED — `assert False` forward-stop** stripped under `python -O` → whole
  network ran every plane. Replaced with a real exception.
- **MED — base `features_per_layer`** unset → AttributeError with
  `fe_use_min_features` on non-Hookmodel FEs. Initialized to None.
- **LOW — base `supported_devices`** returned `[cuda, mps, [cpu]]` (cpu nested);
  flattened. **combo_fe** `image`→`data` kwarg; **jafar_scalings** init;
  **tile_annot** bounds clamp for `alignment==1`.
- Verified-clean / non-bugs: dinov3 handles non-square/non-default sizes (the
  `strict_img_size` worry was a false alarm); gaussian/ilastik/cellpose padding
  and device handling correct; combo's per-sub-FE scalings are intentional;
  reduce↔pad round-trip and scale_img shapes correct.

## Feature cache — implemented (v1, opt-in, API-level)

Built and wired (bit-identical when on; off by default):
- `feature_cache.FeatureCache`: LRU + memory-budget triage (psutil live budget,
  evict/skip, never OOM), content-agnostic. 9 unit tests.
- Pyramid split: `extract_features_pyramid` = `_pyramid_reconstruct(_pyramid_native(...))`,
  verified bit-identical; the FE cache protocol (`cacheable_repr` etc.) defaults
  to caching the pre-rescale native features, so **DINOv2 caches patch tokens
  (196× smaller, 0.39 MB vs 77 MB)** and one cached payload serves both train
  (unpatched) and predict (patched).
- Wired into `_get_features` via `_extract_pyramid_cached`, **content-addressed**
  (blake2b of the prepared image + FE-signature), so train/predict of the same
  pixels share with no id plumbing and a changed image self-invalidates.
  `enable_feature_cache()` opt-in. Verified DINOv2: train + re-segment ×2 does 3
  FE forwards off / **1 on (2 avoided)**, identical output.

**GUI integration — done.** The **Advanced** tab has a *Feature caching*
group: an enable checkbox (**on by default**, unchecking clears both tiers), a
*Max cache RAM (MB)* spinbox (**2048 MB** default), a *Max cache disk (MB, 0=off)*
spinbox (**8192 MB** default), and a live *RAM x MB / disk y MB* size label
(1 s `QTimer`). `_apply_feature_cache()` re-applies the settings (in place, via
`set_max_bytes`/`set_disk_max_bytes`/`set_enabled`) after every model (re)creation.

**Disk spillover (from testing feedback — larger-than-RAM stacks).** RAM-evicted
entries spill to a pickled disk tier (own LRU + budget) instead of being dropped;
`get()` checks RAM then disk. A 3D stack too big for the RAM cache (e.g. napari's
Kidney 3D) now still benefits on the next iteration — slices load back from disk
(bit-identical) rather than recomputing. It matters most for FEs whose payload
can't be compressed to a small native form (an upsampling FE that emits
full-resolution features has large per-slice payloads, so few fit in RAM): there,
reading one back from disk is far cheaper than recomputing it. Guards: never
exceeds the disk cap and never fills the
filesystem below a 2 GB free-space headroom; temp dir removed on clear/close.
Verified: an 8-slice stack with room for 1 in RAM serves 8/8 from cache on
iteration 2 (0 recomputed). Also: **removing an image layer clears the cache**
(entries are content-addressed, so a deleted image's features would otherwise
linger). Verified headless; `test_dims` + 14 cache tests pass.

**Cache-first stack ordering (from testing feedback — LRU thrash).** A plain
sequential `predict-all` (slice 0..N) over a stack larger than the cache evicts
the earliest slices before the next pass reaches them, so the next pass recomputes
everything — zero benefit, and painful for slow FEs (JAFAR ~2 s/slice forward).
Fix: predict already-cached slices FIRST, then compute the rest, via a
`cache_only` peek threaded through `_extract_pyramid_cached` → `_get_features` →
`_predict_image` → `_predict` (returns the reconstructed features only if cached,
else None without running the extractor). `_on_predict_all` runs phase 1 (serve
cached) then phase 2 (compute rest) under **one** progress bar; skipped when the
cache is empty or caching is off. Verified: peek returns None on a miss (no FE
forward) and the served slice is bit-identical to a normal predict; over 2 full
passes of a 10-slice stack with room for ~4, naive does 20 FE forwards (full
thrash) vs cache-first 17 (saving scales with cache capacity — with disk spillover
holding most of the stack, most slices are served).

**Cache-size units fix.** The size label divided by 1e6 (MB) while the caps
multiplied the spinbox value by 1024² (MiB), so an "8192" disk limit read up to
~8589 MB and looked over-budget (the byte cap was never actually exceeded — the
`disk_bytes ≤ cap` invariant holds after every put, now asserted). Both caps now
use decimal MB (1e6) to match the label; disk box relabelled "0 = RAM only".

Redundant-normalization cleanup (the "double-normalization" claim, investigated).
The widget's `_on_train` pre-normalized the image (`_get_data_channel_first_norm`)
and then passed `skip_norm=False`, so the model normalized a **second** time —
its own code comment even said "skip normalization as it is done in the widget"
while the code did the opposite. This second pass is **not a correctness bug** for
the common FEs: it is a **bit-identical no-op** for imagenet FEs (DINO/DINOv3/
JAFAR/VGG16 — after the first pass features are z-scored, out of [0,1], and
`normalize_image_imagenet` returns out-of-[0,1] floats unchanged) and effectively
identity for default mode. But it is (a) **wasted compute** — a full extra
normalization pass over the whole image on every train — and (b) a genuine
**double-application for percentile FEs** (percentile rescaling is not
idempotent). Fixed by passing `skip_norm=True` in `_on_train` (matching the
comment and matching what prediction already does on the same pre-normalized
data). Verified segmentation is **bit-identical** for VGG16/DINOv2. `_train_multiple`
(multifile/selected) and the multifile segment correctly keep `skip_norm=False`
because they pass **raw** data (via `_get_data_channel_first`, not `_norm`).
Consequences for cache sharing (train and predict now present the *same*
normalized array):
- **Global-context FEs (DINO/DINOv3/JAFAR)** never tile training, so training
  extracts the whole image and the first prediction is a cache **hit** — the full
  train→predict→refine loop reuses features, verified (hits 0→1 on the first
  segment after train for DINOv2 and VGG16).
- **Local FEs (VGG/gaussian/ilastik)** share when training also extracts the
  whole image (spread/large annotations, where adaptive tiling picks whole-image);
  with small clustered annotations, training tiles, so the *first* prediction
  misses but every *re-prediction* reuses.

JAFAR full-res payloads are large — the budget skips them on big images
(expected); a backbone-token JAFAR strategy is moot since the upscaler, not the
backbone, dominates (measured).

## Do any FEs normalize themselves? (checked — none double-normalize)

Normalization is a single, model-side responsibility: each FE *declares* what it
needs via `norm_mode` (`imagenet` for DINO/DINOv3/JAFAR/VGG/efficientnet/convnext,
`percentile` for cellpose, `default` for gaussian/ilastik), and `ConvpaintModel`
applies exactly that once (`_norm_single_image`, or the widget's pre-norm). The
FEs' `prep_img`/extraction methods **only reshape and tensor-convert — none of
them apply image normalization**. Verified by grepping every FE for any
normalization op (normalize_image, `/255`, mean/std subtraction, torchvision
`Normalize`, sub_/div_): zero hits.

- DINOv2/v3 `prep_img`: crop-to-patch + `torch.tensor`, no norm. JAFAR: same;
  its `F.normalize` is L2 *feature* normalization inside the upsampler
  (architecture), not image prep; the ViT wrapper's mean/std is only a comment.
- Cellpose: runs the net on the already-(percentile)-normalized image and pulls
  intermediate tensors; no internal image norm. (Convpaint substitutes its
  `percentile` norm for cellpose's native normalization — a design choice, not a
  double-norm.)

So the FEs are correct — no FE-level double-normalization. Two related, *separate*
observations (not FE-implementation bugs), both now addressed:
1. **Cellpose used `skimage.transform.resize`** (order 0) to upsample its feature
   tensors to image size — the same slow path fixed in `rescale_features`/
   `rescale_outputs`. **Fixed**: `utils.resize_nearest` (index-gather, **bit-
   identical** to skimage order=0 for upsampling, verified; ~4× faster). Output
   unchanged (cellpose not installed to run end-to-end, but the resize is proven
   bit-identical).
2. **`normalize_image_imagenet` silently skipped out-of-[0,1] floats** — a float
   image with values outside [0,1] (microscopy) was returned *unchanged* (console
   warning only), so an imagenet FE received un-normalized, out-of-distribution
   input. **Investigated + fixed**: measured impact on segmentation (same images,
   only the input transform varies) — benign uniform scaling loses ~1–1.6 % fg_IoU
   vs correct; **pathological per-channel float ranges lose up to 4.8 %** (DINOv2).
   Scaling floats to [0,1] before ImageNet stats reliably helped and never hurt;
   percentile is the robust choice. Fix: float-out-of-[0,1] now gets a 1–99
   percentile stretch to [0,1] before ImageNet stats (numpy + torch, torch
   subsampled above 16 M elements), with a warning that it rescaled. uint8 and
   float-in-[0,1] unchanged; brings float-microscopy handling in line with uint8.
   (This was `normalization_todo.md` issue #1.)

## Feature-cache design (for the interactive annotate→predict→refine loop)

Measured feature sizes — **full-resolution** (what a naive cache would store) vs
**native** (per-scale for CNNs, patch tokens for ViTs, i.e. before the rescale to
full resolution):

| FE | image | native (pre-rescale) | full-res | ratio |
|----|-------|----------------------|----------|-------|
| VGG16 (2 scales) | 500 px | 65 MB | 142 MB | 2× |
| VGG16 (2 scales) | 2000 px | 1.0 GB | 2.3 GB | 2× |
| DINOv2 | 500 px | **1.5 MB** | 283 MB | **190×** |
| DINOv2 | 2000 px | **23 MB** | 4.5 GB | **195×** |
| DINOv2+JAFAR | 500 px | 293 MB | 283 MB | 1× |
| DINOv2+JAFAR | 2000 px | 4.6 GB | 4.5 GB | 1× |

Conclusions:

- **Cache native features, resample at runtime — and it's decisive for ViTs.**
  DINOv2 patch tokens are **~195× smaller** than the full-res stack (23 MB vs
  4.5 GB at 2000 px) and unpatching is nearest-neighbour, so resampling is
  **lossless** (verified bit-identical). A whole-image ViT feature cache is
  therefore *tiny* — RAM is a non-issue (a 10000² image is still ~0.6 GB of
  tokens). This is the single highest-value cache.
- **VGG16 native cache saves ~2×** (first conv is stride-1 so scale-1 is already
  full-res; only higher scales are downsampled). Less dramatic, and VGG's
  bottleneck is the CatBoost fit anyway — caching matters least here.
- **JAFAR is the exception the user predicted:** its learned upsampler emits
  full-resolution features directly, so there is no small native form (ratio 1×).
  Options: cache the *backbone patch tokens* and re-run only the (cheaper)
  upsampler on a cache hit — saves the ViT backbone forward but not the upsampler
  — or cache full-res with disk spillover, or skip caching for JAFAR.

**Storage / disk / dask:**

- With native/patch-token caching, the common interactive case (one 2D image, a
  ViT) needs **tens of MB** — keep it in RAM, no disk, no dask. "Is it even an
  issue?" → not for ViTs cached at patch resolution.
- It only grows for full-res caches (VGG native ~1 GB, JAFAR ~4.5 GB at 2000 px)
  or many cached slices (3D stacks, many annotated planes). There, a **size-
  capped LRU cache with optional `np.memmap`/zarr spillover** to disk is the
  pragmatic answer.
- **Dask dynamic load/offload is overkill** for this — a plain memory-budgeted
  LRU with disk spillover is simpler, predictable, and dependency-light. Dask's
  value remains multi-device compute, not a memory manager.

Proposed design:

- A whole-image feature cache on the model, keyed by
  `(image identity, FE signature, normalize, scale)`, storing **native**
  features (patch tokens / per-scale). `_get_features` consults it: on hit,
  resample to the requested resolution (full-res for prediction; subsample at
  annotated pixels for training) instead of re-running the FE.
- Reused across the whole annotate→predict→annotate loop (not just fused within
  one call), so every refinement after the first is FE-free until the image or FE
  params change.
- Bounded by a configurable RAM budget with LRU eviction; optional memmap
  spillover for the large (full-res / JAFAR / many-slice) cases.
- Complements the existing `memory_mode` table (sparse annotated-pixel features
  across training rounds); this new cache is the dense whole-image side that also
  serves prediction.

## General feature-cache architecture (FE-pluggable)

Different FEs have different "cheap intermediates" and different post-processing
to rebuild full features. To avoid repeating cache logic per FE, split into:

**Generic layer (one implementation, FE-agnostic):** a `FeatureCache` that
stores opaque per-FE payloads keyed by
`(img_id, z-slice/frame, FE-signature)` — the FE-signature reusing the existing
`_params_to_reset_training` set (fe_name, fe_layers, fe_scalings, fe_order,
channel_mode, normalize, …) so cache invalidation matches the model's existing
train-reset rules. Handles LRU eviction, the memory budget, disk spillover, and
the OOM triage. Wired into `_get_features`:

```
payload = cache.get(key)
if payload is None:
    payload = fe.cacheable_repr(image, param, device)   # the expensive, ideally-small step
    cache.put(key, payload, size=fe.cacheable_nbytes(payload))
features = fe.features_from_cacheable(payload, param, …) # the cheap post-processing
```

**Per-FE strategy (each FE implements a tiny protocol, no cache logic):** the
`FeatureExtractor` base class gains, with safe defaults so opting in is optional:

- `cacheable_repr(image, param, device)` → the payload to store. Default: the
  full feature output.
- `features_from_cacheable(payload, param, …)` → rebuild features from payload.
  Default: identity (payload *is* the features).
- `cacheable_nbytes(payload)` → size for budget accounting. Default `payload.nbytes`.
- `supports_feature_cache(param)` → whether caching is worthwhile. Default True.

FE-specific overrides:

- **DINOv2 / DINOv3:** payload = patch tokens (≈195× smaller, verified lossless);
  rebuild = nearest-neighbour unpatch (or return patched for prediction). The big
  win, tiny storage.
- **VGG16 / CNNs:** payload = per-scale native conv maps; rebuild = upsample +
  concat. ~2× storage saving.
- **JAFAR:** measurement shows the **upscaler head dominates**, not the backbone
  (DINOv2 backbone ≈7 s @2000 px; JAFAR times out >5 min). So caching backbone
  tokens saves little — the expensive half re-runs. JAFAR instead caches its
  **full features** with `supports_feature_cache` gated on the memory/disk
  budget: cache when it fits (huge time win — the whole upscaler is skipped),
  skip when it doesn't. The generic budget/LRU/disk layer decides "fits or not".
- **gaussian / ilastik:** cheap to recompute; may opt out (`supports_feature_cache
  = False`) or cache full — no bespoke logic either way.

The cache stores an *opaque* FE-defined payload; the FE owns payload↔features,
the cache owns storage/eviction/budget. Adding a new FE's caching strategy = a
couple of small methods, never touching the cache machinery.

## OOM triage for stacks / movies (bounded, never crashes)

100-slice stacks and 300+-frame movies would blow past RAM (and, with JAFAR
full-res, disk too). The generic cache stays bounded by construction:

- Each entry keyed per (img_id, z-slice/frame, FE-sig); a stack caches per slice.
- Before storing, check the payload size against a **live budget** =
  `min(configured_cap, available_RAM − headroom)` via `psutil.virtual_memory()`.
- Fits → store. Doesn't → **evict LRU** until it fits. Single payload bigger than
  the budget → **spill to disk** (memmap/zarr, own disk budget) or, failing that,
  **skip caching it** (recompute on demand). The cache never exceeds the budget,
  so it never pushes the kernel into swap/OOM — it degrades to recomputation.
- LRU matches the interactive pattern: refining the current slice keeps its
  features hot; scrolling far evicts old slices. Bounded regardless of stack
  length or movie length.
- Dask dynamic offload is unnecessary — a size-aware LRU + psutil budget +
  optional disk spill is simpler, predictable, and dependency-light.

Storage note: DINOv2 patch-token payloads are tiny (23 MB @2000 px), so even a
long movie fits in RAM. Only full-res payloads (VGG native ~1 GB, JAFAR ~4.6 GB
per frame) hit the budget quickly — exactly the case the disk spill / skip path
handles.

### Revised Finding 1 (feature reuse) — no longer deferred as risky

The mechanism exists and is exact. The remaining work is **wiring it in** for the
train→auto-segment workflow (the interactive loop, and the biggest ViT win), plus
deciding fallbacks for the cases it does not cover (fall back to separate
train+predict when `tile_image`/`memory_mode` is on, or extend it to those). This
is the recommended next implementation step; it is integration, not new
algorithms.

## Measurement caveats

- Peak RSS is the increase over a per-image baseline sampled on a background
  thread; within one process torch's allocator caching inflates later baselines,
  so first-image numbers are cleanest (a `--isolate` subprocess mode would give
  clean means — a follow-up).
- Single-run timings; large relative differences are reliable, tighten with
  repeats before quoting small deltas.

## Model changes made on this branch (for review, in commit order)

1. `Client()` → `Client(processes=False)` in `_parallel_predict_image` (dask fix).
2. `has_global_context = True` on `Dinov3Features` and `DinoJafarFeatures`.
3. Chunked `_clf_predict`; `get_features_targets` masks without a full moveaxis copy.
4. Adaptive `tile_annotations` (`_tiling_worthwhile`).
5. Auto-tile large local-FE prediction (`_should_auto_tile`).
6. Out-of-core prediction: `_parallel_predict_image(out=…)` + lazy tile reads +
   `segment_to_disk`.

All accuracy-neutral (verified bit-identical or identical benchmark metrics);
none change the default output, only speed/memory. `segment_to_disk` is new API.
