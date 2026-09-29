# Round-2 final report

All Layer-A numbers below are labeled **NOT PAPER-COMPARABLE**.

## 1. Bugs found, ordered by severity

1. **High:** `default_rng(1993)` did not reproduce PILOT's legacy
   `np.random.seed` + `np.random.permutation`, changing all tasks.
2. **Medium:** plateau-slot and synapse-flip draws shared one RNG stream.
3. **Medium:** ridge columns were preallocated instead of being expanded while
   preserving earlier `B` columns.
4. **Medium:** an invalid base plateau probability could be silently clipped.
5. **Low:** Top-k tie membership had no explicit stable policy.
6. **Low, diagnostics only:** the first smoke attempt passed the dataset index,
   rather than the image tensor, to input-stat collection and failed before
   training.

## 2. Fixes applied

All six findings were fixed and covered either by unit tests or the real smoke.
The older plateau test changed intentionally: `f_base=10` now raises instead of
being silently converted into a valid-looking probability.

## 3. CIFAR-100 B0-Inc5 protocol status

Strictly connected at the PILOT-protocol level: `cifar224`, shuffle true,
seed 1993, `init_cls=5`, increment 5, memory 0, 20 stages, current-task-only
training, cumulative seen-class testing, remapped incremental labels, and no
task ID.

## 4. Exact class-order parity

Pass. An actual pinned official PILOT `DataManager` is instantiated in the
test suite and compared elementwise. The complete order is saved in the result
directory.

## 5. Backbone, checkpoint, and preprocessing

- Paper level: frozen ViT-B/16 pretrained on ImageNet-21K.
- PILOT level: timm 0.6.12 name `vit_base_patch16_224_in21k`, 768-D output.
- Runtime inference: timm 1.0.29 alias
  `vit_base_patch16_224.augreg_in21k`, backed by the same Google AugReg
  `B_16-i21k-300ep...npz` URL.
- PILOT transforms are reproduced exactly for non-CODA CIFAR224. They end in
  `ToTensor()` and do not apply timm mean/std normalization.
- The model is in eval mode, frozen, and repeated eval extraction is exact.

## 6. Missing parameters found

No previously absent BTSP-CAM numeric value was found as paper-specified.
SimpleBTSP supports only candidate inheritance values: `p=0.5`,
`f_q_base=0.005`, and the sparsity-derived candidate `k=41` at `m=8192`.

## 7. Parameters still unresolved

`k`, `f_q_base`, `p`, `beta`, the state-dependent `lambda` rule, `tau`, and
`gamma`. Alpha is not a blocker because this round is BTSP-only.

## 8. Eta result

With zero initialization, a fixed positive global eta, and eta-independent
decay, every trace is eta times its eta=1 trace. Positive scaling cancels in
cosine similarity, including the explicit zero-trace convention. Eta is
therefore `DERIVED` scale-invariant under the stated assumptions and is fixed
to 1.

## 9. Real CIFAR smoke result

Pass for two complete tasks. Train/test sizes were 2,500/500 and 2,500/1,000.
Input tensors were float32 `[32,3,224,224]` in `[0,1]`; ViT features were
non-degenerate; Top-k was exactly 3; plateau probability/rate, binary-memory
activity, and CA1 activity were non-degenerate; ridge solves were `ok`; Task 2
evaluated ten classes; replay count was zero. Diagnostic cumulative accuracy
was 0.370 then 0.278 and has no paper-comparison status.

## 10. Readiness for the full 20-task run

The data/protocol/backbone and core execution path are ready. A strict formal
reproduction is **not ready** because critical method parameters remain
unresolved. Do not run or tune a formal 20-task experiment toward the paper's
reported accuracy. A future run with disclosed local choices must be labeled
an independent reproduction, not paper parity.
