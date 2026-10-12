# ImageNet-R protocol audit (before adapter implementation)

Reference: BTSP-CAM local PDF, printed pp. 6-7, Section 5.1 and Table 1; pinned LAMDA-PILOT commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`. This is a cross-dataset operationalization, not a recovered author configuration.

| Item | Finding | Evidence/status |
| --- | --- | --- |
| Class count | 200 | `iImageNetR.class_order = np.arange(200)` in `upstream/LAMDA-PILOT/utils/data.py:224`; **PILOT_SPECIFIED**. Paper Table 1 says ImageNet-R, but does not restate the count. |
| Split | B0-Inc20: ten 20-class tasks, no separate base task | Paper Table 1 (printed p.6) labels `IN-R B0 Inc20`; **PAPER_SPECIFIED**. PILOT `exps/simplecil_inr.json` sets `init_cls=20`, `increment=20`, `shuffle=true`; `utils/data_manager.py:9-24`; **PILOT_SPECIFIED**. |
| Class order | Legacy NumPy permutation with seed 1993 | Paper Section 5.1, printed p.6, fixes seed 1993; **PAPER_SPECIFIED**. PILOT experiment config `seed:[1993]` and `DataManager._setup_data` at `utils/data_manager.py:140-158` call `np.random.seed(seed); np.random.permutation(...)`; **PILOT_SPECIFIED**. |
| Train/test source | Separate supplied `train/` and `test/` ImageFolder trees | `iImageNetR.download_data` in `utils/data.py:227-236` reads `./data/imagenet-r/{train,test}/`; **PILOT_SPECIFIED**. Neither this loader nor the paper specifies how those trees were generated, their counts, or hashes; split provenance **UNRESOLVED** until a dataset manifest is supplied. The CLI root override changes only path resolution, not split semantics; **LOCAL_OPERATIONALIZATION**. |
| Loader / mapping | Sorted ImageFolder class names; train and test must match; PILOT remaps original integer class to its position in shuffled order | `utils/data.py:232-236`, `utils/data_manager.py:157-164,182-183`; **PILOT_SPECIFIED**. Explicit identity and disjointness checks are **LOCAL_OPERATIONALIZATION**. |
| Training/evaluation | Current task train labels only; cumulative seen-class test labels; no task ID for classifier | `DataManager.get_dataset` at `utils/data_manager.py:33-80` selects label intervals. Paper Section 5.1 calls this standard CIL; **PAPER_SPECIFIED** for metric, **PILOT_SPECIFIED** for selection. |
| Backbone | Frozen ViT-B/16 ImageNet-21K, 768-D | Paper Section 5.1 pp.6-7; **PAPER_SPECIFIED**. Existing `PilotFrozenViT` uses modern timm AugReg alias for pinned PILOT's model identifier; version substitution is **LOCAL_OPERATIONALIZATION** already documented in `docs/CIFAR100_PROTOCOL.md`. |
| Non-CODA preprocessing | Train random resized crop 224 (scale .05-1, ratio .75-1.333), horizontal flip .5, ToTensor. Test bicubic resize 256, center crop 224, ToTensor. No Normalize. | `iImageNetR.__init__` in `utils/data.py:208-222` chooses `build_transform`; definition `utils/data.py:95-120`; **PILOT_SPECIFIED**. The fixed one-seeded-augmentation feature-cache policy is **LOCAL_OPERATIONALIZATION** inherited from the CIFAR/CUB project. |

The locally absent `data/imagenet-r` means actual sample counts, split integrity, and server readiness cannot yet be marked PASS. We do not synthesize an 80/20 split or infer train/test provenance from the benchmark name.

`FINAL_CASE = B`: PILOT expects pre-split train/test data; the origin and sample-level split rule remain unresolved. Server infrastructure is ready, but a paper-comparable run is conditional on verifying split provenance.
