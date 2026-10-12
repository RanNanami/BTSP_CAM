# ImageNet-R k96 cross-dataset protocol

Read `IMAGENET_R_PROTOCOL_AUDIT.md` and `IMAGENET_R_PREPROCESS_AUDIT.md` for provenance. This is **not** an author parameter recovery or formal paper-exact reproduction.

- ImageNet-R: supplied PILOT ImageFolder `train/` and `test/`, 200 classes, B0-Inc20, ten tasks. Class order is PILOT's legacy NumPy seed 1993 permutation; labels are positions in that permutation. The loader does not create a new train/test split.
- Frozen `PilotFrozenViT` ViT-B/16 AugReg ImageNet-21K, 768-D. Train: one seeded PILOT random-resized-crop/flip/ToTensor per image; test: PILOT bicubic resize/center crop/ToTensor; no normalization.
- Fixed cross-dataset local working point: `m=8192`, slots=4096, `k=96`, `q=4`, `tau=4/96`, `f_q_base=.005`, beta=1, trace decay=.01, p=.5, eta=1, gamma=.1. No ImageNet-R tuning or test-based threshold choice.
- Main readout `EXPERIMENTAL_CURRENT_WB_REPLAY_REFIT`: historical training sparse codes are stored under `experimental_replay_codes/` and re-evaluated with current W_B. **NOT STRICT EXEMPLAR-FREE; NOT CLAIMED AS PAPER ALGORITHM.**
- Control `LITERAL_STREAMING_EQ8`: the same writer produces current-task Z with that task's W_B, and only its A/B increment is retained. Old Z is neither recomputed nor transported. One writer trajectory serves both readouts.
- The classifier scores all seen classes without a task ID. Current-task 20-way-restricted accuracy is an additional diagnostic only.

No scientific ImageNet-R result exists in this preparation round. The supplied split's creation procedure, content hashes and identity to author-distributed preprocessing remain unresolved until data arrives.
