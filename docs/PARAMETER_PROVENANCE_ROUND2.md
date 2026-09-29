# Parameter provenance round 2

## Search record

Search order followed the requested hierarchy:

1. Supplied 18-page paper and embedded supplement: complete review, no additional numeric method hyperparameters.
2. OpenReview forum `2HlW5mj6Ch`: public page was located, but revision/attachment API is currently challenge-protected; the supplied PDF matches the public May 2026 author upload.
3. Paper-declared GitHub `ericzhengz/BTSP-CAM`: HTTPS page and `git ls-remote` both return repository-not-found as of 2026-09-22.
4. GitHub repository/code search: no public mirror containing the implementation/formula identifiers was found.
5. PILOT official repository pinned at commit `7a6e904...`: inspected `utils/data_manager.py`, `utils/data.py`, `utils/inc_net.py`, configs, and model data-loader code. It contains protocol/backbone evidence but no BTSP-CAM hyperparameters.
6. Supplied official simpleBTSP source and local Nature Communications final paper (`s41467-024-55563-6.pdf`): provide inheritance candidates only.

## Provenance table

| parameter | numeric value | source | exact evidence | confidence | status |
|---|---:|---|---|---|---|
| RF dimension | 8192 | BTSP-CAM PDF | Sec. 5.1 | high | PAPER_SPECIFIED |
| memory slots | 4096 | BTSP-CAM PDF | Sec. 5.1 | high | PAPER_SPECIFIED |
| class-order seed | 1993 | BTSP-CAM PDF | Sec. 5.1 | high | PAPER_SPECIFIED |
| class split | 20 x 5 | PILOT + B0-Inc5 | `init_cls=5`, `increment=5`; `DataManager` increments | high | PILOT_SPECIFIED |
| projection | `R,b ~ N(0,1)` | BTSP-CAM PDF | Eq. (3), Eq. (10) | high | PAPER_SPECIFIED |
| `k` | unresolved | BTSP-CAM PDF | only `k/m << 1` and `k=o(sqrt(m))` | high that value is absent | UNRESOLVED |
| `k` candidate | 41 | simpleBTSP sparsity inheritance | `round(8192*0.005)=41`; simpleBTSP paper Methods uses `f_p=0.005` | low-medium | PLAUSIBLE_INHERITANCE |
| base plateau | unresolved | BTSP-CAM PDF | Eq. (5) names `f_q_base`, no value | high that value is absent | UNRESOLVED |
| base plateau candidate | 0.005 | simpleBTSP paper/code | simpleBTSP Methods default `f_q=0.005`; official feedforward line 25 | medium as a candidate only | PLAUSIBLE_INHERITANCE |
| synaptic flip `p` | unresolved in CAM | BTSP-CAM PDF | Eq. (2)/(6), no value and no explicit inheritance sentence | high that CAM value is absent | UNRESOLVED |
| synaptic flip candidate | 0.5 | simpleBTSP final paper | Eq. (1) discussion: update with probability 0.5; Methods explains equal LTP/LTD timing windows | medium as a candidate only | PLAUSIBLE_INHERITANCE |
| `eta` | any fixed positive global value; implementation uses 1 | mathematical derivation below | cosine scale invariance with zero initialization | high under stated assumptions | DERIVED |
| `beta` | unresolved | BTSP-CAM PDF | Sec. 4.2/Algorithm 1 gives symbol only | high | UNRESOLVED |
| `lambda_c,t` | unresolved | BTSP-CAM PDF | Eq. (4) calls it state-dependent but gives neither value nor state rule | high | UNRESOLVED |
| `tau` | unresolved | BTSP-CAM PDF | Eq. (7), Table 6 only | high | UNRESOLVED |
| `gamma` | unresolved | BTSP-CAM PDF | Eq. (8), Table 6 only | high | UNRESOLVED |
| `alpha` | ignored | BTSP-CAM PDF | only Eq. (9) plugin fusion; BTSP-only does not use it | high | DERIVED |

## Eta scale-invariance proof

Assume zero initialization, one global constant `eta>0`, and a decay sequence independent of `eta`. Repeatedly expanding Eq. (4) gives

```text
e_c,t(eta) = eta * sum_{r<=t, y_r=c} [ product_{u=r+1..t}(1-lambda_c,u) ] s_r
           = eta * e_c,t(1).
```

For any nonzero trace,

```text
cos(s, e_c,t(eta))
= (s . eta e_c,t(1)) / (||s|| * ||eta e_c,t(1)||)
= cos(s, e_c,t(1)).
```

For the zero trace, the local definition is zero for every `eta`. Therefore F, N, I, and `f_q_eff` are invariant to a global positive `eta`. This ceases to hold if traces have nonzero initial values, eta varies by class/time, decay depends on trace magnitude, or eligibility is consumed outside cosine. Under the implemented/published cosine-only, zero-initialized form, fixing `eta=1` is justified as `DERIVED`, not tuned.

## Layer policy

- **Layer A - Protocol Validation:** may use clearly recorded local/inheritance candidates and must be labeled `NOT PAPER-COMPARABLE`.
- **Layer B - Formal Reproduction:** remains blocked by `k`, `f_q_base`, `p`, `beta`, `lambda`, `tau`, and `gamma` provenance. No search or tuning against the paper's 87.82/80.15 values is permitted.
