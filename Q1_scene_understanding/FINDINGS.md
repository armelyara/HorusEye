# Q1 — Scene Understanding under Degraded Visibility: Calibration Findings

*HorusEye system function **Q1**, level 1 (emergency-type classification with calibrated uncertainty). Model: Qwen2.5-VL-7B. Status: benchmarked, findings frozen.*

> Note on naming: this is HorusEye **function Q1**, not the published paper's research questions (`RQ1–RQ4`), which all belong to function **Q2** (arXiv:2606.14741).

## 1. Protocol (locked)

Q1 measures **emergency-type classification under degraded visibility** with calibrated uncertainty. A class is defined by its **content** (fire = flames + burning structure; flood = water mass; collapsed_building = debris; traffic_accident = impacted vehicles; normal = no emergency). **Fog and smoke are visibility veils, not classes.** The simulated "thermal" (MAGMA colormap) is abandoned — it conflated simulated fire with a veil and its results were an artifact. Real thermal (sensor, e.g. VTSaR) remains valid for level 2.

- **Data:** real AIDER images (GNU GPL v3.0), 5 classes, strong imbalance → **F1 macro**.
- **Veils applied only where physically plausible** (degradation pipeline reused verbatim from Q2 — `add_fog` Koschmieder, `add_smoke` Perlin):
  - smoke → fire, traffic_accident, normal
  - fog → flood, collapsed_building, normal
  - `normal` receives both veils (false-positive test).
- **Confidence** is self-reported by the VLM and quantized on {0.7, 0.8, 0.9, 0.95, 1.0} — this measures the model's *metacognition*, not softmax calibration.

## 2. Sample composition (matters for interpretation)

550 predictions, split **250 / 150 / 150**, over **different class sets** per condition:

| Condition | n | Classes present |
|---|---|---|
| clean | 250 | fire, flood, collapsed_building, traffic_accident, normal (50 each) |
| fog | 150 | flood, collapsed_building, normal (50 each) |
| smoke | 150 | fire, traffic_accident, normal (50 each) |

Because the class sets differ, **raw cross-condition comparisons mix the veil effect with a class-composition effect.** `normal` is the only class present in all three conditions → it is the matched comparison axis (§4).

## 3. Results by condition

| Condition | n | Accuracy | F1 macro | ECE | AUROC |
|---|---|---|---|---|---|
| clean | 250 | 0.976 | 0.976 | 0.046 | 0.796 |
| fog | 150 | 0.947 | 0.956 | 0.046 | 0.866 |
| smoke | 150 | 0.859 | 0.886 | 0.068 | 0.818 |
| **global** | **550** | **≈0.936** | — | **0.042** | **0.847** |

![Reliability by condition](results/q1_reliability_by_condition_EN.png)

## 4. Headline finding — the veil flips the *sign* of miscalibration

Aggregate ECE (an absolute value) hides the key result. The **signed** gap (mean confidence − accuracy) reveals it:

| Condition | Signed gap — all plausible classes | Signed gap — `normal` only (matched) |
|---|:---:|:---:|
| clean | **−0.046** (under-confident) | −0.004 |
| fog | **−0.022** (near-calibrated) | +0.044 |
| smoke | **+0.042** (over-confident) | **+0.173** |

![Signed calibration gap](results/q1_signed_gap_EN.png)

On clean and under fog the model is *more* accurate than it claims (under-confident). Under smoke it flips: confidence stays high while accuracy collapses.

**Composition control — the finding holds and sharpens.** Recomputed at constant composition:
- On the `normal` class (same 50 images per condition): **−0.004 → +0.044 → +0.173**. Under smoke, 36/50 correct (acc 0.72) but mean confidence 0.893 → gap **+0.173**. Monotonic and far stronger than the aggregate.
- On classes common to clean↔smoke (fire, accident, normal): **−0.035 → +0.042** — sign flip holds.
- On classes common to clean↔fog (flood, collapse, normal): **−0.042 → −0.022** — no flip, consistent with "fog does not flip."

**Conclusion:** the over-confidence-under-veil hypothesis is **confirmed for smoke, not for fog, and robust to the composition control.** Cite the `normal`-only or common-class (matched) version, not the raw aggregate alone.

## 5. The danger zone — confidence bin 0.8 under smoke

Decomposed by confidence bin:
- Confidence **0.95** → ~99% accurate in **all** conditions. Reliable.
- Confidence **0.8 under smoke** → **56%** accurate (25 cases). The model's "moderate confidence" is nearly worthless.

**Proposed operating rule:** under degraded visibility, a prediction ≤ 0.8 confidence triggers human review; ≥ 0.95 may be treated as reliable.

## 6. Emergency false alarms on `normal` scenes

| Condition | Recall `normal` | Correct / 50 | False alarms |
|---|---|---|---|
| clean | 0.92 | 46/50 | 8% |
| fog | 0.86 | 43/50 | 14% |
| smoke | 0.72 | 36/50 | 28% |

Under smoke, more than one normal scene in four triggers a false emergency — 7/50 classified as `flood`, an outright hallucination. This is the operational cost of over-confidence, and the same subset that yields the +0.173 signed gap (§4).

## 7. Methodological caveats

- **Confidence is self-reported and quantized** — this is metacognition, not logit calibration. Name it explicitly.
- **Cross-condition comparison is confounded** by class composition → addressed by the matched analysis (§4). Do not cite raw cross-condition gaps without this control.
- **Per-condition AUROC is unstable:** ~7 errors (clean), ~8 (fog), ~21 (smoke). The clean AUROC of 0.796 (below fog) is small-sample noise. Rely on the global AUROC (0.847) and on smoke.

## 8. Reproduce

```bash
python q1_calibration.py results/q1_all_records.json
```

Prints per-condition accuracy, F1 macro, ECE, AUROC and the signed gap (global + `normal`-matched), and writes both figures. ECE / AUROC / F1 are implemented from scratch — no sklearn.

## 9. Next

Q1 level 2 — danger localization + victim detection/counting/localization + calibrated uncertainty (real thermal / VTSaR relevant here).

## Files

`results/q1_all_records.json` (550 records), `q1_summary.json`, `q1_reliability_by_condition_EN.png`, `q1_signed_gap_EN.png`. Run script: `q1_classification_kaggle.py`. Analysis script: `q1_calibration.py`.
