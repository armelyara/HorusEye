# HorusEye: Language as Dynamic Attention for Emergency Visual Analysis

[![arXiv](https://img.shields.io/badge/arXiv-2606.14741-b31b1b.svg)](https://arxiv.org/abs/2606.14741)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/armelyara/refcoco-degraded)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

> **Can natural language feedback serve as a dynamic attention mechanism to refine visual tasks under degraded emergency conditions?**

We investigate this question across four research questions, evaluating five Vision-Language Models (Gemini, Qwen2-VL, BLIP-2, LLaVA, Kosmos-2) on visual grounding, language feedback recovery, health VQA, and hallucination analysis under fog, smoke, and thermal degradation.

**Paper**: [arXiv:2606.14741](https://arxiv.org/abs/2606.14741) — Armel Yara, IFT6765, Mila / Université de Montréal

📎 **Slides**: [Google Slides Presentation](https://docs.google.com/presentation/d/19xKly7EtyxvV5UXuAUsE_jN86wQEsfsp/edit?usp=sharing&ouid=106043540914542772736&rtpof=true&sd=true)

<p align="center">
  <img src="assets/poster_horuseye.jpg" alt="HorusEye Poster — IFT6765, Mila / Université de Montréal" width="90%"/>
</p>

---

## Key Findings

- **Language feedback is model-dependent**: Gemini achieves **+47.3%** IoU recovery under thermal via iterative language feedback; Qwen2-VL shows **-5.1%** degradation under the same protocol
- **Thermal Paradox**: Cropping strategies that improve RGB performance catastrophically fail in thermal imagery (-26% accuracy)
- **BLIP-2 is unsafe for emergency deployment**: It is the only model whose hallucination score *increases* under degradation — fabricating colors in grayscale images

---

## RefCOCO-Degraded Benchmark

| Property | Value |
|----------|-------|
| Base images | 3,811 (RefCOCO val split) |
| Conditions | Clean, Fog, Smoke, Thermal |
| Severities | 0.25, 0.5, 0.75, 1.0 |
| Total samples | 15,244 (at severity 0.5) / 49,543 (all severities) |

**Degradation methods:**
- **Fog** — Koschmieder atmospheric scattering model
- **Smoke** — Perlin noise occlusion with variable opacity
- **Thermal** — Grayscale conversion with INFERNO colormap + sensor noise

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/armelyara/refcoco-degraded).

---

## Repository Structure

```
Horus-Eye/
├── RQ1_visual_grounding/             # Stage 1 & 2: Grounding evaluation
│   ├── degradation_pipeline.py       # Generate degraded images from RefCOCO
│   ├── visual_grounding_gemini.py    # Gemini bbox prediction
│   ├── visual_grouding_Qwen2.py      # Qwen2-VL bbox prediction
│   ├── visual_grounding_Kosmos-2.py  # Kosmos-2 bbox prediction
│   ├── evaluate_grounding.py         # IoU computation & analysis
│   ├── generate_severity_study.py    # Severity curve generation
│   └── horus_bench.py                # Benchmark runner
│
├── RQ2_language_feedback/            # Stage 3: Iterative language feedback
│   ├── language_feedback_gemini.py   # Gemini feedback loop
│   └── language_feedback_qwen2.py    # Qwen2-VL feedback loop
│
├── RQ3_health_vqa/                   # Stage 4: Health VQA posture classification
│   ├── build_rq3_person_only.py      # Filter person-only annotations
│   ├── apply_rq3_annotations.py      # Apply posture annotations
│   ├── rq3_health_annotations.json   # Ground truth posture labels
│   ├── health_vqa_blip2.py           # BLIP-2 posture classification
│   ├── health_vqa_llava/             # LLaVA posture classification
│   ├── health_vqa_qwen.py            # Qwen2-VL posture classification
│   └── health_vqa_gemini.py          # Gemini posture classification
│
├── RQ4_hallucination/                # Stage 5: Hallucination detection
│   ├── hallucination_blip2.py        # BLIP-2 hallucination analysis
│   ├── hallucination_llava.py        # LLaVA hallucination analysis
│   └── hallucination_gemini_qwen.py  # Gemini & Qwen2-VL analysis
│
├── refcoco_degraded_benchmark/       # Dataset annotations & results
│   ├── annotations/                  # Annotations JSON files
│   └── results/                      # Experiment result checkpoints
│
├── results/                          # Aggregated grounding results
├── download_refcoco.sh               # RefCOCO download helper
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
# Clone the repository
git clone -b refcoco-degradation-pipeline https://github.com/armelyara/Horus-Eye.git
cd Horus-Eye

# Install dependencies
pip install -r requirements.txt
```

**Additional model-specific dependencies:**

```bash
# For BLIP-2 and LLaVA
pip install transformers torch accelerate

# For Qwen2-VL
pip install transformers qwen-vl-utils

# For Gemini
pip install google-generativeai

# For Kosmos-2
pip install transformers
```

### Step 1: Download RefCOCO

Download the RefCOCO dataset from [lichengunc/refer](https://github.com/lichengunc/refer) and COCO images from [cocodataset.org](https://cocodataset.org/#download):

```bash
# Place files in this structure:
datasets/
├── refcoco/
│   ├── refs(unc).p
│   └── instances.json
└── coco/
    └── train2014/
        ├── COCO_train2014_000000000009.jpg
        └── ...
```

### Step 2: Generate Degraded Images

```bash
cd RQ1_visual_grounding
python degradation_pipeline.py \
    --refcoco_path ../datasets/refcoco \
    --coco_images ../datasets/coco/train2014 \
    --output_dir ../refcoco_degraded_benchmark/images \
    --severities 0.25 0.5 0.75 1.0
```

### Step 3: Run Evaluations

**RQ1 — Visual Grounding:**
```bash
# Gemini
python visual_grounding_gemini.py --api_key YOUR_KEY

# Qwen2-VL
python visual_grouding_Qwen2.py

# Kosmos-2
python visual_grounding_Kosmos-2.py

# Compute IoU scores
python evaluate_grounding.py
```

**RQ2 — Language Feedback:**
```bash
python ../RQ2_language_feedback/language_feedback_gemini.py
python ../RQ2_language_feedback/language_feedback_qwen2.py
```

**RQ3 — Health VQA:**
```bash
cd ../RQ3_health_vqa
python health_vqa_blip2.py
python health_vqa_gemini.py
python health_vqa_qwen.py
```

**RQ4 — Hallucination Detection:**
```bash
cd ../RQ4_hallucination
python hallucination_blip2.py
python hallucination_llava.py
python hallucination_gemini_qwen.py
```

---

## Models Evaluated

| Model | Encoder | Bbox Output | Used In |
|-------|---------|-------------|---------|
| Gemini 2.0 Flash | Proprietary ViT | Text (regex parsed) | RQ1–RQ4 |
| Qwen2-VL-7B | ViT + NaViT | `<box>` tokens | RQ1–RQ4 |
| Kosmos-2 | ViT | `<loc_XXX>` tokens | RQ1–RQ2 |
| BLIP-2 (FlanT5-XL) | Frozen ViT-G + Q-Former | Text only | RQ3–RQ4 |
| LLaVA-1.5-7B | CLIP ViT-L/14 + MLP | Text only | RQ3–RQ4 |

---

## Results Summary

### RQ1: Visual Grounding (Mean IoU)

| Model | Clean | Fog | Smoke | Thermal | Δ |
|-------|-------|-----|-------|---------|---|
| Gemini | 0.64 | 0.57 | 0.57 | 0.40 | -38% |
| Qwen2-VL | 0.59 | 0.58 | 0.55 | 0.21 | -64% |

### RQ2: Language Feedback Recovery (Thermal)

| Model | Before Feedback | After Feedback | Change |
|-------|----------------|----------------|--------|
| Gemini | 0.40 | 0.59 | **+47.3%** |
| Qwen2-VL | 0.21 | 0.20 | **-5.1%** |

### RQ4: Hallucination (H-Score, lower = safer)

| Model | Clean | Thermal | Trend |
|-------|-------|---------|-------|
| Gemini | 1.2 | 0.8 | ↓ Safer |
| BLIP-2 | 2.1 | 3.5 | ↑ **Dangerous** |

---

## H-Score Formula

```
H-Score = fabricated_objects + (overconfidence × 0.5) − (uncertainty × 0.3)
```

Higher H-Score = more hallucination. A model that expresses uncertainty is penalized less than one that fabricates confidently.

---

## Citation

```bibtex
@article{yara2026horuseye,
  title={HorusEye: Language as Dynamic Attention for Emergency Visual Analysis},
  author={Yara, Armel},
  journal={arXiv preprint arXiv:2606.14741},
  year={2026}
}
```

---

## Acknowledgments

This work was conducted as part of course IFT6765 at Mila / Université de Montréal. The RefCOCO-Degraded benchmark builds upon RefCOCO (Yu et al., 2016) and MS COCO (Lin et al., 2014).

---

## License

Code: [MIT License](LICENSE)
Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
