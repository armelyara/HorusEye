#!/usr/bin/env python3
import json, re, os

BASE = "/Volumes/TheDay/thedayproject/Cours Udem/IFT CV+NL/Horus Eye /Horus dev/datasets/refcoco_degraded_benchmark"
RQ3_RESULTS = "/Volumes/TheDay/thedayproject/Cours Udem/IFT CV+NL/Horus Eye /RQ3_results"

PERSON_KEYWORDS = [
    "man", "woman", "girl", "boy", "guy", "lady", "person", "kid", "child",
    "dude", "gurl", "mom", "dad", "father", "mother", "player", "rider",
    "hitter", "batter", "pitcher", "skater", "surfer", "skier", "driver",
    "passenger", "soldier", "officer", "chef", "cook", "waiter", "nurse",
    "doctor", "athlete", "runner", "biker", "cyclist", "worker", "teen",
    "baby", "infant", "toddler", "elderly", "people", "couple",
    "sister", "brother", "friend", "trainer", "coach", "referee",
    "human", "adult", "male", "female", "u2", "groom", "bride", "family",
    "crowd", "spectator", "fan", "student", "teacher", "professor",
]

def is_person_expression(expression):
    expr_lower = expression.lower()
    for kw in PERSON_KEYWORDS:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, expr_lower):
            return True
    return False

print("Loading evaluation_prompts.json...")
with open(f"{BASE}/annotations/evaluation_prompts.json") as f:
    prompts = json.load(f)
print(f"  {len(prompts)} samples")

print("Loading gemini_results.json...")
with open(f"{BASE}/results/gemini_results.json") as f:
    gemini = json.load(f)
clean_ious = gemini["raw_ious"]["clean"]

print("Loading rq3_annotation_template.json (benchmark copy with rq2 bbox data)...")
with open(f"{BASE}/rq3_annotation_template.json") as f:
    template = json.load(f)
template_by_id = {s["sample_id"]: s for s in template["samples"]}

print("Loading current rq3_health_annotations.json...")
with open(f"{RQ3_RESULTS}/rq3_health_annotations.json") as f:
    current = json.load(f)
current_by_id = {s["sample_id"]: s for s in current["samples"]}

# Find person-referent samples
person_samples = [p for p in prompts if is_person_expression(p["expression"])]
print(f"\nPerson samples found: {len(person_samples)}")

# IDs of already-annotated person samples
existing_person_ids = {sid for sid, s in current_by_id.items() if s.get("posture", "")}
print(f"Already annotated: {sorted(existing_person_ids)}")

# Pick 32 more from remaining pool, sorted by id
remaining = sorted([p for p in person_samples if p["id"] not in existing_person_ids], key=lambda p: p["id"])
need = 50 - len(existing_person_ids)
new_32 = remaining[:need]
new_32_ids = [p["id"] for p in new_32]
print(f"New 32 IDs: {new_32_ids}")

all_50_ids = sorted(list(existing_person_ids) + new_32_ids)
print(f"All 50 IDs: {all_50_ids}")

new_samples = []
for sid in all_50_ids:
    prompt_data = prompts[sid]
    assert prompt_data["id"] == sid

    if sid in template_by_id:
        t = template_by_id[sid]
        rq2_initial_bbox = t["rq2_initial_bbox"]
        rq2_final_bbox = t["rq2_final_bbox"]
        rq2_initial_iou = t["rq2_initial_iou"]
        rq2_final_iou = t["rq2_final_iou"]
    else:
        rq2_initial_bbox = prompt_data["ground_truth_bbox"]
        rq2_final_bbox = prompt_data["ground_truth_bbox"]
        rq2_initial_iou = clean_ious[sid]
        rq2_final_iou = clean_ious[sid]

    c = current_by_id.get(sid, {})
    new_samples.append({
        "sample_id": sid,
        "filename": prompt_data["filename"],
        "expression": prompt_data["expression"],
        "gt_bbox": prompt_data["ground_truth_bbox"],
        "rq2_initial_bbox": rq2_initial_bbox,
        "rq2_final_bbox": rq2_final_bbox,
        "rq2_initial_iou": rq2_initial_iou,
        "rq2_final_iou": rq2_final_iou,
        "posture": c.get("posture", ""),
        "face_visible": c.get("face_visible", ""),
        "needs_help": c.get("needs_help", ""),
        "notes": c.get("notes", ""),
    })

output = {
    "metadata": {
        "description": "RQ3 Health Annotation - Manual Ground Truth",
        "annotator": "VL Engineer",
        "date": "2026-03-09",
        "num_samples": 50,
        "guidelines": {
            "posture": {"STANDING": "Person is upright on their feet", "SITTING": "Person is seated", "LYING": "Person is horizontal (critical for emergency)"},
            "face_visible": {"YES": "Face clearly visible", "PARTIAL": "Face partially visible", "NO": "Face not visible"},
            "needs_help": {"YES": "Person appears to need assistance", "NO": "Person appears fine", "UNCLEAR": "Cannot determine"}
        }
    },
    "samples": new_samples
}

out_path = f"{RQ3_RESULTS}/rq3_health_annotations.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nWrote {len(new_samples)} samples to {out_path}")
print(f"Annotated: {sum(1 for s in new_samples if s['posture'])}")
print(f"Need annotation:")
for s in new_samples:
    if not s["posture"]:
        print(f"  ID {s['sample_id']:4d}  {s['filename']}  '{s['expression']}'")
