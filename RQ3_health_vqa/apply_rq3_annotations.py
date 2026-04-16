#!/usr/bin/env python3
"""Apply all visual annotations to the 34 new person samples in rq3_health_annotations.json"""
import json

RQ3_RESULTS = "/Volumes/TheDay/thedayproject/Cours Udem/IFT CV+NL/Horus Eye /RQ3_results"

# Annotations from visual inspection (sample_id: (posture, face_visible, needs_help, notes))
NEW_ANNOTATIONS = {
    2:  ("STANDING", "PARTIAL", "NO", "Little girl at hay bales, profile view"),
    3:  ("STANDING", "YES",     "NO", "Woman at hay bales holding umbrella"),
    4:  ("SITTING",  "NO",      "NO", "Mom sitting at desk with child, seen from side"),
    5:  ("SITTING",  "YES",     "NO", "Child sitting on woman's lap at desk"),
    6:  ("SITTING",  "YES",     "NO", "Man sitting at outdoor cafe"),
    7:  ("STANDING", "YES",     "NO", "Lady on right playing Wii"),
    8:  ("STANDING", "PARTIAL", "NO", "Man playing Wii bottom panel"),
    9:  ("STANDING", "YES",     "NO", "Woman in top panel playing Wii"),
    10: ("STANDING", "YES",     "NO", "Man playing Wii bottom right"),
    13: ("STANDING", "YES",     "NO", "Man standing in living room playing Wii"),
    14: ("SITTING",  "PARTIAL", "NO", "Person sitting in chair watching Wii"),
    15: ("STANDING", "YES",     "NO", "Girl standing at elephant enclosure fence"),
    16: ("STANDING", "NO",      "NO", "Woman in shorts seen from back at elephant enclosure"),
    19: ("SITTING",  "YES",     "NO", "Kid sitting at school table eating"),
    20: ("SITTING",  "YES",     "NO", "Girl in pink sitting at school table"),
    25: ("STANDING", "YES",     "NO", "Girl with elbow bent brushing teeth"),
    26: ("STANDING", "YES",     "NO", "Girl brushing teeth in mirror"),
    31: ("STANDING", "YES",     "NO", "Man in black coat at US Open tennis court"),
    32: ("STANDING", "PARTIAL", "NO", "Baseball batter wearing helmet, face partially visible"),
    33: ("STANDING", "YES",     "NO", "Man on left in blue shirt outdoors"),
    34: ("STANDING", "YES",     "NO", "Lady standing eating sandwich"),
    35: ("STANDING", "YES",     "NO", "Woman on left standing eating sandwich"),
    36: ("STANDING", "YES",     "NO", "Man on right standing eating sandwich"),
    48: ("STANDING", "YES",     "NO", "Guy in white shirt at ribbon-cutting ceremony"),
    49: ("STANDING", "NO",      "NO", "Person at bottom of crowd, only back of head visible"),
    51: ("STANDING", "PARTIAL", "NO", "Baseball batter at plate with helmet"),
    52: ("SITTING",  "YES",     "NO", "Second person from right sitting in cafe"),
    53: ("SITTING",  "YES",     "NO", "Man on right sitting in cafe"),
    55: ("SITTING",  "YES",     "NO", "Man in jersey sitting in cafe"),
    59: ("SITTING",  "YES",     "NO", "Far left guy sitting in cafe"),
    60: ("SITTING",  "YES",     "NO", "Men sitting in cafe"),
    66: ("STANDING", "YES",     "NO", "Baseball pitcher standing on mound"),
    67: ("STANDING", "PARTIAL", "NO", "Guy in back on baseball field, partially visible"),
    72: ("STANDING", "YES",     "NO", "Guy far right standing at tennis court"),
}

with open(f"{RQ3_RESULTS}/rq3_health_annotations.json") as f:
    data = json.load(f)

updated = 0
for sample in data["samples"]:
    sid = sample["sample_id"]
    if sid in NEW_ANNOTATIONS:
        posture, face_visible, needs_help, notes = NEW_ANNOTATIONS[sid]
        sample["posture"] = posture
        sample["face_visible"] = face_visible
        sample["needs_help"] = needs_help
        sample["notes"] = notes
        updated += 1
        print(f"  ID {sid:3d}: {posture} / {face_visible} / {needs_help}")

with open(f"{RQ3_RESULTS}/rq3_health_annotations.json", "w") as f:
    json.dump(data, f, indent=2)

total_annotated = sum(1 for s in data["samples"] if s.get("posture"))
print(f"\nUpdated {updated} samples. Total annotated: {total_annotated}/50")
unannotated = [s["sample_id"] for s in data["samples"] if not s.get("posture")]
if unannotated:
    print(f"Still unannotated: {unannotated}")
else:
    print("All 50 samples annotated!")
