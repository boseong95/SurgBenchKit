# CholecT50-VQA: Full Prompt Specification

## System Prompt

```
You are a surgical assistant monitoring a laparoscopic cholecystectomy in real time.

PROCEDURE PHASES (in order):
  [1] Preparation
  [2] Calot Triangle Dissection
  [3] Clipping & Cutting
  [4] Gallbladder Dissection
  [5] Gallbladder Packaging
  [6] Cleaning & Coagulation
  [7] Gallbladder Retraction

CRITICAL VIEW OF SAFETY (CVS):
  During 'Calot Triangle Dissection', the surgeon must achieve the Critical View
  of Safety before proceeding to 'Clipping & Cutting'. CVS requires:
    - two_structures: cystic duct and cystic artery clearly identified (0/1/2)
    - cystic_plate: lower third of cystic plate visible (0/1/2)
    - hepatocystic_triangle: cleared of fat and fibrous tissue (0/1/2)
  CVS score = sum of three criteria (0–6). Score >= 4 is adequate.

QUALITY ASSESSMENT CRITERIA:
  For each phase, evaluate:
    - efficiency: Are the correct instruments being used? Is progress being made?
    - safety: Are critical structures being respected? Any signs of uncontrolled bleeding or thermal injury?
    - completeness: Are the phase objectives being met before transitioning?

FAILURE MODES:
  Common failure patterns in cholecystectomy include:
    - instrument_idle: instrument present but not engaging target (null_verb/null_target)
    - wrong_instrument: instrument not typical for the current phase
    - repeated_action: same action repeated without progress (struggling)
    - missed_structure: critical structure not addressed (e.g., cystic artery not clipped)
    - premature_transition: moving to next phase before current phase objectives are met
    - bleeding: uncontrolled bleeding requiring coagulation or irrigation

You will be shown consecutive frames (1 fps) from the procedure.
Analyze the frames and respond in JSON with this exact format:

{
  "phase": "Calot Triangle Dissection",
  "current_triplet": [
    {"instrument": "hook", "verb": "dissect", "target": "cystic_duct"},
    {"instrument": "grasper", "verb": "retract", "target": "gallbladder"}
  ],
  "next_phase": "Clipping & Cutting",
  "quality": {
    "score": "adequate",
    "reason": "Correct instruments in use. Cystic duct being properly dissected."
  },
  "failed": false,
  "failure_reason": null,
  "next_triplet": [
    {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"}
  ],
  "recovery_triplet": null
}

VOCABULARY:
  Instruments: grasper, bipolar, hook, scissors, clipper, irrigator
  Verbs: grasp, retract, dissect, coagulate, clip, cut, aspirate, irrigate, pack
  Targets: gallbladder, cystic_plate, cystic_duct, cystic_artery, cystic_pedicle,
           blood_vessel, fluid, abdominal_wall_cavity, liver, adhesion, omentum,
           peritoneum, gut, specimen_bag

OUTPUT RULES:
  - phase: one of the 7 phases listed above (current phase)
  - current_triplet: the action(s) currently being performed, using the
    vocabulary below. This is your reasoning step — identify what instruments
    are visible and what they are doing before predicting next actions.
  - next_phase: the phase that will follow the current one. Usually the next
    phase in sequence, but may repeat a previous phase (e.g., return to
    "Cleaning & Coagulation" after "Gallbladder Packaging") or be
    "End of Procedure" if this is the final phase.
  - quality.score: "adequate" | "marginal" | "inadequate"
  - quality.reason: brief explanation (1 sentence)
  - failed: true if a failure mode is detected, false otherwise
  - failure_reason: one of [instrument_idle, wrong_instrument, repeated_action,
    missed_structure, premature_transition, bleeding] or null
  - next_triplet: the next action(s) the surgeon should perform
  - recovery_triplet: if failed=true, the corrective action(s); null if failed=false
```

---

## User Prompt

```
[IMAGE: frame_001.png, ..., frame_005.png]

Analyze the current state of the procedure.
Report the phase, status, quality assessment, any failures, and recommended next actions.
```

---

## Expected Answer Examples

### Example 1: Normal in-progress (adequate quality)

Scenario: Mid Calot Triangle Dissection, grasper retracting gallbladder while hook dissects.

```json
{
  "phase": "Calot Triangle Dissection",
  "next_phase": "Clipping & Cutting",
  "quality": {
    "score": "adequate",
    "reason": "Hook actively dissecting with grasper providing good retraction exposure."
  },
  "failed": false,
  "failure_reason": null,
  "next_triplet": [
    {"instrument": "hook", "verb": "dissect", "target": "cystic_duct"}
  ],
  "recovery_triplet": null
}
```

### Example 2: Phase transition

Scenario: End of Clipping & Cutting, both cystic duct and artery have been clipped and cut.

```json
{
  "phase": "Clipping & Cutting",
  "next_phase": "Gallbladder Dissection",
  "quality": {
    "score": "adequate",
    "reason": "Cystic duct and artery clipped and divided. Safe to proceed."
  },
  "failed": false,
  "failure_reason": null,
  "next_triplet": [
    {"instrument": "hook", "verb": "dissect", "target": "gallbladder"}
  ],
  "recovery_triplet": null
}
```

### Example 3: Failure — instrument idle

Scenario: During Gallbladder Packaging, grasper is present but not engaging the specimen bag.

```json
{
  "phase": "Gallbladder Packaging",
  "next_phase": "Cleaning & Coagulation",
  "quality": {
    "score": "marginal",
    "reason": "Grasper is idle and not engaging the specimen bag."
  },
  "failed": true,
  "failure_reason": "instrument_idle",
  "next_triplet": [
    {"instrument": "grasper", "verb": "grasp", "target": "specimen_bag"}
  ],
  "recovery_triplet": [
    {"instrument": "grasper", "verb": "grasp", "target": "specimen_bag"}
  ]
}
```

### Example 4: Failure — wrong instrument for phase

Scenario: During Clipping & Cutting phase, but only grasper and hook are visible — no clipper present.

```json
{
  "phase": "Clipping & Cutting",
  "next_phase": "Gallbladder Dissection",
  "quality": {
    "score": "inadequate",
    "reason": "Clipper not present during Clipping & Cutting. Cannot clip structures safely."
  },
  "failed": true,
  "failure_reason": "wrong_instrument",
  "next_triplet": [
    {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"}
  ],
  "recovery_triplet": [
    {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"}
  ]
}
```

### Example 5: Failure — bleeding requires recovery

Scenario: During Gallbladder Dissection, bleeding detected from blood vessel, needs coagulation.

```json
{
  "phase": "Gallbladder Dissection",
  "next_phase": "Gallbladder Packaging",
  "quality": {
    "score": "inadequate",
    "reason": "Active bleeding from blood vessel during dissection."
  },
  "failed": true,
  "failure_reason": "bleeding",
  "next_triplet": [
    {"instrument": "bipolar", "verb": "coagulate", "target": "blood_vessel"}
  ],
  "recovery_triplet": [
    {"instrument": "bipolar", "verb": "coagulate", "target": "blood_vessel"},
    {"instrument": "irrigator", "verb": "aspirate", "target": "fluid"}
  ]
}
```

### Example 6: Marginal quality — premature transition

Scenario: End of Calot Triangle Dissection, but CVS criteria not fully met.

```json
{
  "phase": "Calot Triangle Dissection",
  "next_phase": "Clipping & Cutting",
  "quality": {
    "score": "marginal",
    "reason": "Hepatocystic triangle not fully cleared. CVS may not be achieved."
  },
  "failed": true,
  "failure_reason": "premature_transition",
  "next_triplet": [
    {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"}
  ],
  "recovery_triplet": [
    {"instrument": "hook", "verb": "dissect", "target": "cystic_plate"},
    {"instrument": "grasper", "verb": "retract", "target": "gallbladder"}
  ]
}
```

---

## Evaluation

| Field              | Metric                                                           |
|--------------------|------------------------------------------------------------------|
| `phase`            | Accuracy (exact match, 7 classes)                                |
| `next_phase`       | Accuracy (exact match, 7 classes + "End of Procedure")           |
| `quality.score`    | Accuracy (exact match, 3 classes)                                |
| `failed`           | Accuracy / F1 (binary)                                           |
| `failure_reason`   | Accuracy (exact match among 6 failure types + null)              |
| `next_triplet`     | Per-component: instrument acc, verb acc, target acc; triplet F1  |
| `recovery_triplet` | Per-component: instrument acc, verb acc, target acc; triplet F1  |

---

## Ground Truth Derivation

| Field              | Source                                                            |
|--------------------|-------------------------------------------------------------------|
| `phase`            | CholecT50 phase label (field [14] in annotation vector)           |
| `next_phase`       | Phase label of the next distinct phase segment in the video; "End of Procedure" if final phase |
| `quality.score`    | Heuristic from: CVS scores (Cholec80-CVS), idle instrument ratio, instrument-phase match |
| `failed`           | Derived from: null_verb/null_target patterns, instrument-phase mismatch, CVS < 4 at transition |
| `failure_reason`   | Pattern-matched from triplet annotations (see failure modes above) |
| `next_triplet`     | Look-ahead: first new triplets in next 10 frames                 |
| `recovery_triplet` | Rule-based: depends on failure_reason (e.g., bleeding → coagulate + aspirate) |

### Available ground truth signals

- **CholecT50**: triplet annotations (instrument, verb, target) per frame, phase labels
- **Cholec80-CVS**: Critical View of Safety scores (two_structures, cystic_plate, hepatocystic_triangle) with timestamps — available for all 80 Cholec80 videos (572 CVS assessments)
- **Idle instrument detection**: frames with `null_verb` or `null_target` in triplet annotations (~10K frames across dataset)
- **Phase duration statistics**: can flag outlier-duration phases as potential quality issues
