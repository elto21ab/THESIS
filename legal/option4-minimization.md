# Option 4: Minimization-by-Design — FOLDED INTO OPTION 2

Kept as reference. Now treated as the aggressiveness dial inside Option 2: 2.1 top-down PII mask → 2.2 bottom-up full-msg obfuscation → distilled context. See option2.md. Below = original ladder rationale.

## Idea (original)
OTHER's data is 100% of the legal problem. Shrink it before arguing exemptions. Art. 25 data protection by design; strengthens every other option.

## Ladder of OTHER-message treatments (keep the most that passes)
1. **Full OTHER msgs** (status quo) — maximal Art. 14(5)(b) fight
2. **Context window truncation**: keep OTHER msg only when within k msgs / t hours of a SUBJECT msg actually retrieved → corpus of *contextualized pairs*, not full histories
3. **Donor-side triage**: OTHER msgs flagged Art. 9-adjacent → redact or escalate to direct-notice subset (makes Option 3 fallback #3 feasible)
4. **Lossy context distillation**: replace OTHER msgs w/ structured summary {topic labels, sentiment, relation-type, timestamp} — LM gets conversational context w/o personal text. Signal loss: TBD empirically (pilot)
5. **Synthetic placeholders**: neighbor-model-generated filler — last resort

## Why include
- Converts binary "keep everything vs omit everything" into negotiable dial → DPO sees good-faith minimization (Art. 5(1)(c), 25)
- Cheap to pilot: measure imitation fidelity at each rung → bring curve to DPO: "signal vs exposure" tradeoff quantified

## Talking point
"We don't ask to keep everything; we ask to keep the minimal level where the science still works — and we've measured where that is."
