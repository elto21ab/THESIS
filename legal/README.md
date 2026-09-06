# Legal Strategy — Donor Chat Corpus (SUBJECT + OTHER)

## Problem
Corpus per donor = SUBJECT (donor, consents) + OTHER (chat partners, unreachable) + mentioned persons. OTHER data = the legal problem. LM retrieval pipeline needs full corpus on uCloud.

## The 2 real options (DPO's own framing, corrected)

| # | Option | Legal path | Signal kept | Status |
|---|--------|-----------|-------------|--------|
| 2 | **Notice exemption** (PRIMARY) | 6(1)(e) + 9(2)(j)/§10 + 14(5)(b) + 89(1) safeguards | 100% | argue first |
| 1 | **Anonymization** (FALLBACK) | Rec. 26 "reasonably likely means" + dossier + ladder R0–R5 | 60–100% | concede rung-by-rung if pushed |

## Legal basis (one breath)
University = controller (supervisor project responsible) → Art. 6(1)(e) → Art. 9(2)(j)/DBL §10 → SUBJECT consent → OTHERs: no consent (research regime), no individual notice (14(5)(b)) → objection = ongoing erasure (Art. 21/17) → Art. 89(1) safeguards.

## DPO letter — core corrections
- "Students lack basis → anonymize-all OR consent-all" = false binary. Omitted path: university controllership → §10/9(2)(j) research regime. Consent is not the default basis for research.
- Their anonymization standard ("any detail that could identify") overbroad vs Rec. 26 + *Breyer* C-582/14 (reasonably-likely-means test).
- Their consent-all path is self-defeating: no channel, cohort collapse, selection bias, breaks timing sync → its infeasibility IS our 14(5)(b) evidence.
- Dead framings (never raise): household exemption (2(2)(c) covers donor's own collection step only), "service like OpenAI" (consent can't bind OTHERs), public donor-name list (new processing + membership leak).

## Files
- `supervisor-brief.md` — read first; bring to supervisor meeting
- `option2-notice-exemption.md` — primary strategy
- `option1-anonymization.md` — fallback + redaction ladder
- `dpo-flowchart.md` — meeting decision tree + legal map
- `donor-consent-tandc.md` — donor-facing consent form/T&C (upload site)
- `dpia-draft.md` — DPIA (Art. 35) draft — internal; supervisor sign-off, DPO consult recorded
- `clause-legal-map.md` — clause → provision reference (internal, flowchart)
