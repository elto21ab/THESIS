# DPO Meeting Flowchart + Legal Map

## Decision tree

```
STEP 0 — SUPERVISOR MEETING (before DPO)
└─ Supervisor signs as project responsible → university = controller (Art. 4(7))
   formalization: supervisor-owned uCloud project, signed DPIA, Art. 30 record names us 2 as members

▼
STEP 1 — DPO: settle controller first (everything hangs on it)
├─ DPO accepts university controllership ──► STEP 2
└─ DPO: "thesis = students' own project"
    → ask: "what formalization satisfies institutional controllership? we adopt it verbatim"
    → fallback: supervisor's own research subproject, thesis = contribution

▼
STEP 2 — Basis: Art. 6(1)(e) + Art. 9(2)(j)/DBL §10 (consent NOT the regime for research)
├─ SUBJECT: explicit consent (6(1)(a)/9(2)(a) + Art. 7 + 13 notice)
└─ OTHER + mentioned persons: no consent — research basis covers
    │
    ▼
STEP 3 — Notice: Art. 14(5)(b) exemption (ONE combined claim)
    effort: no channel (no API), n×10²–10³ contacts, donor-burden → cohort collapse (documented friction)
    impairment: ex-ante notice/objection breaks corpus↔survey timing sync; selection bias
    impact ≈ 0: OTHERs not analyzed, no quotes/profiling, corpus destroyed
    objection: Art. 21(6)/17 ongoing → erasure, NO pre-processing window
    substitute: public page + registry + opt-out mailbox (voluntary goodwill, not required)
    │
    ├─ DPO accepts ──► PATH A (PREFERRED): full-fidelity corpus, header-mask only
    │   safeguards (89(1)/32): header-mask SUBJECT/OTHER, pseudonymized storage,
    │   3 named access, encryption, DPIA (35 + 35(2) consult), destruction schedule,
    │   research-only commitment (§10 "solely scientific study")
    │
    └─ DPO rejects ──► "which safeguard makes 14(5)(b) sufficient?"
        ▼
        CONCESSION LADDER (option1, gentlest-first, cite fidelity cost each):
        R1 consistent pseudonyms → R2 chunk-varying → R3 decoy injection (RR deniability)
        → R4 full-msg obfuscation → R5 distilled context labels
        ▼
        Still refused → written reasoning → escalate; anonymization dossier route (Rec. 26
        "reasonably likely means" + Breyer vs their absolutist standard; motivated-intruder test)
```

## Legal map

| Provision | Role |
|---|---|
| Art. 4(7) | controller = university via supervisor |
| Art. 6(1)(e) | general basis: public-interest research |
| Art. 9(2)(j) + DBL §10 | special-category processing for research, no consent; §10(1) "solely scientific study" → research-only commitment |
| Art. 6(1)(a), 7, 9(2)(a), 13 | SUBJECT consent + notice |
| Art. 14(5)(b) | OTHER notice exemption (impossible / disproportionate effort / impairs objectives — "in particular for research") |
| Art. 14(5)(a) | supplementary: donors who already told contacts (log only) |
| Art. 21(6), 17 | ongoing objection → erasure; no pre-window |
| Art. 89(1), 5(1)(c), 25, 32 | safeguards: header-mask, pseudonymization, access control, minimization |
| Art. 30, 35(2), 28 | record, DPIA + DPO consult, uCloud DPA |
| Rec. 26 + Breyer C-582/14 | identifiability = "means reasonably likely to be used" (fallback dossier) |
| WP29 3 tests | singling-out / linkability / inference (fallback dossier) |
| ~~Art. 2(2)(c)~~, ~~OpenAI-ToS~~, ~~donor-name list~~ | dead framings — never raise |

## Narrative order
1. Controller (supervisor formalized) → 2. Basis (6(1)(e) + 9(2)(j)/§10) → 3. 14(5)(b) package → 4. Safeguards offered generously → 5. Content fidelity = red line → 6. Ladder only if pushed → 7. Written asks: basis confirmed, 14(5)(b) accepted, DPIA review timeline
