# DPO Meeting Flowchart + Legal Basis Map

## Decision tree (meeting flow)

```
START: Who is controller?
│
├─ University (supervisor signs as project responsible, u2 = authorized members)
│   │
│   ▼
│   Art. 6(1)(e) public-interest research + §10 DBL / Art. 9(2)(j)
│   │
│   ▼
│   SUBJECT: explicit informed consent (donation + surveys) ── Art. 6(1)(a), 9(2)(a), Art. 7
│   │
│   ▼
│   OTHER (+ nth-order mentioned persons): no consent feasible
│   │   Q: individual notice required? (Art. 14)
│   │   │
│   │   ├─ DPO accepts Art. 14(5)(b): disproportionate effort / seriously impairs objectives
│   │   │   evidence: n×10²–10³ contacts, donor-only channel, cohort collapse (documented friction)
│   │   │   └─► PATH A (PREFERRED): full-fidelity corpus, no content edits
│   │   │       safeguards (Art. 89(1), 32): header-mask SUBJECT/OTHER, pseudonymized storage,
│   │   │       access control (3 named), encryption, DPIA (Art. 35), destruction schedule,
│   │   │       opt-out mailbox + erasure-on-objection (Art. 21 ongoing, Art. 17)
│   │   │       goodwill extras: public notice page + uni registry entry (NOT required, offer)
│   │   │
│   │   └─ DPO rejects 14(5)(b)
│   │       ▼
│   │       CONCESSION LADDER (rung-by-rung, each w/ fidelity-cost argument):
│   │       R1: consistent pseudonyms for OTHER identifiers (keep coreference)
│   │       R2: chunk-varying pseudonyms (kill cross-corpus linkability)
│   │       R3: decoy injection on PII-model-flagged msgs (RR plausible deniability)
│   │       R4: full-msg obfuscation of flagged msgs (2.2 bottom-up)
│   │       R5: distilled context labels only (last resort)
│   │       │
│   │       ▼
│   │       Still refused? → demand written reasoning; escalate (faculty/univ. research integrity);
│   │       or re-scope as supervisor's own subproject w/ thesis as contribution
│
└─ DPO insists students = independent controllers ──► REJECT framing:
    thesis under uni supervision + uni infrastructure + supervisor accountability
    = processing "som led i" research activity (§10). Ask: what formalization would satisfy?
    (supervisor-owned uCloud project, signed DPIA, Art. 30 record naming u as members)
```

## Legal map per option

| Ref | Provision | Role |
|---|---|---|
| **A. Basis (all paths under uni controllership)** | | |
| | GDPR Art. 4(7) | controller = university via supervisor |
| | GDPR Art. 6(1)(e) | lawful basis: public-interest research |
| | GDPR Art. 9(2)(j) + DBL §10(1) | special-category processing in research |
| | GDPR Art. 89(1) | safeguards condition for research derogations |
| **B. SUBJECT** | | |
| | GDPR Art. 6(1)(a), 7 + 9(2)(a) | explicit consent: donation, surveys, Art. 9 content |
| | GDPR Art. 13 | notice to donor (info duty fulfilled directly) |
| **C. OTHER (no consent)** | | |
| | GDPR Art. 14(5)(b) | notice exemption: disproportionate effort / impairs objectives |
| | GDPR Art. 21(6) + 17 | ongoing objection → erasure procedure (no pre-window) |
| | GDPR Art. 5(1)(c), 25 | minimization + by-design (header-masking, access limits) |
| **D. Anonymization fallback (ladder R3+)** | | |
| | Recital 26 + C-582/14 Breyer | identifiability = "means reasonably likely to be used" |
| | WP29/EDPB 3 tests | singling-out, linkability, inference |
| | DPIA annex | motivated-intruder adversary model + residual risk |
| **E. Governance artifacts** | | |
| | GDPR Art. 30 | record of processing (names u2 as members) |
| | GDPR Art. 35 (+35(2)) | DPIA + DPO consultation |
| | GDPR Art. 32 | security: uCloud, encryption, access control |
| | GDPR Art. 28 | uCloud DPA (processor) |
| **F. Dead ends (do not raise)** | | |
| | Art. 2(2)(c) household | only donor-side collection step, never our processing |
| | "service/ToS like OpenAI" | consent can't bind OTHERs; retention → controllership |
| | public donor-name list | new processing + membership inference |

## One-page narrative for meeting
1. Controller question first — everything hangs on it
2. Basis: 6(1)(e) + 9(2)(j)/§10 — consent not the regime for OTHERs
3. Notice: 14(5)(b) package (effort × impact × safeguards), objection = ongoing erasure
4. Safeguards menu offered generously; content fidelity = red line (scientific validity)
5. If pushback: ladder R1→R5, quantify fidelity cost per rung
6. End w/ written asks: basis confirmation, 14(5)(b) acceptance, DPIA review timeline
