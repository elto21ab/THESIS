# DPIA — Chat-Corpus Research ("LLMs as proxy survey participants")

**Draft v1.0 — [date] — internal working document (Art. 35 GDPR)**
Cross-refs: `donor-consent-tandc.md` v1.0, `option2-notice-exemption.md`, `clause-legal-map.md`. All date/name fields marked `[ ]` must match the T&C + Art. 30 record before finalization.

---

## 0. Screening — why a DPIA is required

| Trigger | Basis | Present? |
|---|---|---|
| Processing of special-category data on a large scale | Art. 35(3)(b); rec. 91 | Yes — private messages + surveys |
| Innovative technology (LLM retrieval pipeline) | Art. 35(1); rec. 91 | Yes — custom RAG inference |
| University research regime w/ exemption from individual notice | Art. 35(1) "high risk to rights and freedoms" | Yes — OTHERs can't self-assert |

DPO consulted incrementally (Art. 35(2)) — consultation ≠ approval: DPO advises, controller (supervisors) decides; no DPO veto (Art. 38(3) independence). Datatilsynet prior consultation (Art. 36) only if residual risk stays high after mitigations — §7 shows it does not.

## 1. Processing system — systematic description (35(7)(a))

### 1.1 Stakeholders

| Role | Party | Notes |
|---|---|---|
| Controller (Art. 4(7)) | [University], CVR [nr.] | formalized via supervisor-signed project, uCloud project, Art. 30 record |
| Project responsible | **[Supervisor 1 name]** (+ [Supervisor 2 name]) | sign DPIA; names in Art. 30 record |
| Team (named) | [2 student names] | the only named access holders besides supervisors |
| Processor (Art. 28) | uCloud / DeiC, Denmark | GDPR-compliant, DPA in place; no sub-processors outside EU/EEA |
| DPO | [name/email] | Art. 35(2) consultation |
| Data subjects | donors (SUBJECT); chat partners (OTHER); mentioned persons; minors in chats | three distinct classes, §1.3 |

### 1.2 Data categories (Art. 35(7); 5(1)(c) minimization)

- **Chat exports** (donor-selected conversations; FB/IG JSON, WhatsApp txt) incl. **low-quality media** (images, voice messages)
- **Survey responses** (Likert-scale)
- **Consent record** (timestamp + T&C version + checkboxes; Art. 7(1))
- **Contact email**

Media note: images/voice are personal data in the same corpus; NOT biometric data under Art. 9(1) — never used for identification (rec. 51). Special categories present in message *content* (health, opinions, etc.) → Art. 9 processing, basis covered in §2.

### 1.3 Data subject classes & basis

| Class | Data | Legal basis | Notice |
|---|---|---|---|
| SUBJECT (donor) | own chats, surveys, email | Art. 6(1)(a) + 9(2)(a) explicit consent | Art. 13 (the T&C) |
| OTHER (chat partners) | their messages (header-masked) | Art. 6(1)(e) + 9(2)(j) GDPR; §10 DBL | exempt: Art. 14(5)(b); substitute = T&C + public page + registry + opt-out mailbox |
| Mentioned persons | references inside messages | same as OTHER | same as OTHER |
| Minors in chats | incidental | same as OTHER + Art. 8/24(1) §6.4 | same; donor must be 18+ |

### 1.4 Data flow

```
DONOR ── T&C vX + 3 checkboxes (consent record) ──► upload site ──► uCloud (EU)
   │                                                        │
   └─ chats selected by donor; media low-quality ───────────┘
                                                            ▼
                    pseudonymized corpus: names → SUBJECT/OTHER; encrypted at rest
                                                            ▼
                    retrieval pipeline (RAG inference) — NO training, NO raw export
                                                            ▼
                    [12 months after thesis assessment] → destruction incl. backups
                    (only non-identifiable aggregates remain → Rec. 26 zone)
```

Retention: **[date — must equal T&C §7 and Art. 30 record]**.

### 1.5 Purposes (5(1)(b); §10 DBL "solely scientific study")

Build personal retrieval corpus → local LLM imitates donor's survey answers (proxy personas) → compare proxy vs. real answers. **Never:** raw data publication, model training, commercial/demo/startup use, per-individual reporting on OTHERs. OTHERS' messages are retrieval context only — not analyzed, quoted, profiled.

## 2. Necessity & proportionality (35(7)(b), 5(1)(c), 25)

| Question | Answer |
|---|---|
| Is processing necessary for the purpose? | Yes — imitation validity requires authentic conversational signal; no anonymized/synthetic substitute (see option1 file: anonymization destroys signal, fidelity cost ladder) |
| Full corpus needed — why not only SUBJECT msgs? | OTHER msgs are the majority of retrieval context; cutting them changes register/lexicon → invalidates imitation (documented in thesis method) |
| Why no individual consent/notice for OTHERs? | impossible channel (no platform API); n×10²–10³ contacts per donor; would break corpus↔survey timing + bias sample → Art. 14(5)(b) exemption (§3.2 in option2 file; recruitment-friction exhibit) |
| Consent for OTHERs why not? | consent is personal — donors cannot consent for partners; consent-all = self-defeating (same evidence) |
| Least-intrusive means? | yes — donor picks conversations; header masking; low-quality media; no media extraction/analysis; destruction schedule; research-only use |

Proportionality verified: benefit (academic knowledge + open reproduction, university research in the public interest, Art. 6(1)(e)) outweighs harm to OTHERs — near-zero (§3/§4), because OTHERs are never the unit of analysis.

## 3. Rights of data subjects (channels must exist, 12–22)

| Right | SUBJECT | OTHER / mentioned |
|---|---|---|
| Access (15), rectification (16) | yes, via mailbox | opt-out mailbox, located via donor-side key |
| Erasure (17) | on withdrawal (7(3)) | on objection (21(6)) → delete, confirm in writing |
| Restriction (18), objection (21) | yes | mailbox (21(6)) |
| Portability (20) | surveys only (20(4): corpus contains third-party data — porting would harm others' rights) | n/a |
| No automated decisions (22) | guaranteed — no profiling/automated decisions | guaranteed |
| Complaint (77) | Datatilsynet | Datatilsynet |

Substitute transparency for unreachable persons: T&C published + public project page + university research-registry entry, all describing purpose, categories, basis, retention, rights channel. Voluntary donor notification logged where it happens (14(5)(a)) — never required.

## 4. Risk assessment (35(7)(c); baseline: likelihood × severity)

| # | Risk to data subjects | Likelihood | Severity | Baseline |
|---|---|---|---|---|
| R1 | Breach/leak of chat corpus (unauthorized access) | Low | High | Med-High |
| R2 | Re-identification of OTHERs from published outputs | Very low | High | Med |
| R3 | Misuse outside research (commercial/other) | Low | Med | Med |
| R4 | OTHERs unaware → no chance to object | Certain (by design) | Low | Med |
| R5 | Retention overrun → exposure extends | Low | Med | Med |
| R6 | Donor regret (withdrawal friction) | Low | Med | Med |
| R7 | Loss (accidental deletion → research validity, not privacy) | Low | Med | Low (integrity) |

## 5. Safeguards & mitigations (35(7)(d); 89(1), 32, 25, 5(1)(f))

| # | Mitigation | Effect | Residual |
|---|---|---|---|
| R1 | encryption at rest + in transit; pseudonymized storage; access = 4 named (2 students + 2 supervisors); uCloud DPA (28); least-privilege | unauthorized access hard | Low |
| R2 | no raw corpus publication (preemptive concession); aggregates only; header-masking; no quotes, no per-individual OTHER output | re-ID channel closed | Very low |
| R3 | §10 "solely scientific study" commitment in T&C + internal policy; no demo/portfolio use | misuse excluded | Low |
| R4 | substitute notice package (T&C, public page, registry, opt-out mailbox); Art. 21(6)/17 erasure honored anytime, no pre-processing window | reachable channel exists | Low (accepted residual — the 14(5)(b) trade) |
| R5 | automated destruction incl. backups; written confirmation; DPIA + Art. 30 record pin same date | overrun controlled | Low |
| R6 | documented erasure procedure; 30-day SLA; no friction, no questions | regretted donation reversible | Low |
| R7 | backup + restore test; versioned export | integrity | Low |

Plus: DPO consultation recorded (35(2), comments incorporated or answered); breach notification per Art. 33/34 + [national rules]; staff (team) GDPR awareness — only 4 people touch data.

## 6. Special-risk measures

1. **Media files** — low-quality only, stored encrypted, never extracted/analyzed for identity; treated as text-equivalent in corpus. Not biometric (§1.2).
2. **Minors** — incidental data only; donor 18+; minors' data under same safeguards; if contacted via mailbox, standard erasure path.
3. **Consent integrity** — 2 distinct consent acts (6(1)(a) / 9(2)(a)) + acknowledgment box; pre-ticked off; record of timestamp/version/boxes (burden of proof, rec. 42).
4. **Objection loop** — OTHERs reach via mailbox → donor-side key → located → deleted → confirmed. No pre-window: objection honored immediately (21(6)).

## 7. Residual risk conclusion

All risks ≤ **Low** after mitigations. No high residual risk → **no prior consultation with Datatilsynet required (Art. 36)**. The 14(5)(b) trade (R4) is the lawfully chosen, documented one.

## 8. Review & document control

DPIA must be reviewed when: retention date changes, scope changes (e.g., platform extensions), access roster changes, processor changes. Review at least every [12 months] while data is held.

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0 draft | [ ] | initial | [students] |

## 9. Sign-off

| Role | Name | Signature/Date |
|---|---|---|
| Project responsible (supervisor 1) — approves | [ ] | |
| Supervisor 2 — approves | [ ] | |
| Students (DPIA authors) | [ ][ ] | |
| DPO — consultation record (35(2), comment log) | [ ] | |
| Processor (uCloud) DPA confirmed | [ ] | |

---

**Before go-live:** fill all `[ ]`, align retention date with T&C §7 + Art. 30 record, publish substitute-notice page + registry entry (T&C checklist, clause-legal-map §4).