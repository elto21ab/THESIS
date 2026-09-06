# Clause → Provision Map (flowchart style)

Reference for supervisor/DPO. Cross-refs `donor-consent-tandc.md` v1.0. The T&C is the only donor-facing document; this file is internal.

## 1. Flow — one breath

```
DONOR on upload site
 ├─ reads T&C vX + ticks 3 checkboxes (consent + explicit + ack)  → Art. 7(2)/(4), 13 info
 ├─ submits consent record (ts + version + boxes)                 → Art. 7(1) demonstrate
 ├─ uploads chat JSON (FB/IG/WA, donor-picked) + surveys
 ▼
CORPUS = SUBJECT ∪ OTHER ∪ mentioned persons
 ├─ SUBJECT data
 │    └─ consent: 6(1)(a) + 9(2)(a)                               → §4(a), boxes 2–3
 ├─ OTHER + mentioned data
 │    ├─ research basis: 6(1)(e) + 9(2)(j) + DBL §10              → §4(b)  [NO consent needed]
 │    ├─ no individual notice: 14(5)(b)  (effort + impairment)    → §4(b)
 │    │    └─ substitute: T&C + public page + registry + mailbox  → §5, §8
 │    └─ supplementary: donor-informed contacts logged (14(5)(a)) → §5 [optional, never required]
 ▼
SAFEGUARDS layer (price of no-notice/no-consent)
 ├─ header-mask SUBJECT/OTHER, pseudonymize, encrypt              → 89(1), 25, 32, 5(1)(c)
 ├─ 4 named access holders (2 students + 2 supervisors)          → 32(1)(d)
 ├─ DPIA + DPO consult + Art. 30 record                           → 35(1), 35(2), 30
 ▼
uCloud (processor, Denmark, EU)                                     → 28 DPA; no 3rd-country transfer
 ▼
RAG inference — no training, no raw publication, research-only      → §10 DBL "solely scientific study"
 ├─ withdrawal (donor) / objection (OTHER) anytime                  → 7(3), 21(6), 17 → erasure, no delay
 ▼
DESTRUCTION (≤ retention per DPIA); only aggregates remain          → 5(1)(e); aggregates = rec. 26 zone
```

## 2. Clause-by-clause

| T&C § | Provision | Role | Trap / note |
|---|---|---|---|
| 1 purpose | Art. 13(1)(c); 5(1)(b) | purpose limitation; "solely scientific study" (§10) framing | never promise commercial value |
| 2 collection | 5(1)(c) minimization; donor picks chats | donor-controlled scope = stronger consent | don't auto-ingest whole account |
| 3 use-limits | 5(1)(b); §10 "solely scientific study" | no training/no publication = core red line | statement, not legal basis |
| 4(a) donor basis | 6(1)(a), 9(2)(a), 7, 13(1)(a–e) | load-bearing: donor consent | explicit box for special categories is mandatory (9(2)(a)) |
| 4(b) OTHER basis | 6(1)(e), 9(2)(j), §10, 14(5)(b), 13→14 | **the load-bearing clause** | must NOT say "they consented" — consent is personal, donors can't give it |
| 5 voluntary notice | 14(5)(a) supplementary | optional log; adds belt | **requiring** notification = self-defeating (kills 14(5)(b) effort argument) |
| 6 storage/security | 28 (uCloud DPA), 32, 89(1), 25, 35, 30 | the safeguards package = condition for 14(5)(b) | claim EU-only, not "no data leaves DK" unless verified |
| 7 retention | 5(1)(e); destruction schedule | minimize exposure window | pick date, put in DPIA, match both docs |
| 8 rights | 12–18, 20(4), 21(6), 17, 22, 8 (minors) | donor: full rights; OTHER: mailbox channel | portability limited to surveys (20(4): don't harm others) |
| 9 withdrawal | 7(3), 17, 21(6) | erasure = the objection loop | no pre-processing window — honor w/o friction |
| 10 record | 7(1) demonstrate consent | timestamp+version+boxes; keep 5y | recital 42: burden of proof on controller |
| 11 checklist | 7(2), 7(4), EDPB 05/2020 §77 | 2 consent acts (6(1)(a) vs 9(2)(a)) must be distinguishable; 3rd = ack/transparency proof for 14(5)(b) | one "I agree to everything" box alone = invalid |
| contacts | 13(1)(a), 13(2)(b), 13(2)(e) | Art. 13 completeness | always include Datatilsynet complaint route |

## 3. Never claim in donor-facing docs

- ✗ "By donating you confirm your contacts consent" — false, voids the whole architecture
- ✗ "The corpus is anonymized" — it is pseudonymized; mislabeling invites Breyer/Rec. 26 scrutiny and contradicts 14(5)(b) logic
- ✗ Household exemption (2(2)(c)) — covers donor's own collection only
- ✗ "Like OpenAI/service X" — consent can't bind OTHERs
- ✗ Public donor-name list — new processing + membership-inference leak
- ✗ Consent as basis for OTHERs' data — basis is research regime, period
- ✗ "No personal data is processed" — corpus is personal data, end of story

## 4. Site implementation checklist

- [ ] T&C version pinned + referenced by consent record (Art. 7(1))
- [ ] 3 checkboxes individually required, pre-ticked = off (Art. 7(2); rec. 32)
- [ ] Consent log: timestamp + version + boxes (+ optional email)
- [ ] Opt-out mailbox wired to donor-side-key lookup procedure
- [ ] Public project page + registry entry (substitute notice) live before recruitment
- [ ] Danish version published (Art. 12(1) plain language; DK audience)
- [ ] DPIA + Art. 30 record reference the same retention date as §7