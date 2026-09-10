# Supervisor Brief — GDPR path decision (10 min read)

**Decision needed from u: pick path A, B, or C.** Our recommendation: **A**.

## Challenge
Process third-party messages (OTHERs and mentioned persons). DPO pushed binary: consent from all, or full anonymisation.

**The DPO's binary assumes we students need an individual legal basis. We don't** — no
individual basis is required when processing sits under the university's public-interest
research mandate: university = controller, we're named processors under its safeguards.
That is the path the DPO's letter omitted.

## The 3 paths
> From easiest to safest.

| | A — research basis, DPO out | B — formalize w/ DPO | C — anonymisation |
|---|---|---|---|
| Legal basis | Art. 6(1)(e) + 9(2)(j) + DBL §10 + 14(5)(b) | same, + DPO sign-off | none needed (out of scope, Rec. 26) |
| Signal kept | full | full | reduced (unknown) |
| DPO touch | none | ask | inform |
| Risk | no DPO validation if challenged | DPO may push back → costs time, burn A | DPO might still push back |
| Limiting ourself | all options stay open | A burned if DPO declines | A and B got burned, since this is last resort |

A Keeps full signal, burns nothing, and the legal is the same
B would formalize — B only buys DPO comfort/responsibility at the price of a veto. C stays on the table as fallback if A is later challenged.

## Path A — research basis, DPO out

### Legal spine (university = controller; u sign as project responsible)

1. **[Art. 6(1)(e)](https://gdpr-info.eu/art-6-gdpr/)** public-interest research = general basis (transposed via **DBL §10**)
2. **[Art. 9(2)(j)](https://gdpr-info.eu/art-9-gdpr/)** → special-category data processable for research, **consent not required**
3. **[Art. 14(5)(b)](https://gdpr-info.eu/art-14-gdpr/)** → individual notice to OTHERs exempt (impossible channel / disproportionate effort / seriously impairs objectives)
4. **[Art. 21(6)](https://gdpr-info.eu/art-21-gdpr/)/[17](https://gdpr-info.eu/art-17-gdpr/)** → objection = ongoing right → erasure on request, no pre-processing window
5. **[Art. 89(1)](https://gdpr-info.eu/art-89-gdpr/)** safeguards = the condition we fulfill: header-masking, pseudonymized storage, access control, [DPIA (Art. 35)](https://gdpr-info.eu/art-35-gdpr/), destruction schedule, research-only use

### Why we qualify as research (not "ordinary thesis work")

Betænkning 1565 has **no black-letter thesis rule** — Datatilsynet's test is standards-based
(new knowledge + collective benefit). So the claim rests on evidence, not statute:

- Co-authored paper + supervisor-funded travel → de facto integration into your research agenda
- Funded conference invite + accepted poster + peer-reviewed abstract = external peer validation
- [Recital 157](https://gdpr-info.eu/recitals/no-157/) "knowledge society" contribution
- Controllership formalized: you sign DPIA, Art. 30 record, uCloud project, named access → purposes/means determined at university level ([Art. 4(7)](https://gdpr-info.eu/art-4-gdpr/))

### Necessity & proportionality (already defenseced in DPO email correspondence)

- Per-OTHER data surface ≈ 0.1% of corpus (concervatively assuming 500 OTHERs per donor ==> SUBJECT≈50%, OTHERs ≈0.2% if incl. complementing SUBJECT-side); OTHERs are never the unit of analysis — their messages exist only as retrieval context, two steps removed from any output
- No profiling, no quotes, no per-individual results; destroy-on-schedule → per-person harm ≈ 0
- Effort side: consent-all = infeasible + scales badly w/ donor count (14(5)(b) effort/
  impairment limbs grow w/ n donors); documented recruitment friction = the exhibit
- Proportionality = effort-to-notify vs impact-of-not-notifying → falls clearly our way;
  residual risk covered by Art. 89(1) safeguards

**Risk, stated plainly:** if Datatilsynet ever reviews us, we are on our own
w/o the DPO's prior blessing. We judge the legal position strong enough; DPO contact then becomes "here is our basis," not "may we."

Note: We are only legally obligated to seek consultation of DPO, and we are not required to get approval from the DPO.

## Path B — formalize w/ DPO

Same legal spine as A, but we ask the DPO to validate it first. Buys institutional cover.
Cost: once DPO engages they may demand other formalities or entirely reject the proposal → A is burned and we're negotiating from their binary again.

**Meeting flow if B chosen** (applies to any DPO contact):
1. **BASIS first** — state the university research mandate (DBL §10 → 6(1)(e) → 9(2)(j) →
   14(5)(b)); present it as established. Opening ask: *"what does it take
   to formalize this under the university research exception?"*
2. **MECHANISM second** — state our safeguards as fact, no permission-seeking: header
   pseudonymisation (names → SUBJECT/OTHER), access control, destruction schedule. No PII
   detection, no vector/paraphrase obfuscation — content stays untouched (that's path C
   only). Don't volunteer internal pipeline detail — it invites uninformed veto and burns
   negotiation room.
3. Recital 157 / peer-validation evidence held in reserve; deploy only if challenged.

## Path C — anonymisation (out of GDPR scope)

Differential-privacy (DP) anonymisation **at ingestion** → data anonymous → GDPR doesn't apply
([Rec. 26](https://gdpr-info.eu/recitals/no-26/); betænkning 1565 confirms, "herunder til
forskningsmæssige formål"). No basis, no DPIA, no DPO gate. DPO contact = one-line written
inform, zero questions.

Two honest flags: (1) fidelity cost **unmeasured** — DP preserves population stats, we
believe the imitation signal survives, but it's unquantified; (2) only works if anonymisation
happens at ingestion — anonymisation must happen on data donor's computers (complication of hardware-agnostic design).

## What we need from u

**Pick A, B, or C.** Logistics follow once the path is set — A and B both need u as project
responsible (DPIA co-owner, Art. 30 record, uCloud, named access); C needs nothing.

## Exhibits (in this folder)
- `option2-notice-exemption.md` — full legal chain + 14(5)(b) evidence package (paths A/B)
- `option1-anonymization.md` — anonymisation standard, redaction ladder, DP strategy (path C)
- `dpia-draft.md`, `dpo-flowchart.md`, `clause-legal-map.md` — DPIA, process diagram, clause map
- Documented recruitment friction (dated, quantified) = the 14(5)(b) "disproportionate effort" exhibit
