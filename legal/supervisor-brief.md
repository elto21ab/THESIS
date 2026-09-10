# Supervisor Brief — GDPR path decision

**Decision needed from u: confirm Path A (premise-change re-consultation); posture + share level are ours (see Sub-decisions).**

## Challenge
Process third-party messages (OTHERs and mentioned persons). DPO pushed binary: consent from all, or full anonymisation.

**The DPO's binary assumes we students need an individual legal basis. We don't** — no individual basis is required when processing sits under the university's public-interest research mandate: university = controller, we're named processors under its safeguards. That is the path the DPO's letter omitted.

## The 2 paths (A and B merged)

**Why A and B are now one path:** our consultations with the DPO ran on the premise that we students are controllers — her consent-or-anonymise binary follows from that premise. New premise: university = controller, we are named processors under its safeguards (Art. 4(7) controllership formalised). The premise change invalidates the old consultation's footing: Art. 35(2) mandates a record that matches what we actually do → re-consultation is mandatory, not a choice. And because her role is advice (Art. 38(3): no veto), "get approval" was never a real alternative either. Old A (DPO out) and old B (ask approval) collapse into one path; the only levers left are posture and share level (Sub-decisions).

| | A — premise-change re-consultation (primary) | C — anonymisation fallback |
|---|---|---|
| Legal basis | 6(1)(e) + 9(2)(j) + BL §10 + 14(5)(b) + 89(1) | none needed (out of scope, Rec. 26) |
| Signal kept | full | reduced (unmeasured) |
| DPO touch | re-consult (mandatory) | no |
| Risk | her objection on *new* basis → answer or escalate | anonymisation bar contested |
| Default | primary | last resort |

## Path A — premise-change re-consultation (A+B merged; primary)

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

### Necessity & proportionality (already defended in DPO email correspondence)

- Per-OTHER data surface ≈ 0.1% of corpus (conservatively assuming 500 OTHERs per donor ⇒ SUBJECT ≈ 50%, OTHERs ≈ 0.1% each if incl. complementing SUBJECT-side); OTHERs are never the unit of analysis — their messages exist only as retrieval context, two steps removed from any output
- No profiling, no quotes, no per-individual results; destroy-on-schedule → per-person harm ≈ 0
- Effort side: consent-all = infeasible + scales badly w/ donor count (14(5)(b) effort/
  impairment limbs grow w/ n donors); documented recruitment friction = the exhibit
- Proportionality = effort-to-notify vs impact-of-not-notifying → falls clearly our way;
  residual risk covered by Art. 89(1) safeguards

**Risk, stated plainly:** worst case = written objection on the *new* basis. Art. 38(3):
her advice is reasoned, no veto → we answer in writing, log advice + our response in DPIA
§8, controller (u) decides whether to proceed. Art. 35(2) obligates us to *seek* her advice
on the DPIA — nothing obligates us to *follow* it, only to document deviations. Only true
escalation beyond her: Art. 36 prior consultation w/ Datatilsynet, triggered only by
unmitigable high residual risk — our safeguards aim to stay below that threshold.

## Sub-decisions — posture & share level (the only real choices left in A)

### Posture — degree of ask vs statement

| Posture | Wording | Buys | Costs |
|---|---|---|---|
| 1. Ask | *"what does it take to formalize the research exception?"* | max invite; supervisor comfort | frames us as supplicants; lets her define the gate |
| 2. **Statement (rec)** | present basis + DPIA as established; *"comments welcome (Art. 35(2) consultation; approval not sought — Art. 38(3) gives no veto)"; we proceed unless u name a specific defect; comments logged in §8* | keeps initiative; compresses her into a y/n on the NEW basis; fulfills the mandatory re-consult | small risk: written objection → answer or escalate |
| 3. Proceed-first | minimal consult now; act; inform later | fastest | weakest record; looks like hiding the premise change |

Rec: **posture 2** — the only one that is both an honest re-consultation and keeps possession of the frame. Escalate to 1 only if supervisor prioritizes her blessing over initiative (B's old logic); it re-imports her gate.

### Share level tiers (what we reveal)

| Tier | Content | When |
|---|---|---|
| T1 — always | the premise change: univ=controller, we=processors; basis chain (6(1)(e)/9(2)(j)/§10/14(5)(b)); 89(1) safeguards; DPIA exists | opening — is the argument |
| T2 — on direct Q | necessity evidence (Occhipinti 0.509→0.351, Park); 0.1% surface; consent-all infeasibility + 14(5)(b) effort; recruitment-friction exhibit | only if she questions necessity (already in emails) |
| T3 — reserve (never volunteer) | PII-detection / NER / vector / DP / txt2vec method detail | only if forced to Path C or she demands the anonymisation story — then deploy as fallback concession, with measured utility |

Rule: T1 always, T2 answered, T3 held. Note (our DPO-confidence insight): she lacks the toolkit to price DP/vector re-identification — do not hand her that card unasked.

### If she objects in writing (posture-2 failure path)

Art. 38(3): reasoned advice, no veto. Escalate: written reply to her reasoning, log in DPIA §8, controller (u) decides. Last resort: re-scope as ur research subproject w/ the thesis as contribution.

## Path C — anonymisation (out of GDPR scope)

Differential-privacy (DP) anonymisation **at ingestion** → data anonymous → GDPR doesn't apply
([Rec. 26](https://gdpr-info.eu/recitals/no-26/); betænkning 1565 confirms, "herunder til
forskningsmæssige formål"). No basis, no DPIA, no DPO gate. DPO contact = one-line written
inform, zero questions.

Two honest flags: (1) fidelity cost **unmeasured** — DP preserves population stats, we
believe the imitation signal survives, but it's unquantified; (2) only works if anonymisation
happens at ingestion — anonymisation must happen on data donor's computers (complication of hardware-agnostic design).

## What we need from u

**Confirm path + posture.** A needs u as project responsible (DPIA co-owner, Art. 30 record, uCloud, named access) + posture pick (rec: statement); C needs the anonymisation decision only.

## Exhibits (in this folder)
- `option2-notice-exemption.md` — full legal chain + 14(5)(b) evidence package (path A)
- `option1-anonymization.md` — anonymisation standard, redaction ladder, DP strategy (path C)
- `dpia-draft.md`, `dpo-flowchart.md`, `clause-legal-map.md` — DPIA, process diagram, clause map
- Documented recruitment friction (dated, quantified) = the 14(5)(b) "disproportionate effort" exhibit
