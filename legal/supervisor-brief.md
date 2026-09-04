# Supervisor Brief — GDPR blocker & our strategy (10 min read)

## The challenge
Corpus per donor contains SUBJECT (donor, consents) + OTHERs (their chat partners) + mentioned persons. DPO says: OTHERs' personal data lacks legal basis → offered only (1) full anonymization of ALL third-party data, or (2) consent from every OTHER. Both kill the study: anonymization destroys the conversational signal the imitation task needs; consent-all is infeasible (no channel to 10²–10³ contacts per donor; recruitment friction already observed).

## The path DPO's letter omitted
Their binary assumes *we students* need an independent legal basis. We don't: under university controllership, the standard Danish research regime applies:
- **University = controller**, u sign as project responsible (DPIA, uCloud project, Art. 30 record names us as project members)
- **Art. 6(1)(e)** public-interest research + **Art. 9(2)(j) / DBL §10** → special-category data processable for research **without consent**
- **Art. 14(5)(b)** → individual notice to OTHERs exempt (impossible channel / disproportionate effort / seriously impairs objectives)
- Objection = ongoing right → erasure on request (Art. 21/17), no pre-processing window
- **Art. 89(1)** safeguards = the condition we fulfill: header-masking (names→SUBJECT/OTHER), pseudonymized storage, access control, DPIA, destruction schedule, research-only use

We keep full-fidelity data; the price is procedural hygiene, not redaction.

## What we need from u
1. Sign on as project responsible (formalizes university controllership — the linchpin)
2. Confirm institution's standard practice for thesis research w/ personal data under §10 (if any)
3. Co-own the DPIA + be named access holder w/ us
4. Join/back us in the DPO meeting — our ask: "what safeguards make 14(5)(b) sufficient?" not "may we?"

## If DPO still refuses
Fallback ladder of redaction rungs (consistent pseudonyms → chunk-varying → decoy injection → full obfuscation), each traded against measured fidelity cost. Last resort: re-scope as ur research subproject w/ the thesis as contribution.

## Key exhibits we'll bring
- Documented recruitment friction (donors refuse to notify friends) = 14(5)(b) effort evidence
- Dataflow diagram, DPIA draft, retention/destruction schedule
- Legal map: legal/dpo-flowchart.md
