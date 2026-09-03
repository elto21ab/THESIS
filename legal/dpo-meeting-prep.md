# DPO Meeting Prep

## Position (one-liner)
"SUBJECT consents; OTHER handled under Art. 9(2)(j)/§10 + Art. 14(5)(b) disproportionate-effort/research-impairment, wrapped in Art. 89 safeguards (pseudonymization, minimization, DPIA, destruction)."

## Bring
1. DPIA draft (or at minimum risk register: singling-out/linkability/inference × SUBJECT/OTHER)
2. Dataflow diagram: donor device (household-step script) → uCloud (pseudonymized corpus) → retrieval/LM → destroyed corpus; mark where OTHER data exists and in what form
3. Quantified minimization curve (Option 4 pilot: fidelity vs OTHER-data level) — even preliminary
4. Notice-exemption memo: effort/impact balancing as ONE combined 14(5)(b) claim (see option3.md)
5. Retention & destruction schedule; access list (3 named people); uCloud DPA ref

## Anticipated objections → responses
- "Students lack legal basis" → irrelevant; university = controller, supervisor = PI; we process under institutional authority (standard for all student research)
- "Why not consent OTHER?" → no channel, n×10²–10³, cohort collapse (documented recruitment friction), selection bias; consent-all infeasibility = the 14(5)(b) trigger itself
- "Why not fully anonymize (their Opt 1)?" → their standard overbroad vs Rec. 26 "reasonably likely means" (Breyer); we offer anonymization DOSSIER: decoy-injected obfuscation + adversary model + measured residual risk; reID needs confirmation, confirmation needs ground truth, ground truth provably corrupted
- "Objection window?" → no such legal requirement; Art. 21 = ongoing right → erasure on objection; public notice + opt-out mailbox as substitute
- "nth-order OTHERs (mentioned persons)?" → decoy injection corrupts their references equally; no confirmation path; also covered by 14(5)(b)
- "Art. 9 data of OTHER?" → donor-side triage redacts/flags; residual covered by 9(2)(j) + safeguards

## Asks (concrete, so meeting ends w/ decisions)
1. Confirm Art. 6 basis preference (6(1)(e) vs 6(1)(f)) for this institution
2. Confirm 14(5)(b) package acceptable in principle + which safeguards are load-bearing
3. Agree DPIA scope + review timeline
4. Get minimum OTHER-data level DPO would accept (informs Option 4 rung)
5. Written confirmation pathway (email summary → DPO sign-off)

## Red lines (don't concede)
- Per-OTHER direct consent requirement (= project death + their own letter's infeasibility is our evidence; if insisted, demand written reasoning for escalation)
- Public donor-name list as "notice" (creates new processing + membership leak; refuse on privacy grounds — ironic win)

## Easy concessions (offer early, buy goodwill)
- Public notice webpage, opt-out mailbox for OTHERs
- No raw quotes in any publication, aggregate-only results
- Corpus destruction certificate
- Donor-side pre-processing (Option 1 salvage) as standard
