# Option 3: Transparency route — RECOMMENDED PRIMARY

## Architecture
- **Lawful basis**: Art. 6(1)(e) (public-interest research) or 6(1)(f) (legitimate interest + balancing)
- **Art. 9 special categories**: §10 Databeskyttelsesloven / Art. 9(2)(j) research basis
- **SUBJECT**: explicit informed consent (donor; covers own data + donation act)
- **OTHER**: no consent (impracticable) → transparency handled via **Art. 14(5)(b) exemption**
- **Safeguards**: Art. 89(1) — pseudonymization (Option 2 techniques), minimization (Option 4), access control, destruction schedule, DPIA

## Art. 14(5)(b) claim for OTHER (one package, present together)
Provision of notice "impossible or would involve disproportionate effort" / "seriously impair achievement of objectives":

**Effort side**
- Donor = only channel to their contacts; no API for personal FB Messenger/IG/WA; export-only platforms
- n donors × 10²–10³ contacts × manual outreach = structural impairment, not mere inconvenience
- Cost falls on donors: time + social/reputational burden → cohort becomes unobtainable (selection collapse)

**Impact side (harm ~0)**
- OTHER is not the unit of analysis — SUBJECT is; OTHER msgs used only as retrieval context
- No profiling, no per-individual results, no quotes/publication of OTHER content, corpus destroyed on schedule
- Likert-only surveys → no free-form bleed into comparison targets
- Pseudonymization + DP → negligible reID/harm surface
- **Arg: notice confers ~zero protection while eliminating feasibility**

**Kill the "objection window" if proposed — it doesn't exist in law.** No pre-processing objection window in GDPR. Art. 21 objection = ongoing right → honor via erasure when raised. Timing sync preserved. Substitute safeguards: public notice + opt-out mailbox + pre-registered destruction + no-individual-results. Expected objection rate ≈ 0 (near-zero awareness × near-zero harm); pre-commit to documented exclusions/re-runs if any arise.

## Public notice design (what suffices)
**NOT a public donor-name list** (GitHub repo idea rejected): donor list = itself personal data; publishing = new processing + membership-inference leak (Art. 9-adjacent). Instead:
- Project webpage + university research registry entry: purpose, data categories (chat logs incl. conversation partners + mentioned persons), §10/9(2)(j) basis, retention/destruction schedule, contact
- Opt-out/erasure mailbox + documented procedure (any OTHER can request exclusion; we locate via donor-side key, delete, confirm)
- Recruitment-channel posts linking the notice
- Log evidence of publication (screenshots, timestamps) for the record

## Fallback ladder (if DPO rejects full exemption)
1. Art. 14(5)(b) + **public notice** (above) — cheap, additive
2. Art. 14(5)(a) where donor demonstrably already informed contacts
3. Direct notice only for OTHERs in small high-risk subset (Art. 9-flagged msgs — Option 2 triage makes feasible)
4. Worst case: donor-mediated notice → cohort collapse; document as evidence for (b) retroactively

## Known risk
14(5)(b) research limb requires Art. 89 safeguards + documented balancing. DPO gatekeeps the balancing. DPO's letter ignored this route entirely → our reply (option0) forces them to engage w/ §10/9(2)(j) explicitly. Recruitment friction evidence (donors refuse friend-notification) = our strongest empirical exhibit.
