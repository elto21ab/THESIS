# Legal Strategy — Donor Chat Corpus (SUBJECT + OTHER)

## Problem
Corpus per donor contains SUBJECT (donor, consented) + OTHER (3rd-party friends, unconsented, unnotified). OTHER data = the legal problem. Goal: max signal retention, min GDPR exposure.

## DPO letter mapping
DPO "Option 1 full anonymization" = our Option 2 (but overbroad standard — see option0).
DPO "Option 2 consent-all" = new Option 5 (defective, see option0 §consent).
DPO omitted: §10/9(2)(j) + 14(5)(b) research route — our Option 3, PRIMARY COUNTER.

## Options ranked (defensibility × science-value)

| # | Option | GDPR path | Signal kept | Defensibility |
|---|--------|-----------|-------------|----------------|
| 3 | §10/9(2)(j) basis + 14(5)(b) notice exemption + Art. 89 safeguards + public notice + ongoing erasure | 6(1)(e) + 9(2)(j) | 100% | ★★★★ |
| 2 | "Effective anonymization" dossier: PII-triage → decoy-injected obfuscation (DP/RR plausible deniability) → motivated-intruder test | exits scope (Rec. 26 risk standard) | 60–90% | ★★★ |
| 5 | Consent from all OTHERs (DPO's path) | 6(1)(a)+9(2)(a) | 100% | ★★ (feasibility ★) |
| 1 | Household exemption restructure | 2(2)(c) | 100% | ★ — do not raise |

**Strategy: lead w/ Option 3 counter-frame (option0). Offer Option 2 techniques as Art. 89 safeguards OR as genuine anonymization dossier if DPO insists on their Opt 1. Option 4 folded into Option 2 as aggressiveness dial (2.1 PII-mask → 2.2 full-msg obfuscation → distilled context).**

## Non-negotiable artifacts (any path)
- DPIA (Art. 35) — likely mandatory: large-scale Art. 9 data + systematic monitoring
- Record of processing (Art. 30)
- uCloud DPA + Art. 32 security doc
- Retention/destruction schedule
- Motivated-intruder / reID risk assessment (if any anonymization-flavored claim made)
