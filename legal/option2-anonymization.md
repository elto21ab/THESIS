# Option 2: Anonymization / Pseudonymization

## Critical distinction — don't conflate
- **Anonymization (Rec. 26)**: irreversible → exits GDPR scope entirely. Bar: no singling-out, linkability, inference, accounting for "means reasonably likely to be used" (motivated intruder).
- **Pseudonymization (Art. 4(5))**: still personal data, GDPR applies, but counts as Art. 89(1) / Art. 32 safeguard → reduces risk, supports DPIA + 14(5)(b) arguments.

**Strategy: pursue "effective anonymization" as dossier, not bare claim.** DPO letter demands absolutist standard — counter w/ Rec. 26 "reasonably likely means" + Breyer relativity + WP29 three-risk test (singling-out/linkability/inference). Core technical arg (randomized-response logic): PII-model triage → whole-msg obfuscation w/ plausible-false decoys → reID requires confirmation, confirmation requires ground truth, ground truth provably corrupted → inference below reasonable-likelihood. Deliverable: adversary model + measured residual risk + DPO sign-off.

## Where it runs
Donor-side script (own hardware) → only processed output uploaded to uCloud. Same front-end as Option 1 salvage. Proves minimization intent.

## Methods (top-down PII)
- NER/pattern PII removal: Presidio > spaCy NER > Flair; Faker for replacements
- Mask: redact `(NAME)` / consistent tag `(NAME:1)` / plausible decoy (culturally adjacent fake)
  - Consistent tags preserve coreference (science) but keep linkability *within* corpus — fine for pseudonymization, fatal for anonymization claim
  - **Vary tags across chunks/donors → kills cross-source linkability**
- Option 4 folded in as aggressiveness dial: 2.1 top-down mask → 2.2 bottom-up full-msg obfuscation → distilled context {topic, sentiment, relation, timestamp}. Pick rung by measured fidelity curve.
- Models: bardsai/eu-pii-anonimization-multilang(-v2), tabularisai/eu-pii-safeguard, flowxai/piiguard, perplexity pplx-pii-masking, Roblox pii-classifier-v2, agentlans multilingual-e5/embeddinggemma PII detectors

## Methods (bottom-up obfuscation)
- Paraphrase / seq2seq / txt2vec2txt — context-preserving style scrambling
- Vec2Text (Morris et al.) rebuttal: purpose-built inverter, 2 models, train-data leakage, accuracy collapses >32 tokens → inversion output unverifiable (plausible deniability); lossy scrambler ≈ guess, and a guess ≠ reID under Rec. 26 "reasonably likely means"

## Differential privacy (randomized-response core)
DP noise / RR-style decoy injection on flagged msgs → per-attribute plausible deniability (coin-flip cheating-survey logic): even correct linkage cannot confirm any attribute. reID = isolation + linkage + CONFIRMATION; we break confirmation structurally. Strongest single argument for OTHER.

## Compliance artifact (motivated intruder test)
Can't prove negative → characterize strongest reasonably-likely attack, show structural failure:
1. Define adversary model (resources, auxiliary data: LinkedIn, press, corp filings)
2. Test singling-out / linkability / inference separately (WP29 three-risk test)
3. Show obfuscation creates *crowds* (k-anonymity-like generalization), not injective renames
4. Also cover nth-order OTHERs (people *mentioned* in msgs — DPO explicitly flagged this): decoy injection corrupts references to non-participants equally; no confirmation path exists for them either
5. Documented + measured tests of known sensitivity → DPO reviews; Datatilsynet/courts decide only if challenged

## Cost
Removing Art. 9 topics + heavy paraphrase degrades imitative signal → quantify expected performance drop (run pilot: retrieval accuracy / imitation fidelity pre- vs post-processing) and bring numbers to DPO. Turns "sabotages the science" from vibe into evidence.
