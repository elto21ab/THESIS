# Option 1: Anonymization (FALLBACK / concession ladder)

Exit GDPR scope by making OTHER data non-identifiable (Rec. 26). DPO demanded this as their "Option 1" — but w/ an overbroad standard. Primary strategy is option2; this file = fallback if DPO insists on redaction, or source of concession rungs.

## Legal standard — counter DPO's absolutism
DPO letter: "any information that could directly or indirectly identify... any other contextual details."
- Actual test (Rec. 26): identifiability by "means **reasonably likely** to be used" — relative, practical, not metaphysical
- *Breyer* C-582/14 (CJEU 2016): dynamic IP = personal data for website because ISP linkage was reasonably available; where linkage requires disproportionate/impracticable/illegal means → NOT identifiable. Identifiability is contextual.
- WP29/EDPB three-risk test: **singling-out**, **linkability**, **inference** — assess each separately
- Deliverable = anonymization **dossier** (techniques + adversary model + measured residual risk + DPO review log), not metaphysical proof

## Core technical argument (randomized-response logic)
- **RR (randomized response)**: survey trick — coin flip decides truthful vs forced answer → individual answers unconfirmable, population stats valid
- Our version: PII-model flags msgs → whole-msg obfuscation w/ plausible-false decoys → reID requires confirmation; confirmation requires ground truth; ground truth provably corrupted → inference falls below "reasonably likely means"
- **DP (differential privacy)**: formal version — output barely changes whether any record included → membership unconfirmable
- Covers nth-order OTHERs (mentioned persons): decoys corrupt their references equally; no confirmation path
- Vec2Text rebuttal: purpose-built inverter, 2 models, train-data leakage, accuracy collapses >32 tokens → inversion output unverifiable = plausible deniability

## Redaction ladder (rung-by-rung concessions w/ fidelity cost)
- R0: header-mask names → SUBJECT/OTHER (our default, already in option2)
- R1: consistent pseudonyms for identifiers (keeps coreference; keeps intra-corpus linkability)
- R2: chunk-varying pseudonyms (kills cross-corpus linkability)
- R3: decoy injection on PII-flagged msgs (RR plausible deniability)
- R4: full-msg obfuscation/paraphrase of flagged msgs (bottom-up seq2seq/txt2vec2txt)
- R5: distilled context labels only {topic, sentiment, relation, timestamp} — last resort

Pick gentlest rung DPO accepts; quantify fidelity cost per rung (pilot only if forced past R2).

## Methods
- PII detection: Presidio > spaCy NER > Flair; Faker for plausible replacements
- Models: bardsai/eu-pii-anonimization-multilang(-v2), tabularisai/eu-pii-safeguard, flowxai/piiguard, perplexity pplx-pii-masking, Roblox pii-classifier-v2, agentlans multilingual-e5/embeddinggemma PII detectors
- Replacement: redact `(NAME)` / consistent `(NAME:1)` / culturally-adjacent decoy

## Compliance artifact (motivated-intruder test)
Can't prove negative → characterize strongest reasonably-likely attack:
1. Adversary model (resources, auxiliary data: LinkedIn, press, corp filings)
2. Test singling-out / linkability / inference separately
3. Show obfuscation creates *crowds* (generalization), not injective renames
4. Cover mentioned persons too (DPO explicitly flagged)
5. Documented tests → DPO reviews; Datatilsynet/courts decide only if challenged

## Cost
Heavy redaction degrades imitation signal → if forced here, pilot fidelity pre/post per rung; numbers double as DPIA proportionality evidence and thesis experiment.
