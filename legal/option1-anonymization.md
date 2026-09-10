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
- R4: obfuscation/paraphrase of flagged msgs/words (bottom-up seq2seq/txt2vec2txt)
- R4a: PII only flags
- R4b: PII consistent pseudonym
- R4c: PII consistent mask
- R4d: PII flag w/ regex blocking same token output (only at certain percentage of cases)
- R4e: PII flag w/ regex blocking same token output (every time)
- ~~R5: distilled context labels only {topic, sentiment, relation, timestamp} — last resort~~

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

## DPO strategy at full anonymisation: don't ask — minimal inform
- Full DP anonymisation → outside GDPR scope ([Rec. 26](https://gdpr-info.eu/recitals/no-26/); betænkning 1565 confirms "herunder til forskningsmæssige formål") → nothing for DPO to approve
- **Don't ask.** Asking a non-technical DPO invites an uninformed veto on a legally moot question → burns the position for no gain
- **Do** one-line written inform, zero questions: *"Pipeline applies differential-privacy anonymisation at ingestion; output non-reidentifiable → outside GDPR scope."* Records good faith, draws no ruling
- Burden of proof judged objectively (Rec. 26 "reasonably likely means") → keep technical dossier (ε guarantee, threat model) on file; defense = method rigor, not DPO sign-off
- **Gate before relying on this path:** anonymisation must happen at ingestion. Any stage touching raw PII first = that window is personal data → needs basis for that stage

## Cost
Heavy redaction degrades imitation signal → if forced here, pilot fidelity pre/post per rung; numbers double as DPIA proportionality evidence and thesis experiment.
