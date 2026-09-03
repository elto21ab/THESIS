# Response to DPO letter — core counter-framing

## Their binary is incomplete — the omitted third path
DPO: "students have no independent legal basis → (1) full anonymization or (2) consent from all."

**Counter**: students don't *need* independent basis. Processing occurs within university research activity, supervisor = PI, **university = controller** (Art. 4(7)). Then:
- Art. 6(1)(e) task in public interest (research) — consent not required
- Art. 9(2)(j) + Databeskyttelsesloven §10 — special categories in research, consent not required
- Art. 14(5)(b) — individual notice to OTHERs exempt (disproportionate effort / seriously impairs objectives)
- Art. 21 objection → handled ongoing via erasure, no pre-processing "window"
- Art. 89(1) safeguards: pseudonymization, minimization, access control, destruction, DPIA

This is the standard architecture for register/chat-data research in DK. Datatilsynet guidance recognizes §10 processing w/o consent precisely because donor-mediated third-party consent is structurally infeasible + selection-biasing.

**Ask DPO**: "Why does §10/9(2)(j) + 14(5)(b) not apply here? What safeguard would make it sufficient?" Force them to argue against the research regime, not against consent absence.

## Their anonymization demand is overbroad
Letter: "any information that could directly or indirectly identify... any other contextual details."
- Legal test (Rec. 26): identifiability by "means **reasonably likely** to be used" — relative standard, not absolute (confirmed: *Breyer* C-582/14; EDPB/WP29 anonymization guidance: risk-based, three tests: singling-out, linkability, inference)
- "Could conceivably identify" ≠ "reasonably likely to identify". Operationalize via **motivated-intruder test w/ documented adversary model** — the accepted method.
- Deliverable: anonymization dossier (techniques + adversary model + measured residual risk), signed off by DPO — not a metaphysical proof.

## Consent-all path is defective on its own terms
- Donor-mediated consent collection from 10²–10³ contacts/donor → cohort collapse (already observed recruitment friction — document this!)
- Selection bias: consenting friends ≠ representative → validity threat DPO should care about as research institution
- Explicit Art. 9(2)(a) consent standard per contact is impracticable → the regime itself (9(2)(j)) exists for exactly this
- Therefore consent path "seriously impairs objectives" → which is *also* the 14(5)(b) trigger. Their Option 2's infeasibility is evidence for our Option 3.

## On the "service like OpenAI" idea — DO NOT RAISE
- ToS checkbox = donor's consent only; legally void for OTHERs' data (consent is personal, Art. 7)
- OpenAI is its own controller w/ own basis; the analogy imports their contested status, not an exemption
- Retaining data for research purposes → we determine purposes → controller regardless of framing
- Only salvageable: donor-side pre-processing script (household step) as pipeline front-end

## Supervisor subproject — yes, but reframe
Not "siloed access → household". Instead: supervisor-anchored project → university controllership → §10/9(2)(j) unlocked. Access siloing (supervisor-only) = strong Art. 89 safeguard to offer as concession.
