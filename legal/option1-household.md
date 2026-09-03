# Option 1: Household Exemption — WEAK as primary

## Claim
Donor processes own data (Art. 2(2)(c) purely personal activity); we act as mere processor providing compute (uCloud API/SSH), never controller.

## Why it likely fails DPO review
1. **Exemption attaches to the *individual's* processing, not ours.** Donor exporting own chats at home = exempt. Institution ingesting them into a research pipeline = new processing, new controller (us). EDPB: exemption assessed per-processing-activity.
2. **Controller = whoever determines purposes & means (Art. 4(7)).** We define research question, pipeline, models, retention → we are controller. Calling donors "controllers" inverts the test. Processor acts *on documented instructions* (Art. 28); donors give no instructions, they donate.
3. **"No storage on our server" irrelevant.** Controllership follows purpose/means determination, not bytes-at-rest. Inference-only processing is still processing (Art. 4(2)).
4. **Joint controllership risk (Art. 26).** Even if donor has some role, we'd be joint controllers → still full obligations.
5. Institutional prerequisites (PI liability insurance, §10 standing, continuity) don't cure the structural inversion.

## Salvageable part
- Household framing **valid for the collection step**: donor exports/sanitizes own data on own device before upload. Use as *front-end* of Option 2/3 (donor-side pre-processing script), not as the legal basis for our research processing.
- ZDR (zero data retention) / inference-only = good Art. 32/25 *measure*, not a scope escape.

## If DPO asks "could donors just run it themselves?"
Only true if we genuinely never touch data (donor-run pipeline, we receive aggregate stats). That kills the corpus study. Don't pretend otherwise.
