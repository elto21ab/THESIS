Subject: GDPR path forward

Hi Daniel,

TL;DR:
- DPO's initial 2 options: consent from everyone or full anonymisation.
- That binary followed from a wrong premise (students = controllers), not a misreading of the law. Corrected: this is university research — university = controller (we are named processors under its safeguards).

Reason for the research basis (no consent required):
- Danish Data Protection Act (DBL) §10 + GDPR Art. 6(1)(e) + 9(2)(j): research does not require consent, not even for special categories.
- Art. 14(5)(b): individual notice to chat partners is exempt when disproportionately burdensome — exactly our situation.
- In return we must deliver safeguards (Art. 89(1)): pseudonymisation, access restriction, DPIA, deletion schedule — all of which we already have.

Path A: research basis under the university. We re-open the dialogue with the DPO on the corrected premise (university = controller, you = project responsible), present the legal basis + DPIA as established fact (required to present DPIA to DPO), and ask for her comments — not her permission (her role is advisory, no veto, Art. 38(3)). All signal preserved; nothing burned.

What we need from you:
1. Confirm you back the primary route
2. Sign on as project responsible (DPIA co-signer, Art. 30 record), to formally establish the university as the controller. (uCloud ownership transfer is a reserve action if the DPO requires it.)

Backup if the DPO still rejects Path A.
Path C: full anonymisation. If the basis is rejected, we can anonymise data at collection → data falls outside GDPR entirely (Recital 26 + betænkning 1565). Cost: unknown amount of [irreversible] signal loss.

Path C options, gentlest → hardest:
0. header-mask names → SUBJECT/OTHER (our default = Path A)

Detection of private data: REGEX header names < NER (all "primary key" that can be used to ID a person alone) < PII (every piece of quasi-ID that can be used to ID a person in combination with other data).

Options below use PII to detect and replace them with _:
1. generic mask → <NAME>
2. consistent mask → <NAME:1> represents John Doe, <NAME:2> represents Peter Pan, across all chunks.

Decoy injection on PII-flagged messages → adds plausible false decoys to make re-identification harder.
3. consistent pseudonyms for identifiers → John Doe
4. chunk-varying pseudonyms → John Doe in one chunk, Peter Pan in another 

Decoy entire message or sentence WHERE PII FLAGED >0 WORDS. These options can build on top of the above options (to completely obscure the original content), or be used in isolation. Here we can use a dec-only, or a enc-dec (seq2seq/txt2vec2txt).
5. paraphrase
5a. Naive paraphrase w/o checking if the PII detected is still present (plausible deniability)
5b. regex block PII token[s] from output (percentage-based activation; not all the time)
5c. regex block PII token[s] from output (every time)

Best, Elias
