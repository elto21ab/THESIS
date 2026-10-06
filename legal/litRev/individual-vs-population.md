# Individual vs Population — citation bank (for Daniel)

## The defensible claim
Population-level results are a **sampling convenience**, not the scientific ideal. For non-ergodic processes (i.e. human behavior), group-level statistics **do not generally transport to individuals** (Molenaar 2004; Fisher et al. 2018). When the action/decision targets the individual, individual-level modeling is required. And LLM-based *individual* simulation is an **open frontier** (SOTA models still fail at person-level behavior chains) — that's our niche.

## References (all verified, w/ source links)

| # | Ref | Link | What it shows |
|---|---|---|---|
| 1 | Molenaar 2004, "A Manifesto on Psychology as Idiographic Science" | https://www.tandfonline.com/doi/abs/10.1207/s15366359mea0204_1 | Group stats assume ergodicity; psych processes are non-ergodic → nomothetic (population) methods can't describe individuals; individual-level analysis required |
| 2 | Fisher, Medaglia & Jeronimus 2018, "Lack of group-to-individual generalizability is a threat to human subjects research" | https://www.pnas.org/doi/10.1073/pnas.1711978115 | Across 6 intensive repeated-measures studies: inter-individual (group) correlations rarely generalize to intra-individual (person) level. Empirical, not just theoretical |
| 3 | Peng et al. 2025, "Twin-2K-500" | https://arxiv.org/abs/2505.17479 · https://pubsonline.informs.org/doi/10.1287/mksc.2025.0262 | N=2,058 × 500 questions × 4 waves of **individual-level** ground truth, built explicitly for LLM digital twins; goal: predict behavior at individual AND aggregate levels. Field invests in individual data *because* aggregate doesn't suffice |
| 4 | Binz et al. 2025, "Centaur: a foundation model of human cognition" | https://www.nature.com/articles/s41586-025-09215-4 · https://arxiv.org/abs/2410.20268 | Foundation model fine-tuned on Psych-101 (individual participants' choices) to predict/simulate human behavior in experiments — individual-level prediction as a research frontier |
| 5 | Wang et al. 2024, "Building personalized ML models... idiographic suicidal thoughts" | https://doi.org/10.1038/s44220-024-00335-w | N=89 patients, EMA; personalized (idiographic) models predict *which patient, when*. Clinical action is individual-level — population models can't answer it |
| 6 | "Personalized vs. population-based speech models for multi-dimensional mental health prediction" (2025) | https://pmc.ncbi.nlm.nih.gov/articles/PMC13287071/ | Population-only models struggle (speaker-specific variance); personalization beat population-level across depression/anxiety/stress (RMSE 6.95/7.15/4.95, DASS-21) |
| 7 | "From one, many: How can nudges be personalized?" (PeN review, 2025) | https://journals.sagepub.com/doi/10.1177/23794607251403327 | Personalized nudging promises greater effectiveness, *especially for heterogeneous populations*; one-size-fits-all underperforms |
| 8 | Li et al. 2025, "How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation" (BehaviorChain) | https://aclanthology.org/2025.findings-acl.813/ | 15,846 behaviors / 1,001 personas benchmark; **SOTA models struggle** at continuous person-level simulation → open gap |
| 9 | Argyle et al. 2023, "Out of One, Many" | https://arxiv.org/abs/2209.06899 | The population-level baseline ("silicon sampling"): simulates *distributions*, not specific individuals; algorithmic fidelity defined at group level |

## Debate-ready one-liners
- "Population = lower resolution by necessity, not by design. Full-resolution individual data dominates whenever the decision acts on the individual." (Molenaar/Fisher)
- "Not just intuition — PNAS 2018 quantified that group correlations generally don't hold for individuals."
- "The field agrees it's the frontier: Twin-2K-500 was built to enable individual-level digital twins; BehaviorChain (ACL'25) shows today's best models still can't do it."
- "Already superior in applied settings: personalized speech models beat population models on individual-level prediction (mental health)."

## Where individual-level bites (concrete domains)
- LLM digital twins / persona simulation (our space)
- Precision & idiographic mental health (patient-level, moment-level)
- Personalized nudging / behavioral interventions (heterogeneous effects)
- Individual decision-maker simulation (firms, negotiators, traders)
- Consumer/marketing individual choice (Twin-2K-500, heterogeneous treatment effects)

## Notes / traps
- Don't claim "individual is always superior" — claim: *population findings don't transport to individuals (non-ergodicity), so for individual-targeted goals the population method is systematically lossy.*
- Don't overclaim novelty: Twin-2K-500 = survey answers as ground truth, no free-text context; our gap = **chat/free-text context for individual imitation** — verify wording w/ Airidas.
- Centaur predicts behavior (choices), not survey-text imitation — adjacent, cite carefully as "individual-level behavioral prediction frontier".
- Direction of translation (for mail wording): **individual → population is possible (aggregation); population → individual is not** (the non-ergodicity point). Write it in that direction.
