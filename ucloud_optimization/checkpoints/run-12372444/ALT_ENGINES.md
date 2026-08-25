# Alternative engine smoke results
- Artifact `sakamakismile/LFM2.5-8B-A1B-NVFP4`: full HF repo downloaded (weights, config, tokenizer, chat template, `hf_quant_config.json`, run docs); ModelOpt 0.44.0 NVFP4 W4A4. Not generated locally. Unsloth publishes no LFM2.5 NVFP4 target repos as of run date.
- vLLM 0.27.1: fails weight load: merged-column tensor shape assertion. Artifact docs target vLLM 0.21.0; retest pinned 0.21.0.
- SGLang 0.5.18: reaches scheduler then causal_conv1d JIT rejects BF16 (allows FP16 only). Retest once with `--dtype float16`; otherwise treat unsupported.
- Neither alt engine reached serving state, so no concurrency/KV behavior measured.
