# Experiment index

| # | Date | UCloud job | Model / quant | Engine | Verdict | Docs |
|---|------|-----------|---------------|--------|---------|------|
| 1 | 2026-08-19 | 12372444 (llama-inference) | Qwen3.6-35B-A3B (MoE) | llama.cpp | llama.cpp viable; router-aware `-ngl` offload proven | [dir](2026-08-19_llama_first/) |
| 2 | 2026-08-26 | vllm-sglang-offload-test | LFM2.5-8B-A1B (bf16/fp8/NVFP4/Q4_K_M) | vLLM vs SGLang vs llama.cpp | vLLM 2× SGLang @ c32; fp8 +25% tg; NVFP4 blocked by SGLang hang + vLLM #40885; offload any % kills throughput | [dir](2026-08-26_lfm25_sweep/) · [results](../results/RESULTS_8B.md) |
| 3 | 2026-08-28 | 12375523 | Qwen3.8-27B-NVFP4 (Inferact vs unsloth) | vLLM | **STOPPED, 0 benchmarks** — full postmortem + checklist inside | [dir](2026-08-28_qwen38_nvfp4/) |

Settled decisions: engine = vLLM ≥0.27 on Blackwell; no weight offload in vLLM; NVFP4 = target quant; GGUF only for llama.cpp path.
