# LFM2.5-8B-A1B Q4_K_M — B200 MIG 1g.23gb

Model fully GPU-resident. llama.cpp `95b8e33`, CUDA 13.0/SM100, FA on, unified KV + continuous batching, 6 CPU threads, context pool 128K, no prompt cache. Mixed16 = 16 independent length-spread V7 chats, 184,200 tokenized input tokens. Reasoning workload forces 512 output tokens/request (8,192 total).

## Reasoning / ubatch
| ubatch | C | wall s | input tok/s | output tok/s | p95 s | success |
|---:|---:|---:|---:|---:|---:|---:|
|512|1|99.713|1847.30|82.16|12.60|16/16|
|1024|1|90.746|2029.85|90.27|10.95|16/16|
|2048|1|86.966|2118.06|94.20|10.25|16/16|
|2048 validation|1|86.985|2117.59|94.18|10.26|16/16|

2048 vs 512: 12.8% lower wall time, 14.7% higher aggregate rate. 2048 vs 1024 gain 4.16% (<5% stopping threshold). 1024 = safer runner-up if future concurrency requires workspace/KV headroom.

## Concurrency
Reasoning ub512: C1 16/16, C2 16/16 but same throughput + much worse p95, C4 12/16. Mixed-real short output ub512: C1 best; C8 only 8/16. Production-safe result for heterogeneous long requests: C1. Dynamic/unified KV does not remove finite context-pool capacity.

Rates overlap prefill/decode and are workload aggregate, not isolated llama-bench PP/TG.
