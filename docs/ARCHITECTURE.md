# Architecture

Deep dive into how edge-llm-cluster splits one transformer across many devices. For the phased plan see [ROADMAP.md](ROADMAP.md); for the byte level wire contract see [PROTOCOL.md](PROTOCOL.md).

## The problem shape

Autoregressive LLM inference is memory bound: the model weights must be resident, and each generated token touches all of them once. A 4-bit 7B model (about 4 GB of weights plus KV cache and activations) is at or over the per app memory ceiling of most phones, comfortable on a modern laptop, and far beyond older devices, while the sum of a household's devices has memory to spare.

Pipeline parallelism fits this shape exactly:

- **Memory splits.** Each device holds only its contiguous layer range, plus the embedding on the first shard and the head on the last.
- **Traffic stays tiny.** One hidden state vector per shard boundary per token.
- **Latency adds, bandwidth does not.** Per token the wire cost is one hidden state hop per boundary plus one token id hop back from the sampling shard to the embedding shard. With N devices that is N hops of a few milliseconds each on local Wi-Fi.

Tensor parallelism was rejected: it requires collective communication inside every layer (dozens of round trips per token per layer), which Wi-Fi latency destroys.

## System overview

```
 workers (any platform, any count)          coordinator (currently macOS)
┌──────────────────────────────┐          ┌──────────────────────────────┐
│ iPhone   MLX     layers 0..k │◄────────►│  discovery (mDNS)            │
│ Pixel    llama.cpp k..m      │  frames  │  layer allocation            │
│ iPad     MLX     m..p        │          │  chat loop + sampling        │
│ old Mac  MLX     p..n        │          │  cluster state               │
└──────────────────────────────┘          │  web dashboard (HTTP)        │
                                          └──────────────────────────────┘
```

The protocol between the two sides is runtime neutral. Apple workers use MLX; a llama.cpp based worker (the path for Android and Linux) speaks the same frames. There is no fixed device count: allocation is proportional and recomputed from whoever is connected.

## Components

### 1. Shard model (`EdgeLLMKit/Qwen2Shard`)

A minimal, shard aware port of the Qwen2 architecture, weight compatible with `mlx-community` `qwen2` checkpoints:

| Shard position | Owns |
|---|---|
| first (`start == 0`) | token embedding plus layers `0..<end` |
| middle | layers `start..<end` only |
| last (`end == n`) | layers plus final RMSNorm plus LM head |

Tied embedding checkpoints (Qwen2.5 0.5B/1.5B/3B) have no separate `lm_head` tensor, so the last shard loads its own copy of the embedding table and uses it as the output projection. This duplication is inherent to splitting tied models and is counted in memory planning.

`ShardLoader` reads the standard Hugging Face snapshot (`config.json` plus `*.safetensors`), keeps only the tensors the shard owns, rebases global keys (`model.layers.17.…` becomes `layers.0.…` for a shard starting at 17), applies the checkpoint's group wise quantization to exactly the modules that carry `scales`, and verifies the load is exact. Weights outside the shard are never materialized.

### 2. Generation loop (`EdgeLLMKit/PipelineGenerator`)

Greedy decoding across a shard chain. The hand-off between consecutive shards is a pluggable `Link`:

- `directLink`: in memory, used by reference runs
- `codecLink`: full serialize to bytes and back through the wire format, used by the Phase 0 proof
- the networked runtime replaces the link with an actual socket; the loop itself does not change

Each shard creates its own KV cache per layer. **Caches never cross the link.** This is what keeps per token traffic constant regardless of context length.

Prefill: the whole prompt goes through the pipeline once as a `[1, prompt_len, hidden]` tensor (a few MB, one time). Decode: `[1, 1, hidden]` per token.

### 3. Wire format (`EdgeLLMKit/TensorFrame`)

Fixed little endian binary framing, version tagged, fully specified in [PROTOCOL.md](PROTOCOL.md). Three frame kinds: hidden states, token ids, and JSON control messages (hello, assignment, metrics, goodbye). Hidden states travel as float16: `hidden_dim x 2` bytes per decode step (Qwen2.5-3B: 4 KB; 7B or 8B class: 7 to 8 KB). Decoding validates magic, version, rank and payload size against shape; a malformed or truncated frame is rejected, never partially applied.

On TCP each frame is preceded by a `UInt32` length prefix (`EdgeLLMTransport/FrameConnection`).

### 4. Discovery and transport (`EdgeLLMTransport`)

- `PeerAdvertiser`: listener advertising `_edgellm._tcp` over mDNS/DNS-SD, wrapping accepted connections in `FrameConnection`
- `PeerBrowser`: browser for the same service type; connects to discovered peers
- TCP first (predictable, ordered); QUIC is a possible later experiment

mDNS exists everywhere the project targets: Bonjour on Apple platforms, `NsdManager` on Android, Avahi on Linux. Apple app targets must declare `NSBonjourServices` (`_edgellm._tcp`) and `NSLocalNetworkUsageDescription` in Info.plist, or iOS blocks local network traffic.

### 5. Coordinator and dashboard (`EdgeLLMCoordinator`, `EdgeCoordinatorCLI`)

The coordinator:

1. advertises the service and accepts worker connections from any platform
2. collects `hello` reports (id, name, platform, runtime, total and usable memory)
3. computes the layer split with `ShardSpec.split(totalLayers:weights:)`, proportional to usable memory with a minimum of one layer per device, and sends each worker its `assignment`
4. runs the chat loop: tokenize, drive the pipeline, sample on the last shard
5. tracks everything in `ClusterState` and serves the web dashboard

The dashboard is a single embedded HTML page served by a dependency free HTTP server (`StatusServer`) with two routes: `/` for the page and `/api/status` for the JSON snapshot. The page polls once per second and renders one card per device: platform and runtime, memory quota bar (used versus quota, with warning colors), assigned layer range, per token latency and online state, plus cluster level tokens per second. Any browser on the LAN can open it, phones included. `edge-coordinator --demo` seeds simulated devices for a quick look.

## Memory math (worked example)

Qwen2.5-7B-Instruct 4-bit, about 4.2 GB of weights, 28 layers, hidden 3584:

| Device | Usable memory (approx.) | Layers | Shard weight |
|---|---|---|---|
| iPhone (8 GB) | 4.5 GB (jetsam ceiling, with entitlement) | 6 | 0.9 GB |
| Pixel 8 (8 GB) | 4 GB | 6 | 0.9 GB |
| iPad (8 GB) | 4.5 GB | 6 | 0.9 GB |
| MacBook (16 GB) | 10 GB | 10 | 1.5 GB plus head |

Per token wire traffic: 3 hidden state hops of 7 KB plus a 4 byte token id, about 21 KB per token. That is well under 1 ms of bandwidth on any modern Wi-Fi; the cost that matters is per hop latency (1 to 5 ms times 4 hops).

Platform memory notes: iOS gives an app only part of physical RAM (jetsam); budget from `os_proc_available_memory()` and ship the `com.apple.developer.kernel.increased-memory-limit` entitlement. Android's per app limits vary by device; the worker reports its own `usableMemory` and the coordinator plans around whatever is offered.

## Mixed runtime correctness

- **Phase 0 (in repo):** unsharded versus sharded with serialization on the same machine must match token for token at temperature 0, since the same GPU accumulates float16 identically.
- **Cross device, same runtime:** different GPUs may round differently, occasionally flipping an argmax. Runs are validated for text level agreement against a single device reference.
- **Cross runtime (MLX plus llama.cpp):** both compute the same architecture from the same base model, but quantization formats differ (MLX group wise int4 versus GGUF k-quants), so logits diverge more than in the same runtime case. The contract at the boundary stays exact float16 hidden states; validation is text level agreement and perplexity sanity, never bitwise.

## Failure handling (Phase 2 target)

Minimum viable: a dropped connection surfaces a clean error, the worker disappears from the dashboard, and the cluster re-forms via discovery. Generation restarts from the prompt, because KV caches are device local and lost with the device. Mid generation shard migration is out of scope for v1.

## Security posture

v0 and v1 target a trusted local network: frames are unauthenticated and unencrypted, and the service accepts any peer advertising the right service type. Do not run it on networks you do not control. Peer authentication (a pre-shared key first, then TLS) is on the roadmap before any non-demo use. See [SECURITY.md](../SECURITY.md).
