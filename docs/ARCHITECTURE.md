# Architecture

Deep-dive into how edge-llm-cluster splits one transformer across several Apple devices. For the phased plan see [ROADMAP.md](ROADMAP.md).

## The problem shape

Autoregressive LLM inference is memory-bound: the model weights must be resident, and each generated token touches all of them once. A 4-bit 7B model (~4 GB weights + KV cache + activations) is at or over the per-app memory ceiling of most iPhones, comfortable on a modern Mac, and far beyond older devices — while the *sum* of a household's devices has memory to spare.

Pipeline parallelism fits this shape exactly:

- **Memory splits**: each device holds only its contiguous layer range (plus the embedding on the first shard and the head on the last).
- **Traffic stays tiny**: one hidden-state vector per shard boundary per token.
- **Latency adds, bandwidth doesn't**: per token the wire cost is `(#shards − 1)` hidden-state hops **plus one token-id hop** back from the sampling shard to the embedding shard. With 3 devices that is 3 hops ≈ a few ms each on local Wi-Fi.

Tensor parallelism was rejected: it requires collective communication inside every layer (dozens of round-trips per token per layer), which Wi-Fi latency destroys.

## Components

### 1. Shard model — `EdgeLLMKit/Qwen2Shard`

A minimal, shard-aware port of the Qwen2 architecture (weight-compatible with `mlx-community` `qwen2` checkpoints):

| Shard position | Owns |
|---|---|
| first (`start == 0`) | token embedding + layers `0..<end` |
| middle | layers `start..<end` only |
| last (`end == n`) | layers + final RMSNorm + LM head |

Tied-embedding checkpoints (Qwen2.5 0.5B/1.5B/3B) have no separate `lm_head` tensor — the **last shard loads its own copy of the embedding table** and uses it as the output projection. This duplication is inherent to splitting tied models and is counted in memory planning.

`ShardLoader` reads the standard Hugging Face snapshot (`config.json` + `*.safetensors`), keeps only the tensors the shard owns, rebases global keys (`model.layers.17.…` → `layers.0.…` for a shard starting at 17), applies the checkpoint's group-wise quantization to exactly the modules that carry `scales`, and verifies the load is exact. Weights outside the shard are never materialized.

### 2. Generation loop — `EdgeLLMKit/PipelineGenerator`

Greedy decoding across a shard chain. The hand-off between consecutive shards is a pluggable `Link`:

- `directLink` — in-memory (reference runs),
- `codecLink` — full serialize → bytes → deserialize round-trip through the wire format (Phase 0 proof),
- Phase 1 replaces the link with an actual socket; the loop itself does not change.

Each shard creates its own `KVCacheSimple` per layer. **Caches never cross the link** — this is what keeps per-token traffic constant regardless of context length.

Prefill: the whole prompt goes through the pipeline once as a `[1, prompt_len, hidden]` tensor (a few MB, one-time). Decode: `[1, 1, hidden]` per token.

### 3. Wire format — `EdgeLLMKit/TensorFrame`

Fixed little-endian binary framing, version-tagged:

```
offset  size  field
0       4     magic "ELC1"
4       1     version = 1
5       1     kind      0 hiddenState · 1 token · 2 control
6       1     dtype     0 f16 · 1 f32 · 2 bf16
7       1     rank
8       4     seqPos
12      2     layerStart      (producing shard)
14      2     layerEnd
16      4·r   shape dims (Int32)
…       4     payload byte count
…       …     payload
```

Hidden states travel as float16: `hidden_dim × 2` bytes per decode step (Qwen2.5-3B: 4 KB; 7–8B class: 7–8 KB). Decoding validates magic, version, rank, and payload size against shape — a malformed or truncated frame is rejected, never partially applied.

On TCP the frame is preceded by a `UInt32` length prefix (`EdgeLLMTransport/FrameConnection`).

### 4. Discovery & transport — `EdgeLLMTransport`

- `PeerAdvertiser`: `NWListener` advertising `_edgellm._tcp` over Bonjour, wrapping accepted connections in `FrameConnection`.
- `PeerBrowser`: `NWBrowser` for the same service type; connects to discovered peers.
- TCP first (predictable, ordered); QUIC is a possible later experiment.

App targets must declare `NSBonjourServices` (`_edgellm._tcp`) and `NSLocalNetworkUsageDescription` in Info.plist — iOS blocks local-network traffic without them.

### 5. Coordinator (Phase 2)

Any device (typically the Mac) acts as coordinator:

1. browse peers, collect `{device, usable memory, chip}` reports,
2. compute the layer split with `ShardSpec.split(totalLayers:weights:)` — proportional to *usable* memory with a per-shard minimum, embedding/head overhead weighted in,
3. verify each peer has its weight slice (transfer if missing — future work; v0 assumes the full snapshot is present on every device),
4. run the chat loop: tokenize on the coordinator, drive the pipeline, sample on the last shard, stream tokens to the UI.

## Memory math (worked example)

Qwen2.5-7B-Instruct-4bit, ~4.2 GB weights, 28 layers, hidden 3584:

| Device | Usable memory (approx.) | Layers | Shard weight |
|---|---|---|---|
| iPhone (8 GB) | ~4.5 GB (jetsam ceiling, with entitlement) | 8 | ~1.2 GB |
| iPad (8 GB) | ~4.5 GB | 8 | ~1.2 GB |
| Mac (16 GB) | ~10 GB | 12 | ~1.8 GB + head |

Per-token wire traffic: 2 × 7 KB hidden states + 4 B token id ≈ **14 KB/token** — under 1 ms of bandwidth on any modern Wi-Fi; the cost that matters is per-hop latency (~1–5 ms × 3 hops).

iOS specifics: apps get only part of physical RAM (jetsam). Budget conservatively from `os_proc_available_memory()`, and ship the `com.apple.developer.kernel.increased-memory-limit` entitlement.

## Correctness strategy

- **Phase 0 (in-repo)**: unsharded vs. sharded-with-serialization on the same machine must match token for token at temperature 0 — same GPU, so float16 accumulation is identical.
- **Cross-device**: different GPUs may round differently, occasionally flipping an argmax. Cross-device runs are validated for text-level agreement and perplexity sanity, not bitwise equality.

## Failure handling (Phase 2 target)

Minimum viable: a dropped connection surfaces a clean error and the cluster re-forms via Bonjour; generation restarts from the prompt (KV caches are device-local and lost with the device). Mid-generation shard migration is out of scope for v1.

## Security posture

v0/v1 target a **trusted local network**: frames are unauthenticated and unencrypted, and the service accepts any peer advertising the right Bonjour type. Do not run it on networks you don't control. Peer authentication (pre-shared key → TLS-PSK on the TCP connection) is on the roadmap before any non-demo use.
