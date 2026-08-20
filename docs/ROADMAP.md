# Roadmap

Phased plan. Each phase produces something runnable; nothing depends on a later phase to be demonstrable.

## Phase 0 — Feasibility (single machine) ✅ code complete

The riskiest assumption tested first: *can we split an MLX model by layer range and hand hidden states across a serialization boundary without changing the output?*

- [x] Shard-aware Qwen2 port with `forward` over an arbitrary layer range (`Qwen2Shard`)
- [x] Snapshot loader that materializes only a shard's weights, with checkpoint quantization (`ShardLoader`)
- [x] Versioned binary wire format for hidden states (`TensorFrame` + tests)
- [x] Greedy pipeline generation with pluggable shard-boundary links (`PipelineGenerator`)
- [x] `pipeline-proof` executable: unsharded vs. two-shard + codec round-trip, token-exact comparison
- [ ] **Run the proof on real hardware** (first `swift run` — see [SETUP.md](SETUP.md))
- [ ] Baseline numbers: tok/s for 0.5B and 3B on the Mac, recorded in this file

Fallback if MLX sharding had failed: llama.cpp's RPC backend with a Swift wrapper. Not needed — the shard path is implemented.

## Phase 1 — Two devices, one cable of thought (~1 week)

- [x] Bonjour advertise/browse (`PeerAdvertiser`, `PeerBrowser`)
- [x] Length-prefixed TCP frame connection (`FrameConnection`)
- [ ] `edge-worker` CLI: load a shard, serve it over a `FrameConnection` (Mac ↔ Mac first — two processes, then two machines)
- [ ] Coordinator handshake: hello / shard-assignment / ready control frames
- [ ] End-to-end: Mac (layers 0..k) + second device (k..n), full answer streamed
- [ ] Compare cross-device output against single-device reference (text-level)
- [ ] iOS worker app skeleton (SwiftUI, `Info.plist` local-network keys, increased-memory entitlement)

## Phase 2 — Three devices, real target model (~1 week)

- [ ] Memory reporting (`os_proc_available_memory` on iOS) → `ShardSpec.split` allocation
- [ ] 7B–8B 4-bit model across iPhone + iPad + Mac
- [ ] Snapshot presence check per device; clear instructions when a shard's weights are missing
- [ ] Clean failure on peer loss + automatic cluster re-form
- [ ] Keep-awake handling (`isIdleTimerDisabled`) for demo sessions

## Phase 3 — Demo polish (~3–4 days)

- [ ] Cluster visualization: per-device layer range, ms/token, animated token flow
- [ ] Side-by-side screen: single-device OOM vs. cluster answering (OOM shown on the *phone*, honestly)
- [ ] 60–90 s demo video, README GIF, blog post

## Later / ideas

- Chunked prefill for long prompts
- Peer authentication (PSK → TLS) before any non-demo use
- QUIC transport experiment
- More model families (llama, qwen3) — `ShardLoader` is the only qwen2-specific surface besides the module itself
- Speculative decoding with a small on-coordinator draft model (turns pipeline latency into an advantage)
