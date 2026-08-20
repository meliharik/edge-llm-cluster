# Roadmap

Phased plan. Each phase produces something runnable; nothing depends on a later phase to be demonstrable.

## Phase 0: feasibility on a single machine (done)

The riskiest assumption tested first: can we split an MLX model by layer range and hand hidden states across a serialization boundary without changing the output?

- [x] Shard aware Qwen2 port with forward over an arbitrary layer range (`Qwen2Shard`)
- [x] Snapshot loader that materializes only a shard's weights, with checkpoint quantization (`ShardLoader`)
- [x] Versioned binary wire format for hidden states (`TensorFrame` plus tests)
- [x] Greedy pipeline generation with pluggable shard boundary links (`PipelineGenerator`)
- [x] `pipeline-proof` executable: unsharded versus two shards plus codec round trip, token exact comparison
- [ ] Run the proof on real hardware (first `swift run`, see [SETUP.md](SETUP.md))
- [ ] Baseline numbers: tok/s for 0.5B and 3B on a Mac, recorded in this file

## Phase 1: first real cluster (in progress)

- [x] mDNS advertise and browse (`PeerAdvertiser`, `PeerBrowser`)
- [x] Length prefixed TCP frame connection (`FrameConnection`)
- [x] Control protocol: hello, assignment, metrics, goodbye (`ControlMessage`)
- [x] Language agnostic protocol specification ([PROTOCOL.md](PROTOCOL.md))
- [x] Cluster state registry and web dashboard (`EdgeLLMCoordinator`, `edge-coordinator`)
- [ ] `edge-worker` CLI: load a shard, serve it over a `FrameConnection` (two processes on one Mac first, then two machines)
- [ ] Wire the coordinator chat loop to remote shards through the metrics reporting path
- [ ] End to end: Mac plus a second device, full answer streamed, visible live in the dashboard
- [ ] iOS worker app skeleton (SwiftUI, Info.plist local network keys, increased memory entitlement)

## Phase 2: many devices, real target model

- [ ] Memory reporting on each worker feeding `ShardSpec.split` allocation
- [ ] A 7B or 8B 4-bit model across three or more devices
- [ ] Arbitrary device counts: join and leave while idle, allocation recomputed
- [ ] Snapshot presence check per device; clear instructions when a shard's weights are missing
- [ ] Clean failure on peer loss, automatic cluster re-form, offline state in the dashboard
- [ ] Keep awake handling for demo sessions

## Phase 3: demo polish

- [ ] Token flow animation and per device activity in the dashboard
- [ ] Side by side screen: single device out of memory versus the cluster answering (shown honestly on the phone)
- [ ] 60 to 90 second demo video, README GIF, blog post

## Phase 4: beyond Apple

- [ ] Reference Android worker: Kotlin, `NsdManager` discovery, llama.cpp partial layer execution via the C API, GGUF checkpoints of the same base model
- [ ] Linux worker (llama.cpp, Avahi)
- [ ] Mixed runtime validation suite: text level agreement and perplexity sanity across MLX and llama.cpp shards

## Later / ideas

- Chunked prefill for long prompts
- Peer authentication (pre-shared key, then TLS) before any non-demo use
- QUIC transport experiment
- More model families (llama, qwen3); `ShardLoader` is the only qwen2 specific surface besides the module itself
- Speculative decoding with a small draft model on the coordinator, which turns pipeline latency into an advantage
- Weight streaming from the coordinator to workers that are missing their shard
