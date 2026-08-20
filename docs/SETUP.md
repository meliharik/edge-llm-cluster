# Setup, first run, and publishing

Everything in this repo was authored ahead of the first build. This page is the checklist for bringing it up on your own machine and putting it on GitHub.

## Requirements

- Apple Silicon Mac (MLX requires it)
- Xcode 26 or newer (the `mlx-swift-lm` dependency requires a Swift 6.2+ toolchain)
- ~1 GB free disk for the Phase 0 test model
- Network access on first build (Swift package resolution + Hugging Face model download)

## 1. First build

```bash
cd edge-llm-cluster
swift build
```

The first build is slow (10 to 20 minutes): it compiles MLX's Metal kernels once. Subsequent builds are fast.

**If the build fails with an API mismatch** in `EdgeLLMKit` or `PipelineProof`: the code was written against `mlx-swift-lm` **3.31.4** exactly (pinned in `Package.swift`) by reading that tag's sources, but was not compiled at authoring time. Typical fixes are one-line signature adjustments; the upstream sources to compare against are:

- `mlx-swift-lm/Libraries/MLXLLM/Models/Qwen2.swift`, the model our shard mirrors
- `mlx-swift-lm/Libraries/MLXLMCommon/Load.swift`, the quantize and update pattern `ShardLoader` follows
- `mlx-swift-lm/Libraries/MLXLMCommon/Documentation.docc/using.md`, the factory and macro usage `PipelineProof` follows

## 2. Run the Phase 0 proof

```bash
swift run -c release pipeline-proof
```

Expected output shape:

```
edge-llm-cluster · pipeline-proof
model: mlx-community/Qwen2.5-0.5B-Instruct-4bit

[1/4] downloading & loading upstream model …
      download 100%
      upstream says: The sky appears blue because …
[2/4] reference: our stack, all 24 layers in one shard …
      52 tokens, 95.3 tok/s
[3/4] pipeline: layers 0..<12 → codec → layers 12..<24 …
[4/4] results
...
✓ sharded pipeline matches unsharded reference token for token
✓ our implementation agrees with upstream mlx-swift-lm output
```

The two ✓ lines are the Phase 0 milestone. Then try a bigger model:

```bash
swift run -c release pipeline-proof --model mlx-community/Qwen2.5-3B-Instruct-4bit --max-tokens 128
```

Record the tok/s numbers in `docs/ROADMAP.md` (Phase 0 baseline checkbox).

Run the unit tests too:

```bash
swift test
```

## 2b. Try the dashboard

```bash
swift run edge-coordinator --demo
```

Open http://localhost:8080 in any browser on the network. The `--demo` flag seeds simulated devices so you can see the device cards, memory quota bars and live metrics without running any workers. Without the flag the coordinator advertises `_edgellm._tcp` and waits for real workers to connect.

## 3. Repository housekeeping

- `Package.resolved` is committed on purpose so CI and contributors get identical dependency versions. If you intentionally bump dependencies, commit the updated file with the change.
- CI (`.github/workflows/ci.yml`) builds the package and runs the pure-Swift tests on a macOS runner; the badge in the README tracks `main`.
- When a phase lands, tag it (`git tag v0.x.0 && git push --tags`) and add a matching entry to `CHANGELOG.md`.

## 4. Model cache location

Models download once via the Hugging Face Hub client and are cached under your user directory (`~/.cache/huggingface/` or the Hub client's default). Deleting the cache just re-triggers the download.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: package 'mlx-swift-lm' requires minimum Swift tools version 6.2` | Update Xcode; check `swift --version` ≥ 6.2 |
| Metal shader compile errors on first build | Ensure Xcode (not just CLT) is selected: `sudo xcode-select -s /Applications/Xcode.app` |
| Download hangs at 0% | Hugging Face reachability; retry, or pre-download the repo with `huggingface-cli download mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| `unsupported model_type` | Phase 0 supports `qwen2`-family checkpoints only (Qwen2 / Qwen2.5) |
| Out of memory with `--model …7B…` on an 8 GB Mac | The proof loads the model **twice** (reference + shards); use the 0.5B/3B models for the proof |
