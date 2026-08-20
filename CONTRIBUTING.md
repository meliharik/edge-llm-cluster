# Contributing

Thanks for your interest! The project is early — the highest-value contributions right now are running the phases on real hardware and reporting numbers.

## Ground rules

- **Discuss before large changes.** Open an issue describing what you want to do; the roadmap in [docs/ROADMAP.md](docs/ROADMAP.md) is the source of truth for direction.
- **Keep the core honest.** `EdgeLLMKit` must stay free of UI and network code; `EdgeLLMTransport` must stay free of model code. The `PipelineGenerator.Link` seam is where they meet.
- **Correctness first.** Any change touching the model or codec must keep `pipeline-proof` green (token-exact match) and `swift test` passing.

## Dev loop

```bash
swift build            # first build compiles MLX Metal kernels (slow once)
swift test             # pure-Swift tests, no GPU needed
swift run -c release pipeline-proof   # the real acceptance test (needs Apple Silicon)
```

## Style

- Follow the surrounding code; formatting matches upstream mlx-swift-lm conventions.
- Public API gets doc comments; internals get comments only where the *why* isn't obvious.

## Reporting device results

When you run a phase on new hardware, include in the issue: device + chip + RAM, macOS/iOS version, model id, tok/s, and the proof/verdict lines from the tool output.
