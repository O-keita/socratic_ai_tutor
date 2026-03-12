# Performance Profiling Data

## Test Device
- Device: Huawei
- RAM: 4 GB
- SoC/CPU: (fill in)
- Android version: (fill in)

## Baseline: Online Mode (No Model Loaded)
- **Date**: 2026-03-10
- **Recording duration**: ~08:032s
- **Total**: 479.1 MB
- **Java**: 15.6 MB
- **Native**: 72.4 MB
- **Graphics**: 85.8 MB
- **Stack**: 140 KB
- **Code**: 53.7 MB
- **Others**: 251.4 MB
- **Allocated objects**: 304,184

## Before Optimization: Offline Mode (Model Loaded, n_ctx=2048, no token cap)
- **Date**: 2026-03-10
- **Recording duration**: ~03:182s
- **Total**: 1 GB
- **Java**: 7.9 MB
- **Native**: 784.9 MB
- **Graphics**: 55.6 MB
- **Stack**: 168 KB
- **Code**: 27.1 MB
- **Others**: 183.8 MB
- **Allocated objects**: 144,599

## After Optimization: Offline Mode (Model Loaded, n_ctx=512, 150 token cap, /no_think)
- **Date**: 2026-03-10
- **Build mode**: DEBUG (no native optimizations — -O0)
- **Recording duration**: ~05:499s
- **Total**: 966.9 MB
- **Java**: 14.3 MB
- **Native**: 630.2 MB
- **Graphics**: 87.8 MB
- **Stack**: 164 KB
- **Code**: 33.4 MB
- **Others**: 201 MB
- **Allocated objects**: 271,369
- **Inference time**: ~60s (debug, unoptimized native code)
- **Note**: Debug build compiles libchat.so with -O0 — expected 10-15x slower than release

## After Optimization: Offline Mode (RELEASE build — pending)
- **Date**: (pending)
- **Total**: (pending)
- **Native**: (pending) — expected ~630 MB
- **Inference time (1st turn)**: (pending) — expected 4-7s
- **Inference time (5th turn)**: (pending)

## Optimizations Applied
1. `/no_think` added to system prompt — eliminates ~30-50% wasted think tokens
2. `n_ctx` reduced from 2048 → 512 — ~4x smaller KV cache
3. Max token cap of 150 in native decode loop — prevents runaway generation

## Notes
- Model: Qwen3-0.6B Q4_K_M (~460 MB on disk)
- Native memory delta (model overhead): 784.9 - 72.4 = **712.5 MB** (pre-optimization)
