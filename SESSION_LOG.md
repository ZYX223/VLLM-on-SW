# Session Log

## 2026-04-05

### Goal

Merge the SW emulator prototype back into the main repository and make `VLLM-on-SW` the single active codebase.

### Decisions

- Keep `VLLM-on-SW` as the single development base.
- Port the switchable attention backend from `VLLM-on-SW`.
- Keep scheduler / block manager / sequence flow unchanged for v1.
- Treat `sw_emulator` as a correctness and interface prototype, not a real Sunway performance simulator.
- Keep the Python import path as `nanovllm` for compatibility.

### Implemented Changes

- Added `backend` config plumbing and propagated it into model construction.
- Added `nanovllm/backends/attention.py` and `nanovllm/backends/__init__.py`.
- Refactored `nanovllm/layers/attention.py` to dispatch through the backend layer.
- Wired Qwen3 attention to accept the backend selection.
- Added CPU semantic tests in `tests/test_sw_emulator_backend.py`.
- Updated `README.md` with `backend="sw_emulator"` usage.
- Added `sw_vllm_emulator_blueprint.md` to keep the SW integration plan with the main repository.

### Runtime Fixes Carried Over

- Fixed `rope_scaling` startup failure caused by unhashable dict input in RoPE cache handling.
- Replaced the hard-coded distributed init port with a dynamically allocated local TCP port.
- Made `LLM.exit()` idempotent and unregister its `atexit` callback.
- Made `ModelRunner.exit()` explicitly release model / KV cache / sampler and clear CUDA cache.
- Limited `sw_emulator` warmup length to 512 tokens to avoid quadratic warmup OOM from the dense reference attention path.

### Next Likely Step

- Extend emulator coverage beyond attention / KV cache, or add stronger end-to-end regression tests across multiple prompts and batch cases.
