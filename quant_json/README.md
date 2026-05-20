# Qwen3-4B C8 KV Cache Quantization Files

Generated for `/data/xiyang/vllm-plugin-FL/Qwen3-4B`.

Files:

- `generate_kv_cache_quant.py`: calibration script.
- `calibration_prompts.json`: prompts sampled from eval-style datasets.
- `kv_cache_scales.safetensors`: per-layer K/V KV-cache scale and offset tensors.
- `quant_model_description.json`: minimal ModelSlim-style description with `kv_cache_type=C8`.
- `model.safetensors.index.c8_patch.json`: original model index plus the C8 tensor entries.
- `kv_cache_quant_manifest.json`: generation metadata and scale ranges.

The generated tensor names follow the vllm-ascend C8 convention:

- `model.layers.N.self_attn.k_proj.kv_cache_scale`
- `model.layers.N.self_attn.k_proj.kv_cache_offset`
- `model.layers.N.self_attn.v_proj.kv_cache_scale`
- `model.layers.N.self_attn.v_proj.kv_cache_offset`

To make a loader see these files with the existing model directory, place
`kv_cache_scales.safetensors` and `quant_model_description.json` next to the
model shards, and use `model.safetensors.index.c8_patch.json` as the model's
`model.safetensors.index.json`.
