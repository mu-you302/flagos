import argparse
import dataclasses
import sys
import types


def test_apply_qwen3_4b_benchmark_defaults_sets_scenario_defaults():
    import vllm_fl

    args = argparse.Namespace(
        model="/data/xiyang/vllm-plugin-FL/Qwen3-4B",
        input_len=4096,
        output_len=1024,
    )
    engine_args = argparse.Namespace(
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
    )

    applied = vllm_fl._apply_qwen3_4b_benchmark_defaults(engine_args, args)

    assert applied == {
        "max_model_len": 6144,
        "max_num_batched_tokens": 12288,
        "max_num_seqs": 256,
    }
    assert engine_args.max_model_len == 6144
    assert engine_args.max_num_batched_tokens == 12288
    assert engine_args.max_num_seqs == 256


def test_apply_qwen3_4b_benchmark_defaults_preserves_user_overrides():
    import vllm_fl

    args = argparse.Namespace(
        model="/data/xiyang/vllm-plugin-FL/Qwen3-4B",
        input_len=1024,
        output_len=1024,
    )
    engine_args = argparse.Namespace(
        max_model_len=3072,
        max_num_batched_tokens=None,
        max_num_seqs=64,
    )

    applied = vllm_fl._apply_qwen3_4b_benchmark_defaults(engine_args, args)

    assert applied == {"max_num_batched_tokens": 16384}
    assert engine_args.max_model_len == 3072
    assert engine_args.max_num_batched_tokens == 16384
    assert engine_args.max_num_seqs == 64


def test_patch_engine_args_from_cli_args_applies_benchmark_defaults(monkeypatch):
    import vllm_fl

    fake_vllm = types.ModuleType("vllm")
    fake_vllm_engine = types.ModuleType("vllm.engine")
    fake_arg_utils = types.ModuleType("vllm.engine.arg_utils")

    @dataclasses.dataclass
    class FakeEngineArgs:
        model: str
        max_model_len: int | None = None
        max_num_batched_tokens: int | None = None
        max_num_seqs: int | None = None

        @classmethod
        def from_cli_args(cls, args):
            attrs = [field.name for field in dataclasses.fields(cls)]
            return cls(
                **{
                    attr: getattr(args, attr)
                    for attr in attrs
                    if hasattr(args, attr)
                }
            )

    fake_arg_utils.EngineArgs = FakeEngineArgs
    fake_vllm.engine = fake_vllm_engine

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", fake_vllm_engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", fake_arg_utils)
    monkeypatch.setattr(
        vllm_fl,
        "_should_bypass_enforce_eager_for_cuda_runtime",
        lambda: True,
    )

    vllm_fl._patch_engine_args_for_benchmark_defaults()

    args = argparse.Namespace(
        model="/data/xiyang/vllm-plugin-FL/Qwen3-4B",
        input_len=6144,
        output_len=1024,
        max_model_len=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
    )

    engine_args = FakeEngineArgs.from_cli_args(args)

    assert engine_args.max_model_len == 8192
    assert engine_args.max_num_batched_tokens == 8192
    assert engine_args.max_num_seqs == 128
