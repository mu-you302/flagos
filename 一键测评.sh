cd benchmarks/flagos_eval
CUDA_VISIBLE_DEVICES=0,1 bash run_benchmark.sh /data/xiyang/vllm-plugin-FL/Qwen3-4B &> output.log