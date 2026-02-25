from nanovllm import LLM, SamplingParams
llm = LLM("/home/admin/zhangmaolin7/models/Qwen3-0.6B", enforce_eager=True)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(f"zml: outputs={outputs}")