import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig
import torch
import sys

def debug_main():
    path = os.path.expanduser("/home/admin/zhangmaolin7/models/Qwen3-0.6B/")
    
    print("="*50)
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"Tokenizer loaded: {type(tokenizer)}")
    
    print("\nStep 2: Checking model config...")
    # 检查配置
    config = AutoConfig.from_pretrained(path)
    print(f"Config type: {type(config)}")
    print(f"Config attributes: {dir(config)[:20]}")  # 打印前20个属性
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {getattr(config, 'hidden_size', 'Not found')}")
    print(f"Num layers: {getattr(config, 'num_hidden_layers', 'Not found')}")
    
    print("\nStep 3: Creating LLM instance...")
    try:
        # 尝试创建LLM
        llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
        print("LLM created successfully!")
    except Exception as e:
        print(f"Error creating LLM: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # 尝试调试nano-vllm内部
        print("\n" + "="*50)
        print("Debugging nano-vllm internals...")
        
        # 检查nano-vllm的导入路径
        import nanovllm
        print(f"nano-vllm path: {nanovllm.__file__}")
        
        # 检查Qwen3模型文件
        from nanovllm.models import qwen3
        print(f"Qwen3 module path: {qwen3.__file__}")
        
        # 检查get_rope函数
        if hasattr(qwen3, 'get_rope'):
            print("get_rope function exists")
            import inspect
            print(f"get_rope signature: {inspect.signature(qwen3.get_rope)}")
        else:
            print("get_rope function not found!")
        
        return
    
    print("\nStep 4: Preparing prompts...")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    print("\nStep 5: Generating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    for prompt, output in zip(prompts, outputs):
        print("\n" + "-"*30)
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
    debug_main()