import json
import os

config_path = "/home/admin/zhangmaolin7/models/Qwen3-0.6B/config.json"
with open(config_path) as f:
    config = json.load(f)
    
print(f"rope_scaling: {config.get('rope_scaling')}")
print(f"model_type: {config.get('model_type')}")