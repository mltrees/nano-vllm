import sys
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

from pathlib import Path

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        ## zml add for debug ###########
        if True:
            print(f"zml: after schedule, len(seqs)={len(seqs)}, is_prefill={is_prefill}")
            for i in range(0,len(seqs)):
                print(f"zml: i={i}, len(seq)={len(seqs[i])},seq={seqs[i].debug()}")
        ## zml add for debug ###########
        print(f"zml: run into {Path(sys._getframe().f_code.co_filename).name}, line:{sys._getframe().f_lineno}")
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        ## zml add for debug ###########
        if True:
            print(f"zml: after runner, len(token_ids)={len(token_ids)}, len(seqs)={len(seqs)}")
            for i in range(0,len(token_ids)):
                print(f"zml: i={i}, len(token_ids)={len(token_ids)},token_ids={token_ids[i]}")
            
        ## zml add for debug ###########
        self.scheduler.postprocess(seqs, token_ids)
        ## zml add for debug ###########
        if True:
            print(f"zml: after postprocess, len(token_ids)={len(token_ids)}, len(seqs)={len(seqs)}")
            for i in range(0,len(seqs)):
                print(f"zml: i={i}, len(seq)={len(seqs[i])},seq={seqs[i].debug()}")
        ## zml add for debug ###########
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        ## zml add for debug ###########
        if True:
            print(f"zml: finish one step, len(outputs)={len(outputs)}, num_tokens={num_tokens}")
            
        ## zml add for debug ###########
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
    # ===== zml: 新增：入口调试信息 =====
        if True:
            print("=" * 60)
            print(f"[generate] 开始生成")
            print(f"[generate] 请求数量: {len(prompts)}")
            print(f"[generate] 提示词示例: {str(prompts[0])[:50]}...")
            print(f"[generate] 采样参数: {sampling_params if isinstance(sampling_params, SamplingParams) else f'list[{len(sampling_params)}]'}")
            print("=" * 60)
    # =============================
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            print(f"zml: after step, len(output)={len(output)}, num_tokens={num_tokens}")
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
