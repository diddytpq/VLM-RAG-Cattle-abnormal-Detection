"""텍스트 전용 테스트 - 모델이 정상 작동하는지 확인"""
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    checkpoint_path = os.path.join(os.getcwd(), "weights", "Qwen3-VL-4B-Thinking-FP8")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
        enforce_eager=True,  # 안정성을 위해 eager 모드
        tensor_parallel_size=torch.cuda.device_count(),
        seed=42,
    )
    
    # 간단한 텍스트 프롬프트
    messages = [
        {"role": "user", "content": "Hello, what is 2+2? Answer briefly."}
    ]
    
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Prompt:\n{prompt}\n{'='*50}")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
        top_p=0.9,
    )
    
    outputs = llm.generate([{"prompt": prompt}], sampling_params=sampling_params)
    print(f"Output: {outputs[0].outputs[0].text}")
    
    del llm

