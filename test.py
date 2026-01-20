import torch
import numpy as np
import os
import json
from torchvision.io.video import read_video
from transformers import AutoProcessor, VideoMAEModel
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

# --- [1. 설정 및 VideoMAE 로드] ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# vLLM 입력 준비 함수
def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Qwen2-VL 특화 처리 (Dynamic Resolution 등)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        "mm_processor_kwargs": video_kwargs, # 중요: Qwen2-VL의 grid thw 정보 전달
    }

if __name__ == '__main__':
    # --- [2. Database Indexing (VideoMAE)] ---
    print("Indexing Database...")

    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    # 모델 로드 (경로 주의)
    checkpoint_path = os.path.join(os.getcwd(), "weights", "Qwen3-VL-4B-Thinking-FP8")
    # Qwen-VL용 Processor 로드
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        max_model_len=8192,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )

    tokenizer = processor.tokenizer # 토크나이저 가져오기

    # Qwen 계열의 일반적인 종료 토큰 ID들
    stop_token_ids = [tokenizer.eos_token_id]
    # <|im_end|>, <|endoftext|> 등의 특수 토큰 ID도 포함하는 것이 안전함
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        stop_token_ids.extend(tokenizer.additional_special_tokens_ids)

    sampling_params = SamplingParams(
            temperature=0.2,          # 0보다는 약간 높여서 루프 방지 (0.1 ~ 0.2 추천)
            repetition_penalty=1.1,   # 1.1 정도면 반복 생성을 효과적으로 억제함
            max_tokens=5120,
            top_k=20,                 # -1 대신 적당한 후보군 제한
            stop_token_ids=stop_token_ids, # 종료 토큰 명시
        )

    # --- [4. RAG + vLLM Inference Pipeline] ---
    print("Start Inference...")
    # new_video_path = os.path.join(os.getcwd(), "datasets", "val", "mounting")
    new_video_path = os.path.join(os.getcwd(), "datasets", "val", "normal")

    video_list = os.listdir(new_video_path)

    FP = 0

    for video_name in video_list:
        target_full_path = os.path.join(new_video_path, video_name)
        
        user_msg = (
            "Task: Analyze whether the cow's behavior is Normal or mounting. "
            "Provide a reasoning and conclude with 'Status: Normal' or 'Status: Mounting'."
        )
                
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant capable of visual reasoning. You must first output your thought process between <think> and </think> tags, and then provide the final answer."
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video", 
                        "video": target_full_path,
                        # "max_pixels": 360 * 420,
                        "fps": 3.0, 
                    },
                    {"type": "text", "text": user_msg},
                ]
            }
        ]
        
        # [Step 4] vLLM Inference
        try:
            inputs = prepare_inputs_for_vllm(messages, processor)
            outputs = llm.generate([inputs], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text
            # print(generated_text)
            
            # Thinking 모델: </think> 이후의 실제 답변만 추출
            if "</think>" in generated_text:
                final_answer = generated_text.split("</think>")[-1].strip()
            else:
                final_answer = generated_text
            
            # print(f"Video: {video_name}")
            # print(f"Output: {final_answer}\n" + "-"*30)


            # 결과를 result.json에 누적 저장
            result_dict = {
                "video": video_name,
                "output": final_answer
            }
            
            # result_json_path = "./result.json"
            result_json_path = "./result_normal.json"

            # 기존 결과 로드 (파일이 있으면)
            if os.path.exists(result_json_path):
                with open(result_json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                results = []
            
            # 새 결과 추가 및 저장
            results.append(result_dict)
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # [Step 5] 결과 분석 (False Positive Check)
            if "Normal" in final_answer or "status: normal" in final_answer.lower():
                pass
                FP += 1
            else:
                pass
                # FP += 1
                
        except Exception as e:
            print(f"Inference Error on {video_name}: {e}")


    print(f"Total FP Count: {FP}")
    
    # 정상 종료
    del llm
