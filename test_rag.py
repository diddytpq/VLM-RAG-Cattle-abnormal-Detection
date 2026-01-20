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

# VideoMAE (Retriever)
retriever_model_id = "MCG-NJU/videomae-base" 
feature_extractor = AutoProcessor.from_pretrained(retriever_model_id)
retriever_model = VideoMAEModel.from_pretrained(retriever_model_id).to(device)

def get_video_embedding(video_path):
    """VideoMAE를 사용해 영상 특징 추출 (전처리 포함)"""
    try:
        rgb, audio, info = read_video(video_path, pts_unit='sec')
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return torch.zeros(1, 768).cpu()

    # 프레임 샘플링 (16개)
    if rgb.size(0) > 16:
        indices = torch.linspace(0, rgb.size(0) - 1, 16).long()
        video_frames = rgb[indices]
    else:
        indices = torch.linspace(0, rgb.size(0) - 1, 16).long()
        video_frames = rgb[indices]

    video_frames_numpy = list(video_frames.numpy())
    inputs = feature_extractor(video_frames_numpy, return_tensors="pt")
    
    with torch.no_grad():
        outputs = retriever_model(**inputs.to(device))
        video_emb = outputs.last_hidden_state.mean(dim=1) 
        
    video_emb = video_emb / video_emb.norm(p=2, dim=-1, keepdim=True)
    return video_emb.cpu()

def rag_inference(new_video_path, database_matrix):
    """가장 유사한 영상 1개의 인덱스 반환"""
    query_emb = get_video_embedding(new_video_path)
    similarity = torch.mm(query_emb, database_matrix.t())
    scores, indices = torch.topk(similarity, k=1)
    return indices[0].item() # .item()으로 스칼라 값 반환


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

    video_path = os.path.join(os.getcwd(), "datasets", "videos")
    video_path_list = os.listdir(video_path)
    # JSON 로드 (경로는 실제 환경에 맞게 수정 필요)
    text_json = json.load(open("./datasets/train_video_reason_annotation.json"))

    # 빠른 검색을 위한 매핑
    video_to_conversations = {}
    for item in text_json:
        json_video_name = os.path.basename(item["video"])
        video_to_conversations[json_video_name] = item["conversations"]

    dataset_embeddings = []
    dataset_embeddings_video_name = []
    dataset_context_map = {} # 인덱스 -> 텍스트 매핑

    for i, video_name in enumerate(video_path_list):
        full_path = os.path.join(video_path, video_name)
        emb = get_video_embedding(full_path)
        dataset_embeddings.append(emb)
        dataset_embeddings_video_name.append(video_name)
        
        # 해당 영상의 'GPT 답변'만 추출하여 Context로 저장
        if video_name in video_to_conversations:
            convs = video_to_conversations[video_name]
            context_text = " ".join([turn['value'] for turn in convs if turn['from'] == 'gpt'])
            dataset_context_map[i] = context_text
        else:
            dataset_context_map[i] = "No description available."

    dataset_embeddings = torch.cat(dataset_embeddings)
    dataset_embeddings = dataset_embeddings / dataset_embeddings.norm(p=2, dim=-1, keepdim=True)

    # --- [3. vLLM (Qwen-VL) 설정] ---
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
    new_video_path = os.path.join(os.getcwd(), "datasets", "val", "mounting")
    # new_video_path = os.path.join(os.getcwd(), "datasets", "val", "normal")

    video_list = os.listdir(new_video_path)

    FP = 0

    for video_name in video_list:
        target_full_path = os.path.join(new_video_path, video_name)
        
        # [Step 1] RAG: 유사한 과거 영상 찾기
        idx = rag_inference(target_full_path, dataset_embeddings)
        
        # [Step 2] Context 추출
        retrieved_video_name = dataset_embeddings_video_name[idx]
        retrieved_context = dataset_context_map[idx]
        
        # [Step 3] Prompt 구성 (RAG 적용)
        user_msg = (
            f"Context: I found a similar historical video which was described as follows: \"{retrieved_context}\"\n\n"
            "Task: Based on the visual content of the provided video and the context above, "
            "analyze whether the cow's behavior is Normal or mounting. "
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
            print(generated_text)
            
            # Thinking 모델: </think> 이후의 실제 답변만 추출
            if "</think>" in generated_text:
                final_answer = generated_text.split("</think>")[-1].strip()
            else:
                final_answer = generated_text
            
            print(f"Video: {video_name}")
            print(f"Ref Video: {retrieved_video_name}")
            print(f"Output: {final_answer}\n" + "-"*30)


            # 결과를 result.json에 누적 저장
            result_dict = {
                "video": video_name,
                "retrieved_video": retrieved_video_name,
                "output": final_answer
            }
            
            result_json_path = "./result.json"
            # result_json_path = "./result_normal.json"

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
            break


    print(f"Total FP Count: {FP}")
    
    # 정상 종료
    del llm
