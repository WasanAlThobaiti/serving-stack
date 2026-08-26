import time
import uuid
import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import schemas

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI(title="OpenAI-compatible Serving Service")

print(f"Loading tokenizer and model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu"
)
model.eval()
print("Model loaded successfully!")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}

@app.get("/v1/models")
async def list_models():
    # استخدام كلاسات schemas المتوفرة مباشرة أو إرجاع قاموس متوافق
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "custom"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: schemas.ChatCompletionRequest):
    # 1. صياغة النص عبر المحادثة
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 2. حساب prompt tokens
    model_inputs = tokenizer([prompt_text], return_tensors="pt")
    prompt_tokens = int(model_inputs.input_ids.shape[1])
    
    # 3. توليد الرد
    max_tokens = request.max_tokens or 32
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )
    
    # 4. فك تشفير التوكنز الجديدة
    new_token_ids = generated_ids[0][prompt_tokens:]
    completion_tokens = int(len(new_token_ids))
    content = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    
    # 5. إرجاع الرد بصيغة OpenAI
    response_data = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    
    # إرجاع النموذج عبر schema إن وجد أو كـ JSON
    if hasattr(schemas, "ChatCompletionResponse"):
        return schemas.ChatCompletionResponse(**response_data)
    return response_data
