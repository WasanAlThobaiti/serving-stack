import os
import time
import uuid
import logging
import torch
from fastapi import FastAPI, Header, HTTPException, status
from transformers import AutoModelForCausalLM, AutoTokenizer
import schemas

# 1. قراءة المتغيرات من البيئة
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
API_KEY = os.environ.get("API_KEY", "")
MAX_TOKENS_CEILING = int(os.environ.get("MAX_TOKENS", "256"))

if not API_KEY:
    logging.warning("WARNING: API_KEY is unset! Service is running unauthenticated.")

app = FastAPI(title="OpenAI-compatible Serving Service")

# 2. تحميل النموذج
print(f"Loading tokenizer and model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu"
)
model.eval()
print("Model loaded successfully!")

# 3. دالة فحص المفتاح السري لمسارات v1
def verify_api_key(authorization: str = Header(None)):
    if API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Bearer token"
            )
        token = authorization.split("Bearer ")[1].strip()
        if token != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

# 4. نقطة فحص الصحة (تبقى مفتوحة دائماً بدون مفتاح)
@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_ID}

# 5. استعراض النماذج (محمية بالمفتاح)
@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    verify_api_key(authorization)
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

# 6. توليد الردود (محمية بالمفتاح مع تطبيق سقف التوكنز)
@app.post("/v1/chat/completions")
async def chat_completions(request: schemas.ChatCompletionRequest, authorization: str = Header(None)):
    verify_api_key(authorization)

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
    
    # 3. تطبيق سقف التوكنز الأقصى (Clamp ceiling)
    req_tokens = request.max_tokens if request.max_tokens else 32
    effective_max_tokens = min(req_tokens, MAX_TOKENS_CEILING)
    
    # 4. توليد الرد
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=effective_max_tokens,
            do_sample=False
        )
    
    # 5. فك تشفير التوكنز الجديدة
    new_token_ids = generated_ids[0][prompt_tokens:]
    completion_tokens = int(len(new_token_ids))
    content = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    
    # 6. إرجاع الرد بصيغة OpenAI
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
    
    if hasattr(schemas, "ChatCompletionResponse"):
        return schemas.ChatCompletionResponse(**response_data)
    return response_data