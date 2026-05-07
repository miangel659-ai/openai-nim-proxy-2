import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time

app = FastAPI()

# CORS para Janitor AI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clave de NVIDIA desde variable de entorno
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("Falta la variable de entorno NVIDIA_API_KEY")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# --- LÍMITES AJUSTADOS PARA DeepSeek V4 Pro ---
MAX_CONTEXT_TOKENS = 20000   # Reducido para evitar sobrecarga en capa gratuita
MAX_OUTPUT_TOKENS = 2000     # Más conservador para V4 Pro
WAIT_BETWEEN = 20            # 20 segundos entre peticiones (3 RPM)

def estimar_tokens(messages):
    total = 0
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], str):
            total += len(msg["content"]) // 3.5
    return int(total)

def recortar_historial(messages, max_tokens):
    while estimar_tokens(messages) > max_tokens and len(messages) > 2:
        if messages[0]["role"] == "system":
            if len(messages) > 2:
                messages.pop(1)
            else:
                break
        else:
            messages.pop(0)
    return messages

last_request_time = 0
lock = asyncio.Lock()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global last_request_time

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    # Modelo por defecto
    if "model" not in body or not body["model"]:
        body["model"] = "deepseek-ai/deepseek-v4-pro"

    # --- CONTROL DE CONTEXTO Y TOKENS ---
    if "messages" in body:
        token_estimados = estimar_tokens(body["messages"])
        print(f"ℹ️ Tokens estimados: {token_estimados}")
        if token_estimados > MAX_CONTEXT_TOKENS:
            print(f"⚠️ Recortando historial ({token_estimados} > {MAX_CONTEXT_TOKENS})")
            body["messages"] = recortar_historial(body["messages"], MAX_CONTEXT_TOKENS)
    
    if "max_tokens" not in body or body["max_tokens"] > MAX_OUTPUT_TOKENS:
        body["max_tokens"] = MAX_OUTPUT_TOKENS

    is_stream = body.get("stream", False)

    # --- ESPERAR TURNO ---
    async with lock:
        now = time.time()
        wait = WAIT_BETWEEN - (now - last_request_time)
        if wait > 0:
            print(f"⏳ Esperando {wait:.0f}s...")
            await asyncio.sleep(wait)
        last_request_time = time.time()

    # Llamar a NVIDIA
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            if is_stream:
                req = client.build_request(
                    "POST",
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    json=body,
                    headers=headers,
                )
                response = await client.send(req, stream=True)

                if response.status_code != 200:
                    error_body = await response.aread()
                    try:
                        error_json = httpx.Response(200, content=error_body).json()
                    except:
                        error_json = {"error": {"message": error_body.decode()}}
                    return JSONResponse(status_code=response.status_code, content=error_json)

                return StreamingResponse(
                    stream_sse_chunks(response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                response = await client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    json=body,
                    headers=headers,
                )

                if response.status_code != 200:
                    return JSONResponse(status_code=response.status_code, content=response.json())

                return JSONResponse(content=response.json())

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error: {str(e)}")


async def stream_sse_chunks(response: httpx.Response):
    async for chunk in response.aiter_bytes():
        yield chunk
