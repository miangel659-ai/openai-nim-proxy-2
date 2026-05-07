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

# --- CONTROL DE RATE LIMIT MANUAL ---
# Máximo de peticiones por minuto
MAX_REQUESTS_PER_MINUTE = 4
# Tiempo mínimo entre peticiones (en segundos)
MIN_TIME_BETWEEN_REQUESTS = 15  # 4 peticiones por minuto = una cada 15 segundos

request_timestamps = []  # Guarda timestamps de peticiones recientes
last_request_time = 0  # Timestamp de la última petición enviada a NVIDIA
lock = asyncio.Lock()  # Para operaciones thread-safe

@app.get("/")
def health():
    return {"status": "ok", "message": "Proxy NVIDIA NIM funcionando"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global request_timestamps, last_request_time
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    is_stream = body.get("stream", False)

    # Modelo por defecto
    if "model" not in body or not body["model"]:
        body["model"] = "deepseek-ai/deepseek-r1"

    # --- VERIFICAR RATE LIMIT ---
    current_time = time.time()
    
    async with lock:
        # Limpiar timestamps viejos (más de 60 segundos)
        request_timestamps = [t for t in request_timestamps if current_time - t < 60]
        
        # Verificar si excedemos el límite por minuto
        if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            oldest_in_window = min(request_timestamps)
            wait_time = 60 - (current_time - oldest_in_window) + 1  # +1 de margen
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Límite de peticiones alcanzado. Espera {int(wait_time)} segundos.",
                        "type": "rate_limit",
                        "retry_after": int(wait_time)
                    }
                }
            )
        
        # Verificar tiempo mínimo entre peticiones
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_TIME_BETWEEN_REQUESTS and last_request_time > 0:
            wait_time = MIN_TIME_BETWEEN_REQUESTS - time_since_last + 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Demasiado rápido. Espera {int(wait_time)} segundos entre peticiones.",
                        "type": "rate_limit",
                        "retry_after": int(wait_time)
                    }
                }
            )
        
        # Registrar esta petición
        request_timestamps.append(current_time)
        last_request_time = current_time

    # --- LLAMAR A NVIDIA ---
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if is_stream:
                req = client.build_request(
                    "POST",
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    json=body,
                    headers=headers,
                )
                response = await client.send(req, stream=True)

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "message": f"Límite de NVIDIA alcanzado. Espera {retry_after} segundos.",
                                "type": "nvidia_rate_limit",
                                "retry_after": retry_after
                            }
                        }
                    )

                if response.status_code != 200:
                    error_body = await response.aread()
                    try:
                        error_json = httpx.Response(200, content=error_body).json()
                    except Exception:
                        error_json = {"error": {"message": error_body.decode()}}
                    return JSONResponse(
                        status_code=response.status_code, content=error_json
                    )

                return StreamingResponse(
                    stream_sse_chunks(response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                response = await client.post(
                    f"{NVIDIA_BASE_URL}/chat/completions",
                    json=body,
                    headers=headers,
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "message": f"Límite de NVIDIA alcanzado. Espera {retry_after} segundos.",
                                "type": "nvidia_rate_limit",
                                "retry_after": retry_after
                            }
                        }
                    )

                if response.status_code != 200:
                    return JSONResponse(
                        status_code=response.status_code,
                        content=response.json(),
                    )
                return JSONResponse(content=response.json())

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error de conexión con NVIDIA: {str(e)}")


async def stream_sse_chunks(response: httpx.Response):
    async for chunk in response.aiter_bytes():
        yield chunk
