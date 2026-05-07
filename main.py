import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio

# Rate limiter: 3 peticiones por minuto por IP
limiter = Limiter(key_func=get_remote_address, default_limits=["3/minute"])

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# Solo 1 petición simultánea a NVIDIA
semaphore = asyncio.Semaphore(1)

@app.get("/")
def health():
    return {"status": "ok", "message": "Proxy NVIDIA NIM funcionando"}

@app.post("/v1/chat/completions")
@limiter.limit("3/minute")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    is_stream = body.get("stream", False)

    # Modelo por defecto
    if "model" not in body or not body["model"]:
        body["model"] = "deepseek-ai/deepseek-r1"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    async with semaphore:
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
                        retry_after = response.headers.get("Retry-After", "20")
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": {
                                    "message": f"Límite de NVIDIA alcanzado. Espera {retry_after} segundos.",
                                    "type": "rate_limit",
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
                        retry_after = response.headers.get("Retry-After", "20")
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": {
                                    "message": f"Límite de NVIDIA alcanzado. Espera {retry_after} segundos.",
                                    "type": "rate_limit",
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
