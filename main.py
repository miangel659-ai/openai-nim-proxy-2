import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permitir CORS desde cualquier origen (necesario para Janitor AI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("Falta la variable de entorno NVIDIA_API_KEY")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    is_stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            if is_stream:
                # Petición streaming a NVIDIA
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
                if response.status_code != 200:
                    return JSONResponse(
                        status_code=response.status_code,
                        content=response.json(),
                    )
                return JSONResponse(content=response.json())

        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Error upstream: {str(e)}")


async def stream_sse_chunks(response: httpx.Response):
    """Reenvía los bytes tal cual los manda NVIDIA (SSE)"""
    async for chunk in response.aiter_bytes():
        yield chunk
