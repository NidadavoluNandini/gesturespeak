# server.py
import base64, json, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from inference_tflite import predict_from_bytes, reset_state

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "running"}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    reset_state()
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                b64 = payload.get("image")
            except:
                b64 = data
            if not b64:
                await ws.send_text(json.dumps({"error": "no image"}))
                continue

            image_bytes = base64.b64decode(b64.split(",")[-1])
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, predict_from_bytes, image_bytes)
            await ws.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
