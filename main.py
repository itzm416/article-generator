from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from LLMagent import stream_agent_response   # ✅ import from separate file

app = FastAPI()

# request body schema
class UserInput(BaseModel):
    text: str

# serve HTML page
@app.get("/")
async def home():
    return FileResponse("index.html")

# POST API
@app.post("/api/chat-stream")
async def chat_stream(data: UserInput):
    return StreamingResponse(
        stream_agent_response(data.text),
        media_type="text/plain",
    )