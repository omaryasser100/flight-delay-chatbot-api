from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_module import get_bot_answer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Optional: allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = get_bot_answer(request.query)
    return {"response": response}
