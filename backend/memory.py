from typing import Dict, List
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class SessionMemory:
    def __init__(self):
        # In-memory store: session_id -> list of messages
        self.store: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        if session_id not in self.store:
            self.store[session_id] = []
        return self.store[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.store:
            self.store[session_id] = []
        self.store[session_id].append(ChatMessage(role=role, content=content))

    def get_history_string(self, session_id: str) -> str:
        history = self.get_history(session_id)
        return "\n".join([f"{msg.role}: {msg.content}" for msg in history])

memory_store = SessionMemory()
