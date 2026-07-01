from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # e.g. "K" (knowledge), "P" (personality), "A" (ability)


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class AgentOutput(BaseModel):
    intent: Literal["clarify", "recommend", "refine", "compare", "off_topic", "refuse"]
    reply: str
    recommended_names: list[str] = Field(default_factory=list)
    end_of_conversation: bool = False
