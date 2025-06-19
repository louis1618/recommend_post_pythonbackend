# models.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from database import Base
from sqlalchemy import Column, String, Integer, DateTime, Float, JSON

# API 응답을 위한 Pydantic 모델
class PostResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    platform: str
    topic: str
    author: str
    content: str
    url: str
    timestamp: datetime
    likes: int
    comments_count: int
    shares: int