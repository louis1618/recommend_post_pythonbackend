import os # os 모듈 추가
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, ConfigDict
from datetime import datetime

# --- 1. 데이터베이스 연결 설정 ---
# 개발 환경에서는 SQLite를 사용하고, 배포 환경(PRODUCTION)에서는 환경변수에서 DB URL을 가져옴
if os.getenv("ENV") == "PRODUCTION":
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
else:
    SQLALCHEMY_DATABASE_URL = "sqlite:///./recommend_app.db"


engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. SQLAlchemy 모델 (테이블) 정의 ---
class Post(Base):
    __tablename__ = "posts"
    id = Column(String, primary_key=True, index=True)
    platform = Column(String)
    topic = Column(String, index=True)
    author = Column(String)
    content = Column(String)
    url = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    likes = Column(Integer)
    comments_count = Column(Integer)
    shares = Column(Integer)

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    post_id = Column(String, index=True)
    interaction_type = Column(String)
    engagement_score = Column(Float)
    duration = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, unique=True, index=True)
    preferences = Column(JSON, default=lambda: {"topic_preferences": {}})

# --- 3. Pydantic 모델 (API 응답용) 정의 ---
class PostResponse(BaseModel):
    # Pydantic V2 설정 방식
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