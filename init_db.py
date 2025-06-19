# init_db.py
from database import engine, Base, UserProfile, SessionLocal
import os

DB_FILE = "recommend_app.db"

def initialize_database():
    # 만약 DB 파일이 이미 있으면 삭제 (초기화를 위해)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print(f"Removed existing database file: {DB_FILE}")

    print("Creating new database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

    # 기본 사용자 프로필 생성
    db = SessionLocal()
    try:
        # 테스트를 위한 기본 사용자 생성
        test_user = UserProfile(user_id="current_user_id")
        db.add(test_user)
        db.commit()
        print("Created default user profile for 'current_user_id'.")
    finally:
        db.close()

if __name__ == "__main__":
    initialize_database()