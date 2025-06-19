# load_data.py
import json
from datetime import datetime
from database import SessionLocal, Post

def load_initial_data():
    db = SessionLocal()
    try:
        # 데이터가 이미 있으면 실행하지 않음
        if db.query(Post).count() > 0:
            print("Data already exists. Skipping load.")
            return

        with open('postdata.json', 'r', encoding='utf-8') as f:
            posts_data = json.load(f)

        print(f"Loading {len(posts_data)} posts into the database...")
        for item in posts_data:
            post = Post(
                id=item['id'],
                platform=item['platform'],
                topic=item['topic'],
                author=item['author'],
                content=item['content'],
                url=item['url'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                likes=item['likes'],
                comments_count=item['comments_count'],
                shares=item['shares']
            )
            db.add(post)
        
        db.commit()
        print("Data loaded successfully.")

    finally:
        db.close()

if __name__ == "__main__":
    load_initial_data()