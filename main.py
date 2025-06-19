# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import json
import redis
import logging
import traceback
import asyncio

from database import get_db, Post, UserInteraction, UserProfile, PostResponse
from recommendation_engine import RecommendationEngine
from user_behavior_tracker import UserBehaviorTracker

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정 (프론트엔드 개발 서버 주소 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis 클라이언트 (캐싱용)
try:
    # 환경 변수에서 REDIS_URL을 가져옴
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    redis_client = redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis: {e}")
    redis_client = None

# 추천 엔진 및 행동 추적기 인스턴스
recommendation_engine = RecommendationEngine()
behavior_tracker = UserBehaviorTracker()


# --- 요청 본문을 위한 Pydantic 모델 정의 ---
class FeedRequest(BaseModel):
    user_id: str
    page: int = 0
    limit: int = 10

class InteractionRequest(BaseModel):
    user_id: str
    post_id: str
    interaction_type: str  # view, like, share, scroll_time, click
    duration: Optional[float] = None

# PostResponse에 is_liked 필드를 추가한 새로운 응답 모델 정의
class PostFeedResponse(PostResponse):
    is_liked: bool = False # 좋아요 여부 추가

@app.get("/api/posts/liked", response_model=List[PostFeedResponse])
async def get_liked_posts(user_id: str, db: Session = Depends(get_db)):
    """
    사용자가 '좋아요'한 게시물 목록을 반환합니다.
    """
    try:
        # 1. 사용자가 '좋아요'한 모든 post_id를 조회
        liked_post_interactions = db.query(UserInteraction.post_id).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.interaction_type == 'like'
        ).distinct().all()

        if not liked_post_interactions:
            return []

        liked_post_ids = [interaction.post_id for interaction in liked_post_interactions]

        # 2. 해당 post_id를 가진 Post 객체들을 조회
        liked_posts = db.query(Post).filter(Post.id.in_(liked_post_ids)).all()
        
        # 3. PostFeedResponse 모델로 변환 (is_liked=True로 설정)
        response_posts = []
        for post in liked_posts:
            post_response = PostFeedResponse.model_validate(post)
            post_response.is_liked = True # 이 페이지의 모든 포스트는 좋아요 상태
            response_posts.append(post_response)

        # 최신순으로 정렬해서 반환
        response_posts.sort(key=lambda p: p.timestamp, reverse=True)
        
        logger.info(f"Returning {len(response_posts)} liked posts for user {user_id}")
        return response_posts

    except Exception as e:
        logger.error(f"Error getting liked posts for user {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/api/posts/feed", response_model=List[PostFeedResponse])
async def get_personalized_feed(
    request: FeedRequest,
    db: Session = Depends(get_db)
):
    """개인화된 피드 반환"""
    try:
        user_id = request.user_id
        
        cache_key = f"feed:{user_id}:{request.page}"
        if redis_client:
            cached_feed = redis_client.get(cache_key)
            if cached_feed:
                logger.info(f"Cache HIT for user: {user_id}, page: {request.page}")
                parsed_cached_feed = json.loads(cached_feed)
                # Redis에서 가져온 JSON 데이터를 Pydantic 모델로 다시 로드
                return [PostFeedResponse.model_validate(item) for item in parsed_cached_feed]
        
        logger.info(f"Cache MISS for user: {user_id}, page: {request.page}. Generating new feed.")
        
        recommended_posts_db = await recommendation_engine.get_personalized_posts(
            user_id, request.page, request.limit, db
        )
        
        user_liked_posts_ids = {
            i.post_id for i in db.query(UserInteraction.post_id)
            .filter(UserInteraction.user_id == user_id, UserInteraction.interaction_type == 'like')
            .distinct().all()
        }

        response_posts: List[PostFeedResponse] = []
        for p_db in recommended_posts_db:
            p_response = PostFeedResponse.model_validate(p_db)
            p_response.is_liked = p_db.id in user_liked_posts_ids
            response_posts.append(p_response)
        
        if redis_client:
            # Pydantic 모델 리스트를 JSON 문자열로 변환할 때 model_dump(mode='json') 사용
            # 이는 datetime 객체를 ISO 8601 문자열로 자동 변환하여 JSON 직렬화 가능하게 합니다.
            final_response_json = json.dumps([p.model_dump(mode='json') for p in response_posts]) # 이 부분을 수정
            redis_client.setex(cache_key, 600, final_response_json) # 10분 캐시
            logger.info(f"Cached feed for user: {user_id}, page: {request.page}")
        
        return response_posts
    except Exception as e:
        logger.error(f"Error getting feed for user {request.user_id}, page {request.page}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/api/interactions/track")
async def track_interaction(
    request: InteractionRequest,
    db: Session = Depends(get_db)
):
    """사용자 상호작용 추적"""
    try:
        status_message = "success"
        interaction_id = None

        if request.interaction_type == 'like':
            existing_interaction = db.query(UserInteraction).filter(
                UserInteraction.user_id == request.user_id,
                UserInteraction.post_id == request.post_id,
                UserInteraction.interaction_type == 'like'
            ).first()
            if existing_interaction:
                status_message = "already liked"
                interaction_id = existing_interaction.id
            else:
                post_to_update = db.query(Post).filter(Post.id == request.post_id).first()
                if post_to_update:
                    post_to_update.likes += 1
                    db.add(post_to_update)
                    db.commit()
                    db.refresh(post_to_update)
                
                interaction = await behavior_tracker.track_interaction(
                    request.user_id, request.post_id, 'like', request.duration, db
                )
                interaction_id = interaction.id
                status_message = "success"

                # 좋아요/취소 시 해당 사용자의 모든 피드 캐시를 무효화
                if redis_client:
                    keys_to_delete = redis_client.keys(f"feed:{request.user_id}:*")
                    if keys_to_delete:
                        redis_client.delete(*keys_to_delete)
                        logger.info(f"Deleted feed cache for user {request.user_id}: {keys_to_delete}")
                
                # 비동기적으로 사용자 프로필 업데이트 (응답 시간에 영향 주지 않음)
                asyncio.create_task(recommendation_engine.update_user_profile(request.user_id, interaction, db))
                logger.info(f"User {request.user_id} liked post {request.post_id}.")

        elif request.interaction_type == 'unlike':
            existing_interaction = db.query(UserInteraction).filter(
                UserInteraction.user_id == request.user_id,
                UserInteraction.post_id == request.post_id,
                UserInteraction.interaction_type == 'like' # 'like' 타입 상호작용을 찾아서 삭제
            ).first()
            if existing_interaction:
                db.delete(existing_interaction)
                post_to_update = db.query(Post).filter(Post.id == request.post_id).first()
                if post_to_update and post_to_update.likes > 0:
                    post_to_update.likes -= 1
                    db.add(post_to_update)
                db.commit()
                status_message = "unliked"
                interaction_id = existing_interaction.id

                # 좋아요/취소 시 해당 사용자의 모든 피드 캐시를 무효화
                if redis_client:
                    keys_to_delete = redis_client.keys(f"feed:{request.user_id}:*")
                    if keys_to_delete:
                        redis_client.delete(*keys_to_delete)
                        logger.info(f"Deleted feed cache for user {request.user_id}: {keys_to_delete}")
                logger.info(f"User {request.user_id} unliked post {request.post_id}.")
            else:
                status_message = "not liked before"
                logger.info(f"User {request.user_id} tried to unlike post {request.post_id}, but it was not liked before.")

        else: # view, short_view, long_view, detail_view, share 등 기타 상호작용
            interaction = await behavior_tracker.track_interaction(
                request.user_id, request.post_id, request.interaction_type, request.duration, db
            )
            interaction_id = interaction.id
            status_message = "success"
            
            # 조회수 관련 상호작용 시에도 캐시를 무효화하여 즉시 반영
            if redis_client:
                keys_to_delete = redis_client.keys(f"feed:{request.user_id}:*")
                if keys_to_delete:
                    redis_client.delete(*keys_to_delete)
                    logger.info(f"Deleted feed cache for user {request.user_id}: {keys_to_delete} due to {request.interaction_type}.")

            asyncio.create_task(recommendation_engine.update_user_profile(request.user_id, interaction, db))
            logger.info(f"User {request.user_id} tracked interaction '{request.interaction_type}' on post {request.post_id}.")
        
        return {"status": status_message, "interaction_id": interaction_id}
    except Exception as e:
        logger.error(f"Error tracking interaction for user {request.user_id}, post {request.post_id}, type {request.interaction_type}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/api/posts/{post_id}", response_model=PostFeedResponse)
async def get_post_detail(post_id: str, user_id: str, db: Session = Depends(get_db)):
    """게시물 상세 정보"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        logger.warning(f"Post {post_id} not found.")
        raise HTTPException(status_code=404, detail="Post not found")
    
    # 상세 보기 상호작용 추적 (비동기 처리 가능)
    # 캐시 무효화는 여기서 하지 않음: 상세 보기는 피드 추천에 직접적인 영향을 주지 않으므로
    # 하지만 사용자 프로필 업데이트는 필요
    interaction = await behavior_tracker.track_interaction(
        user_id, post_id, "detail_view", None, db
    )
    asyncio.create_task(recommendation_engine.update_user_profile(user_id, interaction, db))
    logger.info(f"User {user_id} viewed detail for post {post_id}.")
    
    is_liked = db.query(UserInteraction).filter(
        UserInteraction.user_id == user_id,
        UserInteraction.post_id == post_id,
        UserInteraction.interaction_type == 'like'
    ).first() is not None
    
    post_response = PostFeedResponse.model_validate(post) # from_orm 대신 model_validate 사용
    post_response.is_liked = is_liked
    
    return post_response

# --- 알고리즘 초기화 API 엔드포인트 추가 ---
class ResetAlgorithmRequest(BaseModel):
    user_id: str

@app.post("/api/algorithm/reset")
async def reset_user_algorithm_data(
    request: ResetAlgorithmRequest,
    db: Session = Depends(get_db)
):
    """
    특정 사용자의 추천 알고리즘 관련 데이터를 초기화합니다.
    - UserProfile (선호도) 삭제/초기화
    - UserInteraction (상호작용 기록) 삭제
    - 해당 사용자의 Redis 캐시 삭제
    """
    try:
        # 1. UserProfile 삭제 (또는 초기화)
        user_profile = db.query(UserProfile).filter(UserProfile.user_id == request.user_id).first()
        if user_profile:
            db.delete(user_profile)
            logger.info(f"Deleted UserProfile for user: {request.user_id}")
        
        # 2. UserInteraction 삭제
        db.query(UserInteraction).filter(UserInteraction.user_id == request.user_id).delete()
        db.commit()
        logger.info(f"Deleted all UserInteractions for user: {request.user_id}")
        
        # 3. 해당 사용자의 Redis 캐시 삭제
        if redis_client:
            keys_to_delete = redis_client.keys(f"feed:{request.user_id}:*")
            if keys_to_delete:
                redis_client.delete(*keys_to_delete)
                logger.info(f"Deleted all Redis cache for user: {request.user_id}")

        # 모든 데이터를 삭제한 후, 기본 사용자 프로필을 다시 생성 (선택 사항)
        # 이렇게 하면 다음 요청 시 profile이 없어서 기본 선호도로 시작합니다.
        # 필요하다면 여기서 `UserProfile`을 다시 추가할 수 있습니다.
        # 예를 들어, `init_db.py`에서처럼 기본 프로필을 추가하는 로직을 여기에 포함할 수도 있습니다.
        # db.add(UserProfile(user_id=request.user_id))
        # db.commit()
        
        return {"status": "success", "message": f"Algorithm data reset for user: {request.user_id}"}
    except Exception as e:
        logger.error(f"Error resetting algorithm data for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reset algorithm data")