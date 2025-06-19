import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import asyncio
import random
import logging
import traceback

from database import Post, UserInteraction, UserProfile

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        self.content_vectorizer = TfidfVectorizer(max_features=500)
        self.collaborative_model = TruncatedSVD(n_components=20) 
        self.topic_weights = {
            "기술": 1.0, "일상": 1.0, "정치": 1.0,
            "경제": 1.0, "여행": 1.0, "교육": 1.0,
            "사회": 1.0, "문화": 1.0, "예술": 1.0,
            "스포츠": 1.0, "건강": 1.0, "음식": 1.0,
            "환경": 1.0, "과학": 1.0, "역사": 1.0,
            "IT": 1.0, "엔터테인먼트": 1.0, "라이프스타일": 1.0,
            "비즈니스": 1.0, "법률": 1.0, "부동산": 1.0,
            "자동차": 1.0, "패션": 1.0, "뷰티": 1.0,
            "음악": 1.0, "영화": 1.0, "도서": 1.0,
            "게임": 1.0, "디자인": 1.0, "육아": 1.0,
            "반려동물": 1.0, "재테크": 1.0, "세계": 1.0 
        }
        self.posts_df = None
        self.tfidf_matrix = None

    async def _load_and_prepare_posts(self, db: Session):
        """DB에서 게시물을 로드하고 TF-IDF 매트릭스를 미리 계산합니다."""
        if self.posts_df is None or self.posts_df.empty:
            all_posts = db.query(Post).all()
            if not all_posts:
                logger.info("No posts found in the database. recommendation_engine will return empty results.")
                self.posts_df = None
                self.tfidf_matrix = None
                return
            
            self.posts_df = pd.DataFrame([p.__dict__ for p in all_posts])
            self.posts_df.set_index('id', inplace=True)
            
            if 'timestamp' in self.posts_df.columns and pd.api.types.is_datetime64tz_dtype(self.posts_df['timestamp']):
                self.posts_df['timestamp'] = pd.to_datetime(self.posts_df['timestamp']).dt.tz_localize(None)

            post_contents = [f"{row.content} {row.topic}" for index, row in self.posts_df.iterrows()]
            if not post_contents:
                self.tfidf_matrix = None
                logger.warning("No post content available for TF-IDF vectorization.")
            else:
                self.tfidf_matrix = self.content_vectorizer.fit_transform(post_contents)
                logger.info(f"Loaded {len(self.posts_df)} posts and computed TF-IDF matrix.")
        else:
            logger.debug("Posts and TF-IDF matrix already loaded.")


    async def get_personalized_posts(self, user_id: str, page: int, limit: int, db: Session) -> List[Post]:
        """개인화된 게시물 추천"""
        await self._load_and_prepare_posts(db)
        if self.posts_df is None or self.posts_df.empty or self.tfidf_matrix is None:
            logger.warning("Posts DataFrame or TF-IDF matrix is not initialized. Returning empty list.")
            return []

        user_profile = await self._get_user_profile(user_id, db)
        
        # 1. 컨텐츠 기반 필터링
        content_scores = await self._content_based_filtering(user_profile)
        
        # 2. 협업 필터링 (데이터가 충분할 때만)
        interactions_count = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).count()
        
        collaborative_scores = pd.Series(0.0, index=self.posts_df.index)
        if interactions_count >= 10:
            collaborative_scores = await self._collaborative_filtering(user_id, db)
        else:
            logger.info(f"User {user_id} has only {interactions_count} interactions. Skipping collaborative filtering.")
        
        # 3. 시간 기반 가중치
        time_scores = await self._time_based_scoring()
        
        # 4. 인기도 기반 가중치
        popularity_scores = await self._popularity_based_scoring()
        
        # 최종 점수 계산 (하이브리드)
        common_index = self.posts_df.index
        content_scores = content_scores.reindex(common_index).fillna(0)
        collaborative_scores = collaborative_scores.reindex(common_index).fillna(0)
        time_scores = time_scores.reindex(common_index).fillna(0)
        popularity_scores = popularity_scores.reindex(common_index).fillna(0)

        final_scores = (
            content_scores * 0.4 +
            collaborative_scores * 0.2 +
            time_scores * 0.2 +
            popularity_scores * 0.2
        )
        
        # 사용자가 이미 '본' 상호작용만 제외합니다. '좋아요'는 긍정적인 상호작용이므로 제외하지 않습니다.
        seen_posts = {
            i.post_id for i in db.query(UserInteraction.post_id)
            .filter(UserInteraction.user_id == user_id)
            .filter(UserInteraction.interaction_type.in_(['view', 'short_view', 'long_view', 'detail_view']))
            .distinct().all()
        }
        
        # 필터링 및 페이지네이션
        offset = page * limit
        
        # 1. 추천 알고리즘에 의해 정렬된 모든 게시물 ID 가져오기 (점수 내림차순)
        all_ranked_post_ids = final_scores.sort_values(ascending=False).index.tolist()
        
        # 2. 이미 본 게시물을 제외하고 추천 후보 게시물 리스트 생성
        unseen_ranked_post_ids = [pid for pid in all_ranked_post_ids if pid not in seen_posts]

        # 3. 현재 페이지에서 필요한 만큼의 알고리즘 추천 게시물 가져오기
        recommended_for_page = unseen_ranked_post_ids[offset : offset + limit]
        
        # 4. 추천 게시물 수가 limit보다 적을 경우, 보충 전략 사용
        if len(recommended_for_page) < limit:
            logger.info(f"Not enough algorithmically recommended posts for user {user_id} (found {len(recommended_for_page)}). Attempting to fill with supplementary posts.")
            
            num_to_fill = limit - len(recommended_for_page)
            
            # 모든 게시물 ID 가져오기 (posts_df의 인덱스는 이미 고유한 게시물 ID입니다)
            all_available_post_ids = self.posts_df.index.tolist()
            
            # 사용자가 아직 보지 않은 전체 게시물 목록
            # 기존 추천 목록에 이미 포함된 게시물도 제외
            all_unseen_posts_except_current_rec = [
                pid for pid in all_available_post_ids 
                if pid not in seen_posts and pid not in recommended_for_page
            ]
            
            # 보충 게시물 선택 (여기서는 무작위로 선택)
            # 필요에 따라 '최신 게시물', '가장 인기 있는 게시물' 등으로 대체 가능
            random.shuffle(all_unseen_posts_except_current_rec)
            supplementary_posts = all_unseen_posts_except_current_rec[:num_to_fill]
            
            # 최종 게시물 목록 결합
            final_post_ids_combined = list(recommended_for_page) + list(supplementary_posts)
            random.shuffle(final_post_ids_combined) # 최종적으로 섞어서 반환

            # 최종 목록에서 현재 페이지에 필요한 만큼만 잘라서 반환
            top_post_ids = final_post_ids_combined[:limit] 
            
            logger.info(f"Filled with {len(supplementary_posts)} supplementary posts. Total posts for user {user_id}: {len(top_post_ids)}")
        else:
            top_post_ids = recommended_for_page
            logger.info(f"User {user_id} (interactions: {interactions_count}, page: {page}) received score-based recommendations.")
        
        if not top_post_ids:
            logger.info(f"No posts found for user {user_id} after filtering and scoring. Returning empty list.")
            return []
            
        post_ids_to_fetch = top_post_ids
        
        fetched_posts = db.query(Post).filter(Post.id.in_(post_ids_to_fetch)).all()
        
        # 원본 요청 순서대로 게시물을 정렬 (매우 중요)
        post_id_to_post = {p.id: p for p in fetched_posts}
        ordered_posts = [post_id_to_post[pid] for pid in post_ids_to_fetch if pid in post_id_to_post]

        logger.info(f"Returned {len(ordered_posts)} posts for user {user_id}, page {page}.")
        return ordered_posts

    async def _get_user_profile(self, user_id: str, db: Session) -> Dict:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile or not profile.preferences:
            logger.info(f"User profile for {user_id} not found or empty. Returning default preferences.")
            return {'topic_preferences': self.topic_weights.copy()}
        return profile.preferences

    async def _content_based_filtering(self, user_profile: Dict) -> pd.Series:
        """컨텐츠 기반 필터링"""
        user_topic_prefs = user_profile.get('topic_preferences', self.topic_weights)
        
        user_interests = " ".join([topic for topic, weight in user_topic_prefs.items() if weight > 0 for _ in range(max(1, int(weight * 10)))])

        if not user_interests or self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            logger.warning("User interests or TF-IDF matrix is empty/not ready for content-based filtering. Returning default (low) scores.")
            return pd.Series(0.1, index=self.posts_df.index)

        try:
            user_vector = self.content_vectorizer.transform([user_interests])
            similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
            max_sim = similarities.max()
            min_sim = similarities.min()
            if (max_sim - min_sim) > 1e-9:
                scaled_similarities = (similarities - min_sim) / (max_sim - min_sim)
            else:
                scaled_similarities = np.full_like(similarities, 0.5)

            return pd.Series(scaled_similarities, index=self.posts_df.index)
        except ValueError as e:
            logger.error(f"Content vectorizer transform failed: {e}. Traceback: {traceback.format_exc()}. Returning default scores.")
            return pd.Series(0.1, index=self.posts_df.index)


    async def _collaborative_filtering(self, user_id: str, db: Session) -> pd.Series:
        """협업 필터링"""
        interactions = db.query(UserInteraction.user_id, UserInteraction.post_id, UserInteraction.engagement_score).limit(10000).all()
        
        if len(interactions) < self.collaborative_model.n_components + 1:
            logger.warning(f"Not enough interactions ({len(interactions)}) for collaborative filtering SVD. Returning 0 scores.")
            return pd.Series(0.0, index=self.posts_df.index)

        interaction_df = pd.DataFrame(interactions, columns=['user_id', 'post_id', 'engagement_score'])
        
        valid_post_ids = self.posts_df.index.unique()
        interaction_df = interaction_df[interaction_df['post_id'].isin(valid_post_ids)]
        
        if interaction_df.empty:
            logger.warning("Interaction DataFrame is empty after filtering for valid posts. Returning 0 scores.")
            return pd.Series(0.0, index=self.posts_df.index)

        user_item_matrix = pd.pivot_table(interaction_df, index='user_id', columns='post_id', values='engagement_score', fill_value=0)
        user_item_matrix = user_item_matrix.reindex(columns=self.posts_df.index.unique(), fill_value=0)


        if user_id not in user_item_matrix.index:
            logger.warning(f"User {user_id} not in user-item matrix. Cannot perform collaborative filtering. Returning 0 scores.")
            return pd.Series(0.0, index=self.posts_df.index)

        try:
            if user_item_matrix.shape[0] < self.collaborative_model.n_components or \
               user_item_matrix.shape[1] < self.collaborative_model.n_components:
                logger.warning(f"User-item matrix too small for SVD (shape: {user_item_matrix.shape}). n_components={self.collaborative_model.n_components}. Returning 0 scores.")
                return pd.Series(0.0, index=self.posts_df.index)

            user_factors = self.collaborative_model.fit_transform(user_item_matrix)
            item_factors = self.collaborative_model.components_.T
            
            user_idx = list(user_item_matrix.index).index(user_id)
            user_vector = user_factors[user_idx]
            
            scores = np.dot(item_factors, user_vector)
            scores = np.maximum(0, scores)
            
            max_score = scores.max()
            if max_score > 0:
                scaled_scores = scores / max_score
            else:
                scaled_scores = scores

            return pd.Series(scaled_scores, index=user_item_matrix.columns).reindex(self.posts_df.index).fillna(0)
        except Exception as e:
            logger.error(f"Collaborative filtering failed for user {user_id}: {e}. Traceback: {traceback.format_exc()}. Returning 0 scores.")
            return pd.Series(0.0, index=self.posts_df.index)

    async def _time_based_scoring(self) -> pd.Series:
        """시간 기반 점수 - 최신 게시물에 가중치"""
        now = datetime.utcnow()
        posts_timestamps_naive = self.posts_df['timestamp']
        if pd.api.types.is_datetime64tz_dtype(posts_timestamps_naive):
            posts_timestamps_naive = posts_timestamps_naive.dt.tz_localize(None)

        time_diff_days = (now - posts_timestamps_naive).dt.total_seconds() / (3600 * 24)
        time_scores = np.exp(-time_diff_days / 14.0) 
        return time_scores

    async def _popularity_based_scoring(self) -> pd.Series:
        """인기도 기반 점수 (좋아요+댓글+공유)"""
        engagement = (self.posts_df['likes'] * 1.0 + self.posts_df['comments_count'] * 2.0 + self.posts_df['shares'] * 3.0)
        
        if engagement.max() == 0:
            return pd.Series(0.0, index=self.posts_df.index)
        
        log_engagement = np.log1p(engagement)
        
        max_log_engagement = log_engagement.max()
        if max_log_engagement == 0:
            return pd.Series(0.0, index=self.posts_df.index)
        return log_engagement / max_log_engagement

    async def update_user_profile(self, user_id: str, interaction: UserInteraction, db: Session):
        """사용자 상호작용 기반 프로필 실시간 업데이트"""
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        
        if not profile:
            profile = UserProfile(user_id=user_id, preferences={'topic_preferences': self.topic_weights.copy()})
            db.add(profile)
            db.flush()
            logger.info(f"Created new user profile for {user_id}")
        
        post = db.query(Post).filter(Post.id == interaction.post_id).first()
        if post:
            current_prefs = profile.preferences.get('topic_preferences', self.topic_weights.copy())
            
            if interaction.interaction_type == 'like':
                current_prefs[post.topic] = min(5.0, current_prefs.get(post.topic, 1.0) + 0.3)
            elif interaction.interaction_type == 'comment':
                current_prefs[post.topic] = min(5.0, current_prefs.get(post.topic, 1.0) + 0.4)
            elif interaction.interaction_type == 'share':
                current_prefs[post.topic] = min(5.0, current_prefs.get(post.topic, 1.0) + 0.5)
            elif interaction.interaction_type == 'long_view':
                current_prefs[post.topic] = min(5.0, current_prefs.get(post.topic, 1.0) + 0.15)
            elif interaction.interaction_type == 'detail_view':
                current_prefs[post.topic] = min(5.0, current_prefs.get(post.topic, 1.0) + 0.1)
            elif interaction.interaction_type == 'view':
                   current_prefs[post.topic] = min(5.0, current_prefs.get(post.topic, 1.0) + 0.05)
            elif interaction.interaction_type == 'short_view':
                current_prefs[post.topic] = max(0.1, current_prefs.get(post.topic, 1.0) - 0.1) 
            
            for topic in current_prefs:
                current_prefs[topic] = max(0.1, current_prefs[topic] * 0.99)

            profile.preferences = {'topic_preferences': current_prefs}
            db.commit()
            logger.info(f"Updated user profile for {user_id} with interaction {interaction.interaction_type} on post {post.id}. New preferences: {current_prefs.get(post.topic, 'N/A'):.2f}")
        else:
            logger.warning(f"Post {interaction.post_id} not found for updating user profile {user_id}. Profile not updated.")